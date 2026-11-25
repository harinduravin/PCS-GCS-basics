import tensorflow as tf
from tensorflow.keras import Model
import pn_classes
from pn_classes import PhaseNoise, PN_Compensation, PTRS_Adder
import numpy as np
import h5py
from scipy.sparse import csr_matrix
import ccdm
from ccdm_initialize import initialize
from sionna.phy.fec.interleaving import RandomInterleaver, Deinterleaver
from sionna.phy.utils import ebnodb2no
from plotting import PlotBER
from sionna.phy.mapping import Constellation, Mapper, Demapper, BinarySource
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder, LDPCBPDecoder
from sionna.phy.channel import AWGN
import sionna as sn
from end_to_end_model import generate_ldpc_H_matrix, symbol_points_qam64
import os
import matplotlib.pyplot as plt
import time

start_time = time.perf_counter()


class BICM_scheme(Model):

    def __init__(self, batch_size, num_bp_iter, use_dvb, symbol_points):
        super().__init__()

        self.use_dvb = use_dvb
        self.batch_size = batch_size
        llr_max_value = 1000.00
        self.channel = AWGN()
        self.num_bits_per_symbol = 6

        if self.use_dvb:
            r = 3 / 5
            n_ldpc = 16200
            self.k_ldpc = int(n_ldpc * r)

            parity_matrix = generate_ldpc_H_matrix(n_ldpc=n_ldpc, rate=r, q=18, file_path="ldpccode_short_3_5.txt")
            H_csr = csr_matrix(parity_matrix[:, :self.k_ldpc])
            self.H_sparse = tf.sparse.SparseTensor(
                indices=np.vstack((H_csr.nonzero()[0], H_csr.nonzero()[1])).T.astype(np.int64),
                values=H_csr.data.astype(np.int32),
                dense_shape=H_csr.shape
            )
            self.H_sparse_test = csr_matrix(parity_matrix[:, :self.k_ldpc])
            batch_size_tf = tf.constant(batch_size, dtype=tf.int64)
            self.decoder = LDPCBPDecoder(parity_matrix,  # connect encoder (for shared code parameters)
                                         cn_update="boxplus-phi",  # use the exact boxplus function
                                         num_iter=num_bp_iter,
                                         llr_max=llr_max_value,
                                         hard_out=False)
            self.source = BinarySource()

        else:
            self.k_ldpc = 3069
            n_ldpc = 4092

            self.source = BinarySource()
            self.encoder = LDPC5GEncoder(self.k_ldpc, n_ldpc)
            self.siso_decoder = LDPC5GDecoder(self.encoder, return_infobits=False, num_iter=num_bp_iter, hard_out=False,
                                              llr_max=llr_max_value, return_state=True)
            self.decoder = LDPC5GDecoder(self.encoder, return_infobits=True, hard_out=True, num_iter=num_bp_iter,
                                         llr_max=llr_max_value, return_state=False)

        # initialize a random interleaver and corresponding deinterleaver
        self.interleaver = RandomInterleaver()
        self.deinterleaver = Deinterleaver(self.interleaver)

        probabilities = np.ones_like(symbol_points) / (2 ** self.num_bits_per_symbol)
        power = np.sum(probabilities * np.abs(symbol_points) ** 2)
        normalized_points = symbol_points / np.sqrt(power)
        constellation = Constellation("custom", num_bits_per_symbol=6, points=normalized_points)

        # mapper and demapper
        self.mapper = Mapper(constellation=constellation)
        self.demapper = Demapper("app", constellation=constellation)  # APP demapper
        self.num_symbols = int(n_ldpc / self.num_bits_per_symbol)

        self.upsampler = sn.phy.signal.Upsampling(samples_per_symbol=4)
        self.tx_filter = sn.phy.signal.RootRaisedCosineFilter(beta=0.3,
                                                              samples_per_symbol=4,
                                                              span_in_symbols=32)
        self.rx_filter = sn.phy.signal.RootRaisedCosineFilter(beta=0.3,
                                                              samples_per_symbol=4,
                                                              span_in_symbols=32)

        num_ptrs_blocks = 32
        symbols_per_ptrs_block = 4
        block_size_ptrs = int(n_ldpc/self.num_bits_per_symbol) + 128
        bandwidth = 960e3 * 4096
        fc = 120e9
        samples = 4 * (block_size_ptrs + 32)

        self.downsampler = sn.phy.signal.Downsampling(samples_per_symbol=4,
                                                      offset=self.rx_filter.length - 1,
                                                      num_symbols=tf.cast(block_size_ptrs, tf.int32))

        path_to_mat = "PreloadedMats/ptrs_CAZAC_4_32.mat"

        self.ptrs_adder = PTRS_Adder(path_to_mat, num_ptrs_blocks, symbols_per_ptrs_block, block_size_ptrs, batch_size)
        self.pn_compensate = PN_Compensation(self.ptrs_adder.ptrs_indices, self.ptrs_adder.data_indices,
                                             block_size_ptrs, self.ptrs_adder.pilots, batch_size)
        self.pn_tx = PhaseNoise(model_type='13', bandwidth=bandwidth, fc=fc, n_samples=samples)
        self.pn_rx = PhaseNoise(model_type='11', bandwidth=bandwidth, fc=fc, n_samples=samples)

    # @tf.function  # Graph execution to speed things up
    def __call__(self, batch_size, ebno_db):
        # calculate noise variance
        no = ebnodb2no(ebno_db,
                       num_bits_per_symbol=self.num_bits_per_symbol,
                       coderate=3 / 5)

        if self.use_dvb:
            b_stream = tf.cast(self.source([batch_size, self.k_ldpc]), tf.int32)  # generate random data
            p_bits = tf.transpose(tf.sparse.sparse_dense_matmul(self.H_sparse, tf.transpose(b_stream)))
            p_bits = tf.math.mod(p_bits, 2)
            p_bits = tf.math.mod(tf.math.cumsum(p_bits, axis=1), 2)
            codeword = tf.concat([b_stream, p_bits], axis=1)
        else:
            b_stream = tf.cast(self.source([batch_size, self.k_ldpc]), tf.float32)  # generate random data
            codeword = tf.cast(self.encoder(b_stream), tf.int32)

        c_int = self.interleaver(codeword)
        x = self.mapper(c_int)

        #######################
        # print(x.shape)
        ptrs_x_mod = self.ptrs_adder(x)

        x_up = self.upsampler(ptrs_x_mod)
        x_tx_filt = self.tx_filter(x_up)
        pn_out, _, _ = self.pn_tx(x_tx_filt)
        y_channel = self.channel(pn_out, no)
        y_channel_pn, _, _ = self.pn_rx(y_channel)
        y_rx_filt = self.rx_filter(y_channel_pn)
        y = self.downsampler(y_rx_filt)
        # print('y.shape: ',y.shape)
        _, y_data_vec = self.pn_compensate(y)

        #######################

        # # mean_squared = tf.reduce_mean(tf.abs(x) ** 2)
        # # print('meansquare',mean_squared.numpy())
        # y = self.channel(x, no) # transmit over AWGN channel

        llr_ch = self.demapper(y_data_vec, no)  # demapp
        llr_deint = self.deinterleaver(llr_ch)
        llr_u_stream = self.decoder(llr_deint)  # run LDPC decoder

        u_stream_hat = tf.cast(tf.greater_equal(0.0, -llr_u_stream), tf.int32)
        b_stream_hat = u_stream_hat[:, :self.k_ldpc]
        # tf.print(b_stream_hat.dtype)
        # tf.print(b_stream.dtype)
        return b_stream, b_stream_hat


class PAS_scheme(Model):

    def __init__(self, batch_size, num_bp_iter, const_symbol_points, p_basic):
        super().__init__()

        self.batch_size = 1
        self.num_bits_per_symbol = 6
        self.parity_k = 2  # Two bits represent parity bits, other four represent information

        r = 2 / 3  # m = 6 so (m-2)/m rate code is required here
        self.k_ldpc = 10800
        n_ldpc = 16200
        self.n_out = 2700

        self.binary_source = BinarySource()

        parity_matrix = generate_ldpc_H_matrix(n_ldpc=n_ldpc, rate=r, q=15, file_path="ldpccode_short_2_3.txt")
        self.H_sparse = csr_matrix(parity_matrix[:, :self.k_ldpc])

        # p_basic = np.array([0.00622466, 0.01608852, 0.03347011, 0.02951173, 0.01565116, 0.03419713,
        #  0.07408216, 0.06683832, 0.03084535, 0.06940064, 0.14336276, 0.12366232,
        #  0.02854129, 0.06747344, 0.13703237, 0.12361801])

        [p_quant, self.k_input, self.n_i] = initialize(p_basic, self.n_out)

        probabilities = np.tile(p_basic, (1, (2 ** self.parity_k))) / (2 ** self.parity_k)
        power = np.sum(probabilities * np.abs(const_symbol_points) ** 2)
        normalized_points = const_symbol_points / np.sqrt(power)

        self.channel = AWGN()
        constellation = Constellation("custom", num_bits_per_symbol=6, points=normalized_points)
        self.mapper = Mapper(constellation=constellation)
        self.demapper = Demapper("app", constellation=constellation)  # APP demapper

        num_bp_iter = 50
        llr_max_value = 120

        self.decoder = LDPCBPDecoder(parity_matrix,  # connect encoder (for shared code parameters)
                                     cn_update="boxplus-phi",  # use the exact boxplus function
                                     num_iter=num_bp_iter,
                                     llr_max=llr_max_value,
                                     hard_out=True)

        bit_labels = np.array([[int(b) for b in format(i, f'0{6}b')] for i in range(64)])
        p_b1 = np.reshape(np.matmul(probabilities, bit_labels), (-1))
        self.prior = np.log(p_b1 / (1.0 - p_b1))

        self.bit_positions = np.arange((self.num_bits_per_symbol - self.parity_k) - 1, -1, -1)

        self.upsampler = sn.phy.signal.Upsampling(samples_per_symbol=4)
        self.tx_filter = sn.phy.signal.RootRaisedCosineFilter(beta=0.3,
                                                              samples_per_symbol=4,
                                                              span_in_symbols=32)
        self.rx_filter = sn.phy.signal.RootRaisedCosineFilter(beta=0.3,
                                                              samples_per_symbol=4,
                                                              span_in_symbols=32)

        num_ptrs_blocks = 32
        symbols_per_ptrs_block = 4
        block_size_ptrs = self.n_out + 4 * 32
        bandwidth = 960e3 * 4096
        fc = 120e9
        samples = 4 * (block_size_ptrs + 32)

        self.downsampler = sn.phy.signal.Downsampling(samples_per_symbol=4,
                                                      offset=self.rx_filter.length - 1,
                                                      num_symbols=tf.cast(block_size_ptrs, tf.int32))

        path_to_mat = "PreloadedMats/ptrs_CAZAC_4_32.mat"

        self.ptrs_adder = PTRS_Adder(path_to_mat, num_ptrs_blocks, symbols_per_ptrs_block, block_size_ptrs, batch_size)
        self.pn_compensate = PN_Compensation(self.ptrs_adder.ptrs_indices, self.ptrs_adder.data_indices,
                                             block_size_ptrs, self.ptrs_adder.pilots, batch_size)
        self.pn_tx = PhaseNoise(model_type='13', bandwidth=bandwidth, fc=fc, n_samples=samples)
        self.pn_rx = PhaseNoise(model_type='11', bandwidth=bandwidth, fc=fc, n_samples=samples)

    # No Graph execution possible due to sequential CCDM Encode and Decode
    # Due to this no attempt was made to use tensorflow implementation
    def __call__(self, batch_size, ebno_db):
        no = ebnodb2no(ebno_db,
                       num_bits_per_symbol=3.62,
                       coderate=1)

        b_stream = np.array(self.binary_source([self.k_input]), np.int64)
        code_symbols = np.array(ccdm.encode(b_stream, self.n_out, self.n_i))
        reverse_map = (((code_symbols[:, None] - 1) & (
                    1 << np.arange(self.num_bits_per_symbol - self.parity_k)[::-1])) > 0).astype(int)
        enc_input = np.reshape(reverse_map, ((self.num_bits_per_symbol - self.parity_k) * self.n_out))
        parity_bits = self.H_sparse.dot(enc_input) % 2  # mod 2 because XOR over GF(2)
        parity_bits = np.mod(np.cumsum(parity_bits), 2)
        parity_reshaped = np.reshape(parity_bits, (-1, self.parity_k))
        mapper_input_matrix = np.concatenate((parity_reshaped, reverse_map), axis=1)
        mapper_input = np.reshape(mapper_input_matrix, (self.num_bits_per_symbol * self.n_out))
        x = self.mapper(mapper_input)

        #########################

        ptrs_x_mod = self.ptrs_adder(tf.expand_dims(x, axis=0))

        x_up = self.upsampler(ptrs_x_mod)
        x_tx_filt = self.tx_filter(x_up)
        pn_out, _, _ = self.pn_tx(x_tx_filt)
        y_channel = self.channel(pn_out, no)
        y_channel_pn, _, _ = self.pn_rx(y_channel)
        y_rx_filt = self.rx_filter(y_channel_pn)
        y = self.downsampler(y_rx_filt)
        # print('y.shape: ',y.shape)
        _, y_data_vec = self.pn_compensate(y)

        #########################
        # # mean_squared = tf.reduce_mean(tf.abs(x) ** 2)
        # # print('meansquare',mean_squared.numpy())
        # y = self.channel(x, no)
        llr_ch = self.demapper(y_data_vec, no, self.prior)
        llr_reshaped = np.reshape(llr_ch, (-1, self.num_bits_per_symbol))
        llr_parity_comp = np.reshape(llr_reshaped[:, :self.parity_k], (-1))
        llr_amplitude_comp = np.reshape(llr_reshaped[:, self.parity_k:], (-1))
        codeword_llr = np.hstack((llr_amplitude_comp, llr_parity_comp))
        decoded_bits = tf.cast(self.decoder(codeword_llr), tf.int64).numpy()
        info_bits_reshaped = np.reshape(decoded_bits[:self.k_ldpc], (-1, self.num_bits_per_symbol - self.parity_k))
        # Use bitwise left shift to compute powers of two
        shifted_bits = info_bits_reshaped << self.bit_positions
        # Compute the integer representation using bitwise operations
        code_symbols_retrieved = np.sum(shifted_bits, axis=-1) + 1
        # tf.print('ccdm input: ', code_symbols_retrieved.shape)
        decoded_bits_ccdmdec = ccdm.decode(code_symbols_retrieved, self.n_i, self.k_input)
        missing = self.k_input - len(decoded_bits_ccdmdec)
        if missing > 0:
            decoded_bits_ccdmdec.extend([0] * missing)
        # tf.print('ccdm :',np.array(decoded_bits_ccdmdec).shape)
        # tf.print('b_stream :',b_stream.shape)

        return b_stream, decoded_bits_ccdmdec


RESULTS_DIR = "results_gcs_benchmark"

# snr_list = [13]
# papr_list = [5.5, 6.0, 6.5, 7.0]

batch_size = 40
num_bp_iter = 50
use_dvb = False

ber_plot_bicm = PlotBER("ber_plot_bicm")

# for snr_db in snr_list:
#     for papr_db in papr_list:
#         exp_name = f"snr_{snr_db}_papr_{papr_db}"
#         plotname = f"papr_{papr_db}"
#         data = np.load(f"{RESULTS_DIR}/{exp_name}.npz")
#         points = data["constellation_points"]

model_bicm = BICM_scheme(batch_size, num_bp_iter, use_dvb, symbol_points_qam64)

ber_plot_bicm.simulate(model_bicm,  # the function have defined previously
                       ebno_dbs=np.arange(5.0, 13.0, 0.1),  # sim SNR range
                       legend='Test BICM',
                       add_bler=True,
                       add_ber=True,
                       max_mc_iter=250,
                       num_target_block_errors=100,
                       batch_size=batch_size,
                       soft_estimates=False,
                       early_stop=True,
                       show_fig=False,
                       forward_keyboard_interrupt=False)

bicm_ber = ber_plot_bicm.ber[0]
bicm_bler = ber_plot_bicm.ber[1]
bicm_snr = ber_plot_bicm.snr

end_time = time.perf_counter()

elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.6f} seconds")

# =========================
# Create output folder
# =========================
output_folder = "output_plots"
os.makedirs(output_folder, exist_ok=True)

# =========================
# Save data as NPZ
# =========================
np.savez("bicm_benchmark.npz")

# =========================
# Plot BER
# =========================
plt.figure()
plt.semilogy(bicm_snr, bicm_ber, 'o-', label="BICM")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.title("BER vs SNR")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "BER_plot.png"), dpi=300)
plt.close()

# =========================
# Plot BLER
# =========================
plt.figure()
plt.semilogy(bicm_snr, bicm_bler, 'o-', label="BICM")
plt.xlabel("SNR (dB)")
plt.ylabel("BLER")
plt.title("BLER vs SNR")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "BLER_plot.png"), dpi=300)
plt.close()
