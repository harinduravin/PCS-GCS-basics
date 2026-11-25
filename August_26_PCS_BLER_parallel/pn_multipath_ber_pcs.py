import os
import sys


import tensorflow as tf
from tensorflow.keras import Model
import pn_classes
from pn_classes import PhaseNoise, PN_Compensation, PTRS_Adder
from custom_classes_shaping import Demapper_optimized
import numpy as np
import scipy.linalg
from scipy.sparse import csr_matrix
import ccdm
from ccdm_initialize import initialize
from sionna.phy.fec.interleaving import RandomInterleaver, Deinterleaver
from sionna.phy.utils import ebnodb2no
from plotting import PlotBER
from sionna.phy.mapping import Constellation, Mapper, Demapper, BinarySource, SymbolLogits2LLRs
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder, LDPCBPDecoder
from sionna.phy.channel import AWGN
from sionna.phy.block import Block
from sionna.phy.channel.tr38901 import TDL
from tensorflow.signal import ifftshift
from sionna.phy.signal import ifft
from sionna.phy.ofdm import OFDMDemodulator
import sionna as sn
from end_to_end_model import generate_ldpc_H_matrix, symbol_points_qam64
import matplotlib.pyplot as plt
import time

class PAS_scheme(Model):

    def __init__(self, batch_size, num_bp_iter, const_symbol_points, p_basic, p0,  num_bits_per_symbol, use_upsampling_filtering_pn, use_neural_demapper, pn_freq, condition):
        super().__init__()

        #####################
        labels = np.arange(2 ** num_bits_per_symbol)
        bin_labels = ((labels[:, None] & (1 << np.arange(num_bits_per_symbol - 1, -1, -1))) > 0).astype(int)
        first_bits = bin_labels[:, 0]  # First (MSB) bit
        probabilities = tf.where(first_bits == 0, p0 / (2 ** (num_bits_per_symbol - 1)),
                                 (1 - p0) / (2 ** (num_bits_per_symbol - 1)))

        p_basic_mask = (bin_labels[:, 2] == 0) & (bin_labels[:, 3] == 0)
        p_basic = probabilities[p_basic_mask] * 4
        ########################

        self.batch_size = 1
        self.num_bits_per_symbol = num_bits_per_symbol
        self.N = 2**num_bits_per_symbol
        self.parity_k = 2 # Two bits represent parity bits, other four represent information

        r = 2/3   # m = 6 so (m-2)/m rate code is required here

        n_ldpc = 16200
        self.k_ldpc = int(n_ldpc*r)
        self.n_out = int(self.k_ldpc/(self.num_bits_per_symbol - self.parity_k))

        self.binary_source = BinarySource()



        parity_matrix = generate_ldpc_H_matrix(n_ldpc=n_ldpc,rate=r,q=15,file_path="ldpccode_short_2_3.txt")
        self.H_sparse = csr_matrix(parity_matrix[:, :self.k_ldpc])

        [p_quant,self.k_input,self.n_i] = initialize(p_basic,self.n_out)

        # probabilities = np.tile(p_basic,(1,(2**self.parity_k)))/(2**self.parity_k)
        power = np.sum(probabilities * np.abs(const_symbol_points)**2)
        self.normalized_points = const_symbol_points / np.sqrt(power)

        self.channel = AWGN()

        constellation = Constellation("custom", num_bits_per_symbol=num_bits_per_symbol, points = self.normalized_points)
        self.mapper = Mapper(constellation=constellation)

        if pn_freq == '220':
            fc = 220e9
            pn_var = 0.0108
        elif pn_freq == '120':
            fc = 120e9
            pn_var = 0.0032

        self.use_neural_demapper = use_neural_demapper
        if not(use_neural_demapper):
            # self.demapper = Demapper("app", constellation=constellation) # APP demapper
            self.demapper = Demapper_optimized("app", constellation=constellation, pn_var=pn_var, condition=condition) # APP demapper
        else:
            # self.demapper1 = Demapper("app", constellation=constellation) # APP demapper
            # learned_pn_points = np.load(f"../const_shaping_symbol/results_32point_pn_220GHz/32point_const_points.npy")
            demodulator_weights = np.load("../const_shaping_symbol/results_32point_pn_220GHz/32point_const_demodulator_weights.npz")
            self.demapper = Demodulator_bitwise(2**num_bits_per_symbol, 10.0)
            # Dummy forward pass to trigger build
            dummy_x = tf.zeros((1, 5), dtype=tf.float32)
            _ = self.demapper(dummy_x)
            self.demapper.set_weights([
                demodulator_weights["weight_0"],
                demodulator_weights["weight_1"],
                demodulator_weights["weight_2"],
                demodulator_weights["weight_3"],
                demodulator_weights["weight_4"],
                demodulator_weights["weight_5"]
            ])

        num_bp_iter = 50
        llr_max_value = 120

        self.decoder = LDPCBPDecoder(parity_matrix, # connect encoder (for shared code parameters)
                                cn_update="boxplus-phi", # use the exact boxplus function
                                num_iter=num_bp_iter,
                                llr_max=llr_max_value,
                               hard_out=True)

        bit_labels = np.array([[int(b) for b in format(i, f'0{num_bits_per_symbol}b')] for i in range(self.N)])
        p_b1 = np.reshape(np.matmul(probabilities,bit_labels),(-1))
        self.prior = np.log(p_b1 / (1.0 - p_b1))

        self.bit_positions = np.arange((self.num_bits_per_symbol-self.parity_k)-1, -1, -1)


        self.mask = np.ones(self.num_bits_per_symbol, bool)
        self.mask[2:4] = False  # exclude parity positions

        self.use_upsampling_filtering_pn  = use_upsampling_filtering_pn

        if use_upsampling_filtering_pn:
            self.upsampler = sn.phy.signal.Upsampling(samples_per_symbol=4)
            self.tx_filter = sn.phy.signal.RootRaisedCosineFilter(beta=0.3,
                                                             samples_per_symbol=4,
                                                             span_in_symbols=32)
            self.rx_filter = sn.phy.signal.RootRaisedCosineFilter(beta=0.3,
                                                             samples_per_symbol=4,
                                                             span_in_symbols=32)

            num_ptrs_blocks = 32
            symbols_per_ptrs_block = 4
            block_size_ptrs = self.n_out+ 4*32
            bandwidth = 960e3*4096
            # fc = 220e9
            samples = 4*(block_size_ptrs+32)

            self.downsampler = sn.phy.signal.Downsampling(samples_per_symbol=4,
                                                          offset = self.rx_filter.length - 1,
                                                     num_symbols=tf.cast(block_size_ptrs, tf.int32))

            path_to_mat = "../PreloadedMats/ptrs_CAZAC_4_32.mat"

            self.ptrs_adder = PTRS_Adder(path_to_mat, num_ptrs_blocks, symbols_per_ptrs_block, block_size_ptrs, batch_size)
            self.pn_compensate = PN_Compensation(self.ptrs_adder.ptrs_indices, self.ptrs_adder.data_indices, block_size_ptrs\
                                                 , self.ptrs_adder.pilots,batch_size, num_ptrs_blocks, symbols_per_ptrs_block)
            self.pn_tx = PhaseNoise(model_type='13', bandwidth=bandwidth, fc=fc, n_samples=samples)
            self.pn_rx = PhaseNoise(model_type='11', bandwidth=bandwidth, fc=fc, n_samples=samples)


    # No Graph execution possible due to sequential CCDM Encode and Decode
    # Due to this no attempt was made to use tensorflow implementation
    def __call__(self, batch_size, ebno_db):
        no = ebnodb2no(ebno_db,
                       num_bits_per_symbol=3.6,
                       coderate=1)

        b_stream = np.array(self.binary_source([self.k_input]),np.int64)
        code_symbols = np.array(ccdm.encode(b_stream, self.n_out, self.n_i))
        reverse_map = (((code_symbols[:, None]-1) & (1 << np.arange(self.num_bits_per_symbol-self.parity_k)[::-1])) > 0).astype(int)
        enc_input = np.reshape(reverse_map,((self.num_bits_per_symbol-self.parity_k)*self.n_out))
        parity_bits = self.H_sparse.dot(enc_input) % 2  # mod 2 because XOR over GF(2)
        parity_bits = np.mod(np.cumsum(parity_bits) , 2)
        parity_reshaped = np.reshape(parity_bits,(-1,self.parity_k))

        mapper_input_matrix = np.concatenate(
            (reverse_map[:, :2], parity_reshaped, reverse_map[:, -2:]),
            axis=1
        )
        mapper_input = np.reshape(mapper_input_matrix,(self.num_bits_per_symbol*self.n_out))
        x = self.mapper(mapper_input)

        #########################
        if self.use_upsampling_filtering_pn:
            ptrs_x_mod = self.ptrs_adder(tf.expand_dims(x,axis=0))
            x_up = self.upsampler(ptrs_x_mod)
            x_tx_filt = self.tx_filter(x_up)
            pn_out, _, _ = self.pn_tx(x_tx_filt)
            y_channel = self.channel(pn_out, no)
            y_channel_pn, _, _ = self.pn_rx(y_channel)
            y_rx_filt = self.rx_filter(y_channel_pn)
            y = self.downsampler(y_rx_filt)
            # print('y.shape: ',y.shape)
            _, y_data_vec = self.pn_compensate(y)
        else:
            y_data_vec = self.channel(x, no)
        #########################
        # # mean_squared = tf.reduce_mean(tf.abs(x) ** 2)
        # # print('meansquare',mean_squared.numpy())

        if self.use_neural_demapper:
            llr_ch = self.demapper(y_data_vec)

        else:
            # llr_ch = self.demapper(y_data_vec, no, self.prior)
            llr_ch = self.demapper(y_data_vec, no, self.normalized_points, self.prior)
            # print('llr_ch.shape 2:',llr_ch.shape)
        llr_reshaped = np.reshape(llr_ch,(-1,self.num_bits_per_symbol))

        # llr_parity_comp = np.reshape(llr_reshaped[:,:self.parity_k],(-1))
        # llr_amplitude_comp = np.reshape(llr_reshaped[:,self.parity_k:],(-1))
        llr_parity_comp    = np.reshape(llr_reshaped[:, ~self.mask], (-1,))
        llr_amplitude_comp = np.reshape(llr_reshaped[:, self.mask], (-1,))

        codeword_llr = np.hstack((llr_amplitude_comp, llr_parity_comp))
        decoded_bits = tf.cast(self.decoder(codeword_llr),tf.int64).numpy()
        info_bits_reshaped = np.reshape(decoded_bits[:self.k_ldpc],(-1,self.num_bits_per_symbol-self.parity_k))
        # Use bitwise left shift to compute powers of two
        shifted_bits = info_bits_reshaped << self.bit_positions
        # Compute the integer representation using bitwise operations
        code_symbols_retrieved = np.sum(shifted_bits, axis=-1)+1
        # tf.print('ccdm input: ', code_symbols_retrieved.shape)
        decoded_bits_ccdmdec = ccdm.decode(code_symbols_retrieved, self.n_i, self.k_input)
        missing = self.k_input - len(decoded_bits_ccdmdec)
        if missing > 0:
            decoded_bits_ccdmdec.extend([0] * missing)
        # tf.print('ccdm :',np.array(decoded_bits_ccdmdec).shape)
        # tf.print('b_stream :',b_stream.shape)

        return b_stream, decoded_bits_ccdmdec

if __name__ == "__main__":
    start_time = time.perf_counter()

    N = 64
    num_bits_per_symbol = int(np.log2(N))

    RESULTS_DIR = "results_pn_entropy_120GHz_64apsk1"
    snr_list = [13]
    papr_list = [6.4, 7.2, 8.0]
    pas_batch_size = 1
    num_bp_iter = 50
    max_mc_iter = 2

    use_dvb = True
    use_upsampling_filtering_pn = True
    use_neural_demapper = False
    pn_freq = '120'
    condition = 'lowpn'  # awgn highsnr
    # PAS_USE_AWGN = False

    p0 = 0.8544
    labels = np.arange(N)
    bin_labels = ((labels[:, None] & (1 << np.arange(num_bits_per_symbol - 1, -1, -1))) > 0).astype(int)
    first_bits = bin_labels[:, 0]  # First (MSB) bit
    probabilities = tf.where(first_bits == 0, p0 / (2 ** (num_bits_per_symbol - 1)),
                             (1 - p0) / (2 ** (num_bits_per_symbol - 1)))

    p_basic_mask = (bin_labels[:, 2] == 0) & (bin_labels[:, 3] == 0)
    prob_basic_set = probabilities[p_basic_mask] * 4

    ber_plot_pas = PlotBER("ber_plot_pas")
    # channel_tag = "awgn_tap" if PAS_USE_AWGN else "multipath"
    simulated_papr = []

    for snr_db in snr_list:
        for papr_db in papr_list:
            exp_name = f"snr_{snr_db}_papr_{papr_db}"
            npz_path = os.path.join(RESULTS_DIR, f"{exp_name}.npz")
            if not os.path.isfile(npz_path):
                print(f"Skipping missing file: {npz_path}")
                continue

            plotname = f"PAS_{condition}_papr_{papr_db}"
            print(f"Running simulation for {exp_name}...")

            data = np.load(npz_path)
            points = data["constellation_points"]
            # probabilities = data["constellation_probabilities"]
            # prob_reshape = np.reshape(probabilities, [4, 16])
            # prob_basic_set = prob_reshape[0, :] * 4

            pas_model = PAS_scheme(1, 50, points,
                                   prob_basic_set, probabilities, num_bits_per_symbol,
                                   use_upsampling_filtering_pn, use_neural_demapper, pn_freq, condition)

            # (self, batch_size, num_bp_iter, const_symbol_points, p_basic,
            # probabilities, num_bits_per_symbol, use_upsampling_filtering_pn,
            # use_neural_demapper, pn_freq, condition):

            ber_plot_pas.simulate(
                pas_model,
                ebno_dbs=np.arange(5.0, 13.0, 0.2),
                legend=plotname,
                add_bler=True,
                add_ber=True,
                max_mc_iter=max_mc_iter,
                num_target_block_errors=100,
                batch_size=pas_batch_size,
                soft_estimates=False,
                early_stop=True,
                show_fig=False,
                forward_keyboard_interrupt=False,
            )
            simulated_papr.append(papr_db)

    end_time = time.perf_counter()
    print(f"Execution time: {end_time - start_time:.6f} seconds")

    output_folder = "BLER_results_pcs"
    os.makedirs(output_folder, exist_ok=True)

    np.savez(
        os.path.join(output_folder, "pas.npz"),
        snr=ber_plot_pas.snr,
        ber=ber_plot_pas.ber,
        papr=simulated_papr,
        condition=condition,
        pn_freq=pn_freq
    )

    pas_snr = ber_plot_pas.snr
    pas_ber = ber_plot_pas.ber

    plt.figure()
    for i, papr_db in enumerate(simulated_papr):
        plt.semilogy(
            pas_snr[2 * i],
            pas_ber[2 * i].numpy(),
            "o-",
            label=f"PAS, PAPR={papr_db} dB",
        )
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("BER vs SNR")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "BER_plot.png"), dpi=300)
    plt.close()

    plt.figure()
    for i, papr_db in enumerate(simulated_papr):
        plt.semilogy(
            pas_snr[2 * i + 1],
            pas_ber[2 * i + 1].numpy(),
            "o-",
            label=f"PAS, PAPR={papr_db} dB",
        )
    plt.xlabel("SNR (dB)")
    plt.ylabel("BLER")
    plt.title("BLER vs SNR")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "BLER_plot.png"), dpi=300)
    plt.close()

    print(f"Saved plots and data to {output_folder}/")