# For the implementation of the neural receiver
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense, LayerNormalization
from tensorflow.nn import relu
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from sionna.phy.mapping import Demapper, Constellation, SymbolLogits2LLRs, BinarySource
from sionna.phy.utils import split_dim

import tensorflow as tf
import numpy as np
import sionna as sn
import matplotlib.pyplot as plt

class Sampling_sym_sub_constellation(Layer):
    def __init__(self, N, parity_k, tau, device, constellation):
        super().__init__()

        # Number of bits per symbol provided from uniform parity bits in the
        # PAS scheme
        self.parity_k = parity_k
        self.dense1 = Dense(64, activation='tanh')
        self.dense2 = Dense(64, activation='tanh')
        self.dense3 = Dense(int(N/(2**self.parity_k)), activation='softmax')
        self.dense4 = Dense(N, activation='linear')
        self.dense5 = Dense(N, activation='linear')
        self.tau = tau

        self.dense4.build(input_shape=(None, 64))
        self.dense5.build(input_shape=(None, 64))

        self.dense4.set_weights([
            np.zeros((64, N), dtype=np.float32),
            np.real(constellation)
        ])
        self.dense5.set_weights([
            np.zeros((64, N), dtype=np.float32),
            np.imag(constellation)
        ])

        self.device = device
        self.N = N

    def call(self, ebno_db):
        # Create a tensor with the SNR value
        # snr_tensor = tf.constant([[self.snr_db]], dtype=tf.float32)
        snr_tensor = tf.reshape(ebno_db, (-1, 1))

        self.batch_size = snr_tensor.shape[0]
        common_nn = self.dense2(self.dense1(snr_tensor))
        # symbol_prob_basic = self.dense3(common_nn)

        ######################### Trying out uniform

        symbol_prob_basic =  tf.constant(1/int(self.N/(2**self.parity_k)), shape=(self.batch_size, int(self.N/(2**self.parity_k))))
        #########################

        C_real = self.dense4(common_nn)
        C_imag = self.dense5(common_nn)

        C_points = tf.complex(C_real, C_imag)
        symbol_prob = tf.tile(symbol_prob_basic,(1,(2**self.parity_k)))

        # Normalize the constellation
        normalization_constants = tf.sqrt(
            tf.reduce_sum(tf.reshape(symbol_prob * tf.abs(C_points) ** 2,(-1,(2**self.parity_k),int(self.N/(2**self.parity_k)))), 2)
        )
        normalization_constants = tf.expand_dims(normalization_constants, axis=2)
        normalization_constants = tf.complex(tf.reshape(tf.tile(normalization_constants,(1,1,int(self.N/(2**self.parity_k)))),(-1,self.N)),0.0)
        norm_C_points = C_points / normalization_constants

        sampled_indices = np.arange(0, self.N)
        one_hot_samples = np.tile(np.eye(self.N)[sampled_indices], (self.batch_size, 1, 1))
        onehot_hard_sample = tf.convert_to_tensor(one_hot_samples, dtype=tf.float32)

        return onehot_hard_sample, symbol_prob/(2**self.parity_k), norm_C_points, symbol_prob_basic

class EndToEndSystem_bitwise_PAS(Model):
    def __init__(self, N, parity_k, constellation, device, constellation_orig, use_upsampling_filtering ,tau=1.0):
        super().__init__()
        print('device', device)

        # self.sampling = SamplingMechanism_P0_trainable_parameter(N, tau, device, constellation)
        self.sampling = Sampling_sym_sub_constellation(N, parity_k, tau, device, constellation)
        self.modulator = Modulator()
        self.channel = sn.phy.channel.AWGN()
        self.demodulator_test = Demapper_trainable("app", constellation=constellation_orig)
        self.loss_fn = BinaryCrossentropy(from_logits=True, reduction=None, axis=0)
        self.N = N
        self.bits_per_symbol = int(np.log2(self.N))
        self.bit_labels = tf.convert_to_tensor(np.array([[int(b) for b in format(i, f'0{self.bits_per_symbol}b')] for i in range(self.N)]),tf.float32)

        self.upsampler = sn.phy.signal.Upsampling(samples_per_symbol=4)
        self.tx_filter = sn.phy.signal.RootRaisedCosineFilter(beta=0.3,
                                                         samples_per_symbol=4,
                                                         span_in_symbols=32)
        self.rx_filter = sn.phy.signal.RootRaisedCosineFilter(beta=0.3,
                                                         samples_per_symbol=4,
                                                         span_in_symbols=32)
        self.downsampler = sn.phy.signal.Downsampling(samples_per_symbol=4,
                                                      offset = self.rx_filter.length - 1,
                                                 num_symbols=tf.cast(self.N, tf.int32))
        self.use_upsampling_filtering = use_upsampling_filtering

        ###########################
        self.mapper = Mapper(self.bits_per_symbol)
        self.binary_source = BinarySource()

        ###########################

    def call(self, ebno_db, No):
        # Sampling mechanism generates symbol indices and shaping probabilities
        s, symbol_prob, norm_C_points, symbol_prob_basic = self.sampling(ebno_db)
        self.batch_size = norm_C_points.shape[0]

        # log_odds = tf.math.log((1.0 - 0.8125) / 0.8125)  # shape: scalar or tensor
        # log_odds = tf.reshape(log_odds, (-1, 1, 1))
        # log_odds_tiled = tf.tile(log_odds,[1, self.N, 1])

        # Use tf.concat to place log-odds in the correct position
        # self.prior_values = tf.concat([
        #     log_odds_tiled,
        #     tf.zeros((self.batch_size, self.N, self.bits_per_symbol-1), dtype=tf.float32)
        # ], axis=-1)


        # # Marginalize: P(b_j=1) = sum_{s} P(s) * I[b_j(s)=1]
        tiled_bit_labels = tf.tile(tf.expand_dims(self.bit_labels, 0),[self.batch_size, 1, 1])
        p_b1 = tf.matmul(symbol_prob[:, None],tiled_bit_labels)
        # # Compute LLRs
        self.prior_values = tf.tile(tf.math.log(p_b1 / (1.0 - p_b1)),[1, self.N, 1])
        #
        # self.prior_values =

        indices = tf.argmax(s, axis=2, output_type=tf.int64)  # use int64 here!

        # Step 2: Convert to binary bits
        shifts = tf.range(self.bits_per_symbol - 1, -1, -1, dtype=tf.int64)  # also int64

        # Right shift and mask
        bit_repr = tf.bitwise.right_shift(
            tf.expand_dims(indices, -1), shifts
        ) & 1
        # Modulate the symbols
        x_mod = self.modulator(s, norm_C_points)

        if self.use_upsampling_filtering:
            x_up = self.upsampler(x_mod)
            x_tx_filt = self.tx_filter(x_up)
            y_channel = self.channel(x_tx_filt, No)
            y_rx_filt = self.rx_filter(y_channel)
            y = self.downsampler(y_rx_filt)

        else:
            # Transmit through the channel
            y = self.channel(x_mod, No)

        # Demodulate the received signal
        tf_points = tf.reshape(norm_C_points, (self.batch_size, 1, self.N))
        logits = self.demodulator_test(y, No, tf_points, self.prior_values)
        # s_detached = tf.stop_gradient(s)

        logits_reshaped = tf.reshape(logits, (self.batch_size, self.N, self.bits_per_symbol))
        bit_repr_reshaped = tf.cast(tf.reshape(bit_repr, (self.batch_size, self.N, self.bits_per_symbol)), tf.float32)

        bit_repr_reshaped = tf.cast(bit_repr_reshaped, tf.float32)

        entropy_value = tf.math.reduce_mean(-tf.reduce_sum(symbol_prob * tf.math.log(symbol_prob), axis=1))
        symbol_prob = tf.expand_dims(symbol_prob, axis=-1)
        bce_matrix = tf.nn.sigmoid_cross_entropy_with_logits(labels=bit_repr_reshaped, logits=logits_reshaped)
        loss_ordinary = tf.reduce_sum(tf.math.reduce_mean(tf.reduce_sum(bce_matrix * symbol_prob, axis=1), axis=0))

        loss = loss_ordinary - entropy_value

        return loss, loss_ordinary, entropy_value, symbol_prob, norm_C_points, symbol_prob_basic

    @tf.function
    def compute_power_to_average_power_samples(self, papr_batch_size, papr_block_size, norm_C_points):
        bits_power = self.binary_source([papr_batch_size, papr_block_size * self.bits_per_symbol])
        x_power = self.mapper(bits_power, norm_C_points)
        x_up_power = self.upsampler(x_power)
        x_tx_filt_power = self.tx_filter(x_up_power)  # 'same' for discarding the edges
        p_x = tf.square(tf.abs(x_tx_filt_power))
        power_samples = p_x / tf.reduce_mean(p_x)
        return power_samples

    def test_sample_values(self, papr_batch_size, papr_block_size, norm_C_points):
        bits_power = self.binary_source([papr_batch_size, papr_block_size * self.bits_per_symbol])
        x_power = self.mapper(bits_power, norm_C_points)
        x_up_power = self.upsampler(x_power)
        x_tx_filt_power = self.tx_filter(x_up_power)  # 'same' for discarding the edges
        p_x = tf.square(tf.abs(x_tx_filt_power))
        power_samples = p_x / tf.reduce_mean(p_x)
        return power_samples, x_power, x_up_power, x_tx_filt_power, p_x

class Modulator(Layer):

    def __init__(self, **kwargs):
        super(Modulator, self).__init__(**kwargs)

    def call(self, s, norm_C_points):
        norm_C_points = tf.expand_dims(norm_C_points, axis=-1)  # Expand further for batch processing
        x_real = tf.matmul(s, tf.math.real(norm_C_points))
        x_imag = tf.matmul(s, tf.math.imag(norm_C_points))
        x = tf.complex(x_real, x_imag)

        return tf.squeeze(x, axis=-1)


class Demapper_trainable(Demapper):
    def __init__(self,
                 demapping_method,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 precision=None,
                 **kwargs):
        super().__init__(demapping_method=demapping_method, constellation=constellation, precision=precision, **kwargs)

        # Create constellation object
        self._constellation = Constellation.check_or_create(
            constellation_type=constellation_type,
            num_bits_per_symbol=num_bits_per_symbol,
            constellation=constellation,
            precision=precision)

        num_bits_per_symbol = self._constellation.num_bits_per_symbol

        self._logits2llrs = SymbolLogits2LLRs(demapping_method,
                                              num_bits_per_symbol,
                                              hard_out=hard_out,
                                              precision=precision,
                                              **kwargs)

        self._no_threshold = tf.cast(np.finfo(self.rdtype.as_numpy_dtype).tiny,
                                     self.rdtype)

    def call(self, y, no, points, prior=None):
        # Compute squared distances from y to all points
        # shape [...,n,num_points]
        squared_dist = tf.pow(tf.abs(tf.expand_dims(y, axis=-1) - points), 2)

        # Add a dummy dimension for broadcasting. This is not needed when no
        # is a scalar, but also does not do any harm.
        no = tf.expand_dims(no, axis=-1)
        # Deal with zero or very small values.
        no = tf.math.maximum(no, self._no_threshold)

        # Compute exponents
        exponents = -squared_dist / no
        llr = self._logits2llrs(exponents, prior)

        # Reshape LLRs to [...,n*num_bits_per_symbol]
        out_shape = tf.concat([tf.shape(y)[:-1],
                               [y.shape[-1] * \
                                self.constellation.num_bits_per_symbol]], 0)

        llr_reshaped = tf.reshape(llr, out_shape)

        return llr_reshaped


def normalize_constellation(points, probabilities):
    # Compute the power of the constellation
    power = np.sum(probabilities * np.abs(points) ** 2)
    # Normalize the constellation points
    normalized_points = points / np.sqrt(power)
    return normalized_points


def plot_constellation_wlabels(points, probabilities, title):
    points = np.asarray(points)
    probabilities = np.asarray(probabilities).flatten()

    # Scale marker sizes for visibility
    marker_sizes = 300 * probabilities / max(probabilities)

    plt.figure(figsize=(8, 8))
    plt.scatter(points.real, points.imag, s=marker_sizes, alpha=0.7, c='b', edgecolors='k')

    # Annotate with 6-bit binary labels
    for i, pt in enumerate(points):
        label = format(i, '06b')  # 6-bit binary string
        plt.text(pt.real + 0.05, pt.imag + 0.05, label, fontsize=9, ha='left', va='bottom', color='darkred')

    # Add grid and formatting
    plt.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    plt.grid(True, linestyle=':')
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()
    # plt.savefig('const.png')

def generate_ldpc_H_matrix(n_ldpc: int, rate: float, q: int, file_path: str) -> np.ndarray:
    # Compute derived parameters
    k_ldpc = int(n_ldpc * rate)
    N_p = n_ldpc - k_ldpc  # number of parity bits

    data = []
    with open(file_path, "r") as file:
        for line in file:
            values = [int(v) for v in line.strip().split()]
            data.append(values)

    H_matrix = np.zeros((N_p, n_ldpc), dtype=np.uint8)

    for index, base_addresses in enumerate(data):
        base_addresses = np.array(base_addresses)

        for addr in base_addresses:
            H_matrix[addr, 360 * index + 0] = 1

        for m in range(1, 360):
            for addr in base_addresses:
                new_addr = (addr + (m % 360) * q) % N_p
                H_matrix[new_addr, 360 * index + m] = 1

    P_matrix = np.zeros((N_p, N_p), dtype=int)
    np.fill_diagonal(P_matrix, 1)
    for i in range(1, N_p):
        P_matrix[i, i - 1] = 1

    H_matrix[:, k_ldpc:] = P_matrix

    return H_matrix

symbol_points_qam64 = np.array([-1.08012344973464 + 1.08012344973464j
,-1.08012344973464 + 0.771516749810460j
,-1.08012344973464 + 0.154303349962092j
,-1.08012344973464 + 0.462910049886276j
,-0.771516749810460 + 1.08012344973464j
,-0.771516749810460 + 0.771516749810460j
,-0.771516749810460 + 0.154303349962092j
,-0.771516749810460 + 0.462910049886276j
,-0.154303349962092 + 1.08012344973464j
,-0.154303349962092 + 0.771516749810460j
,-0.154303349962092 + 0.154303349962092j
,-0.154303349962092 + 0.462910049886276j
,-0.462910049886276 + 1.08012344973464j
,-0.462910049886276 + 0.771516749810460j
,-0.462910049886276 + 0.154303349962092j
,-0.462910049886276 + 0.462910049886276j
,-1.08012344973464 - 1.08012344973464j
,-1.08012344973464 - 0.771516749810460j
,-1.08012344973464 - 0.154303349962092j
,-1.08012344973464 - 0.462910049886276j
,-0.771516749810460 - 1.08012344973464j
,-0.771516749810460 - 0.771516749810460j
,-0.771516749810460 - 0.154303349962092j
,-0.771516749810460 - 0.462910049886276j
,-0.154303349962092 - 1.08012344973464j
,-0.154303349962092 - 0.771516749810460j
,-0.154303349962092 - 0.154303349962092j
,-0.154303349962092 - 0.462910049886276j
,-0.462910049886276 - 1.08012344973464j
,-0.462910049886276 - 0.771516749810460j
,-0.462910049886276 - 0.154303349962092j
,-0.462910049886276 - 0.462910049886276j
,1.08012344973464 + 1.08012344973464j
,1.08012344973464 + 0.771516749810460j
,1.08012344973464 + 0.154303349962092j
,1.08012344973464 + 0.462910049886276j
,0.771516749810460 + 1.08012344973464j
,0.771516749810460 + 0.771516749810460j
,0.771516749810460 + 0.154303349962092j
,0.771516749810460 + 0.462910049886276j
,0.154303349962092 + 1.08012344973464j
,0.154303349962092 + 0.771516749810460j
,0.154303349962092 + 0.154303349962092j
,0.154303349962092 + 0.462910049886276j
,0.462910049886276 + 1.08012344973464j
,0.462910049886276 + 0.771516749810460j
,0.462910049886276 + 0.154303349962092j
,0.462910049886276 + 0.462910049886276j
,1.08012344973464 - 1.08012344973464j
,1.08012344973464 - 0.771516749810460j
,1.08012344973464 - 0.154303349962092j
,1.08012344973464 - 0.462910049886276j
,0.771516749810460 - 1.08012344973464j
,0.771516749810460 - 0.771516749810460j
,0.771516749810460 - 0.154303349962092j
,0.771516749810460 - 0.462910049886276j
,0.154303349962092 - 1.08012344973464j
,0.154303349962092 - 0.771516749810460j
,0.154303349962092 - 0.154303349962092j
,0.154303349962092 - 0.462910049886276j
,0.462910049886276 - 1.08012344973464j
,0.462910049886276 - 0.771516749810460j
,0.462910049886276 - 0.154303349962092j
,0.462910049886276 - 0.462910049886276j])

class Mapper(Layer):
    def __init__(self, num_bits_per_symbol,**kwargs):
        super(Mapper, self).__init__(**kwargs)
        self.num_bits_per_symbol = num_bits_per_symbol
        self._bit_positions = tf.cast(tf.range(num_bits_per_symbol-1, -1, -1), dtype=tf.int32)

    def call(self, bits, norm_C_points):

         # Convert to int32
        bits = tf.cast(bits, dtype=tf.int32)

        # Reshape last dimensions to the desired format
        n1 = int(bits.shape[-1]/self.num_bits_per_symbol)
        new_shape = [n1 , self.num_bits_per_symbol]
        bits = split_dim(bits, new_shape, axis=tf.rank(bits)-1)

        # Use bitwise left shift to compute powers of two
        shifted_bits = tf.bitwise.left_shift(bits, self._bit_positions)

        # Compute the integer representation using bitwise operations
        int_rep = tf.reduce_sum(shifted_bits, axis=-1)

        # Map integers to constellation symbols
        x = tf.gather(norm_C_points, int_rep, axis=1, batch_dims=1)
        return x