import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Dense
from tensorflow.nn import relu
from tensorflow.keras.losses import BinaryCrossentropy
from sionna.phy.block import Block
from sionna.phy.utils import expand_to_rank, hard_decisions
import importlib
import pn_classes

importlib.reload(pn_classes)
from pn_classes import PhaseNoise, PN_Compensation, PTRS_Adder
import sionna as sn
import matplotlib.pyplot as plt
import os
import gc

symbol_points_qam32 = np.array([-0.6708 + 1.1180j,
                                -0.2236 + 1.1180j,
                                -0.2236 + 0.6708j,
                                -0.2236 + 0.2236j,
                                -1.1180 + 0.6708j,
                                -1.1180 + 0.2236j,
                                -0.6708 + 0.6708j,
                                -0.6708 + 0.2236j,
                                -0.6708 - 1.1180j,
                                -0.2236 - 1.1180j,
                                -0.2236 - 0.6708j,
                                -0.2236 - 0.2236j,
                                -1.1180 - 0.6708j,
                                -1.1180 - 0.2236j,
                                -0.6708 - 0.6708j,
                                -0.6708 - 0.2236j,
                                0.6708 + 1.1180j,
                                0.2236 + 1.1180j,
                                0.2236 + 0.6708j,
                                0.2236 + 0.2236j,
                                1.1180 + 0.6708j,
                                1.1180 + 0.2236j,
                                0.6708 + 0.6708j,
                                0.6708 + 0.2236j,
                                0.6708 - 1.1180j,
                                0.2236 - 1.1180j,
                                0.2236 - 0.6708j,
                                0.2236 - 0.2236j,
                                1.1180 - 0.6708j,
                                1.1180 - 0.2236j,
                                0.6708 - 0.6708j,
                                0.6708 - 0.2236j])

# symbol_points_qam32 = np.array([-1.0 + 1.0j,
#                                 -1.0 + 1.0j,
#                                 -1.0 + 1.0j,
#                                 -1.0 + 1.0j,
#                                 -1.0 + 1.0j,
#                                 -1.0 + 1.0j,
#                                 -1.0 + 1.0j,
#                                 -1.0 + 1.0j,
#                                 -1.0 - 1.0j,
#                                 -1.0 - 1.0j,
#                                 -1.0 - 1.0j,
#                                 -1.0 - 1.0j,
#                                 -1.0 - 1.0j,
#                                 -1.0 - 1.0j,
#                                 -1.0 - 1.0j,
#                                 -1.0 - 1.0j,
#                                 1.0 + 1.0j,
#                                 1.0 + 1.0j,
#                                 1.0 + 1.0j,
#                                 1.0 + 1.0j,
#                                 1.0 + 1.0j,
#                                 1.0 + 1.0j,
#                                 1.0 + 1.0j,
#                                 1.0 + 1.0j,
#                                 1.0 - 1.0j,
#                                 1.0 - 1.0j,
#                                 1.0 - 1.0j,
#                                 1.0 - 1.0j,
#                                 1.0 - 1.0j,
#                                 1.0 - 1.0j,
#                                 1.0 - 1.0j,
#                                 1.0 - 1.0j])

def entropy(pdf, base=2):
    pdf = np.asarray(pdf, dtype=np.float64)
    pdf = pdf[pdf > 0]  # remove zero entries to avoid log(0)
    pdf /= np.sum(pdf)  # normalize to ensure valid probability distribution

    h = -np.sum(pdf * np.log(pdf) / np.log(base))
    return h


def normalize_constellation(points, probabilities):
    # Compute the power of the constellation
    power = np.sum(probabilities * np.abs(points) ** 2)
    # Normalize the constellation points
    normalized_points = points / np.sqrt(power)
    return normalized_points


def plot_constellation_wlabels(points, probabilities, plot_title):
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
    plt.title(plot_title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    # plt.show()


class Modulator(Layer):

    def __init__(self, constellation, N, parity_k, **kwargs):
        super(Modulator, self).__init__(**kwargs)

        # Define the real and imaginary parts of the constellation as trainable variables
        # self.constellation_real = tf.Variable(constellation.real, dtype=tf.float32, trainable=True)
        # self.constellation_imag = tf.Variable(constellation.imag, dtype=tf.float32, trainable=True)
        self.constellation_real_init = constellation.real
        self.constellation_imag_init = constellation.imag
        self.parity_k = parity_k
        self.N = N

    def build(self, input_shape):
        self.constellation_real = self.add_weight(
            name="constellation_real",
            shape=self.constellation_real_init.shape,
            initializer=tf.constant_initializer(self.constellation_real_init),
            trainable=True
        )
        self.constellation_imag = self.add_weight(
            name="constellation_imag",
            shape=self.constellation_imag_init.shape,
            initializer=tf.constant_initializer(self.constellation_imag_init),
            trainable=True
        )

    def call(self, s, probabilities):
        # Combine the real and imaginary parts into a complex-valued constellation
        constellation_combined = tf.complex(self.constellation_real, self.constellation_imag)

        # Expand the signal dimension to align for matrix multiplication
        s = tf.expand_dims(s, axis=-1)

        # Normalize the constellation
        normalization_constants = tf.sqrt(
            tf.reduce_sum(tf.reshape((2 ** parity_k) * probabilities * tf.abs(constellation_combined) ** 2,
                                     (1, (2 ** self.parity_k), int(self.N / (2 ** self.parity_k)))), 2)
        )
        normalization_constants = tf.expand_dims(normalization_constants, axis=2)
        normalization_constants = tf.complex(
            tf.reshape(tf.tile(normalization_constants, (1, 1, int(self.N / (2 ** self.parity_k)))), (1, self.N)), 0.0)
        norm_constellation = constellation_combined / normalization_constants

        # print('norm_constellation',norm_constellation.shape)
        # print('s',s.shape)
        # Compute the modulated signal
        x_real = tf.matmul(tf.math.real(norm_constellation), s)
        x_imag = tf.matmul(tf.math.imag(norm_constellation), s)
        x = tf.complex(x_real, x_imag)

        return tf.squeeze(x, axis=[-1, -2]), norm_constellation


class SymbolLogits2LLRs_Prob_Aware(Block):

    def __init__(self,
                 method,
                 num_bits_per_symbol,
                 *,
                 hard_out=False,
                 precision=None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)
        assert method in ("app", "maxlog"), "Unknown demapping method"
        self._method = method
        self._hard_out = hard_out
        self._num_bits_per_symbol = num_bits_per_symbol
        num_points = int(2 ** num_bits_per_symbol)
        a = np.zeros([num_points, num_bits_per_symbol])
        for i in range(0, num_points):
            a[i, :] = np.array(list(np.binary_repr(i, num_bits_per_symbol)),
                               dtype=np.int32)
        c0 = np.zeros([int(num_points / 2), num_bits_per_symbol])
        c1 = np.zeros([int(num_points / 2), num_bits_per_symbol])
        for i in range(num_bits_per_symbol - 1, -1, -1):
            c0[:, i] = np.where(a[:, i] == 0)[0]
            c1[:, i] = np.where(a[:, i] == 1)[0]
        self._c0 = tf.constant(c0, dtype=tf.int32)  # Symbols with ith bit=0
        self._c1 = tf.constant(c1, dtype=tf.int32)  # Symbols with ith bit=1
        a = 2 * a - 1
        self._a = tf.constant(a, dtype=self.rdtype)
        if self._method == "app":
            self._reduce = tf.reduce_logsumexp
        else:
            self._reduce = tf.reduce_max

    @property
    def num_bits_per_symbol(self):
        """
        `int` : Number of bits per symbol
        """
        return self._num_bits_per_symbol

    def call(self, logits, probabilities):
        # Compute exponents
        exponents = logits
        exp0 = tf.gather(exponents, self._c0, axis=-1, batch_dims=0)
        exp1 = tf.gather(exponents, self._c1, axis=-1, batch_dims=0)
        exp_ps = expand_to_rank(tf.math.log(probabilities), tf.rank(logits), axis=0)
        exp_ps0 = tf.gather(exp_ps, self._c0, axis=-1)
        exp_ps1 = tf.gather(exp_ps, self._c1, axis=-1)
        llr = self._reduce(exp_ps1 + exp1, axis=-2) \
              - self._reduce(exp_ps0 + exp0, axis=-2)
        if self._hard_out:
            return hard_decisions(llr)
        else:
            return llr


class Demapper_Prob_Aware(Block):
    # pylint: disable=line-too-long

    def __init__(self,
                 demapping_method,
                 num_bits_per_symbol=None,
                 hard_out=False,
                 precision=None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)

        self._logits2llrs = SymbolLogits2LLRs_Prob_Aware(demapping_method,
                                                         num_bits_per_symbol,
                                                         hard_out=hard_out,
                                                         precision=precision,
                                                         **kwargs)

        self._no_threshold = tf.cast(np.finfo(self.rdtype.as_numpy_dtype).tiny,
                                     self.rdtype)

    def call(self, y, no, points, probabilities):
        squared_dist = tf.pow(tf.abs(tf.expand_dims(y, axis=-1) - points), 2)
        no = tf.expand_dims(no, axis=-1)
        no = tf.math.maximum(no, self._no_threshold)
        exponents = -squared_dist / no
        llr = self._logits2llrs(exponents, probabilities)

        return llr


class SamplingMechanism(Layer):
    def __init__(self, N, tau, parity_k):
        super(SamplingMechanism, self).__init__()
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(int(N / (2 ** parity_k)), activation=None)
        self.tau = tau
        # self.snr_db = snr_db
        self.parity_k = parity_k
        self.N = N

    def call(self, batch_size, block_size):
        # Create a tensor with the SNR value
        snr_tensor = tf.constant([[1]], dtype=tf.float32)
        # Pass the SNR value through the first dense layer
        a = self.dense1(snr_tensor)

        # Pass through the second dense layer
        logits = self.dense2(a)
        # print('1',logits.shape)
        #
        # Copying the logits 2**parity_k times to constrain according to PAS
        logits = tf.tile(logits, [1, 2 ** self.parity_k])
        # print('2',logits.shape)

        # Compute softmax output
        softmax_out = tf.nn.softmax(logits, axis=-1)

        logits = tf.expand_dims(logits, axis=1)
        # print('3',logits.shape)
        logits = tf.tile(logits, [batch_size, block_size, 1])

        # Compute Gumbel-Softmax with straight-through sampling
        gumbel_dist = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits), 0, 1) + 1e-20) + 1e-20)
        gumbel_logits = logits + gumbel_dist
        gumbel_softmax = tf.nn.softmax(gumbel_logits / self.tau, axis=-1)

        max_value = tf.argmax(gumbel_softmax, axis=-1)
        # print(gumbel_softmax.shape)

        # Step 2: Compare each value in gumbel_softmax with the max_value
        hard_sample = tf.one_hot(max_value, depth=self.N, dtype=tf.float32)
        # print("hard_sample:", hard_sample)

        straight_through = gumbel_softmax + tf.stop_gradient(hard_sample - gumbel_softmax)

        return straight_through, softmax_out


class EndToEndSystem_bitwise(Model):
    def __init__(self, N, constellation, papr_batch_size, batch_size, block_size, bits_per_symbol, parity_k,
                 use_upsampling_filtering, tau=1.0):
        super(EndToEndSystem_bitwise, self).__init__()
        # self.No = 1/(10**(snr_db/10))
        # constellation_orig = Constellation("pam", num_bits_per_symbol=6)
        # self.demodulator_map = Demapper_map("app", constellation=constellation_orig)
        self.demodulator = Demapper_Prob_Aware("app", bits_per_symbol)
        self.sampling = SamplingMechanism(N, tau, parity_k)
        self.batch_size = batch_size
        self.block_size = block_size
        self.papr_batch_size = papr_batch_size
        self.modulator = Modulator(constellation, N, parity_k)
        self.channel = sn.phy.channel.AWGN()
        # self.demodulator = Demodulator_bitwise(bits_per_symbol)
        self.loss_fn = BinaryCrossentropy(from_logits=False, reduction='none', axis=[0, 1])
        self.bits_per_symbol = bits_per_symbol
        self.parity_k = parity_k
        self.use_upsampling_filtering = use_upsampling_filtering
        self.upsampler = sn.phy.signal.Upsampling(samples_per_symbol=4)
        self.tx_filter = sn.phy.signal.RootRaisedCosineFilter(beta=0.3,
                                                              samples_per_symbol=4,
                                                              span_in_symbols=32)
        self.rx_filter = sn.phy.signal.RootRaisedCosineFilter(beta=0.3,
                                                              samples_per_symbol=4,
                                                              span_in_symbols=32)

        num_ptrs_blocks = 32
        symbols_per_ptrs_block = 4
        block_size_ptrs = 4096
        bandwidth = 960e3 * 4096
        fc = 220e9
        samples = 4 * (block_size_ptrs + 32)

        self.downsampler = sn.phy.signal.Downsampling(samples_per_symbol=4,
                                                      offset=self.rx_filter.length - 1,
                                                      num_symbols=tf.cast(block_size_ptrs, tf.int32))

        path_to_mat = "../PreloadedMats/ptrs_CAZAC_4_32.mat"

        self.ptrs_adder = PTRS_Adder(path_to_mat, num_ptrs_blocks, symbols_per_ptrs_block, block_size_ptrs, batch_size)
        self.pn_compensate = PN_Compensation(self.ptrs_adder.ptrs_indices, self.ptrs_adder.data_indices,
                                             block_size_ptrs, self.ptrs_adder.pilots, batch_size)
        self.pn_tx = PhaseNoise(model_type='13', bandwidth=bandwidth, fc=fc, n_samples=samples)
        self.pn_rx = PhaseNoise(model_type='11', bandwidth=bandwidth, fc=fc, n_samples=samples)

        self.prior_values = tf.convert_to_tensor(np.zeros(self.bits_per_symbol), tf.float32)

    def call(self, No):
        # Sampling mechanism generates symbol indices and shaping probabilities

        s, shaping_probs = self.sampling(batch_size=self.batch_size, block_size=self.block_size)
        indices = tf.argmax(s, axis=2, output_type=tf.int64)  # use int64 here!
        # Step 2: Convert to binary bits
        shifts = tf.range(self.bits_per_symbol - 1, -1, -1, dtype=tf.int64)  # also int64
        # Right shift and mask
        bit_repr = tf.bitwise.right_shift(
            tf.expand_dims(indices, -1), shifts
        ) & 1

        # Modulate the symbols
        x_mod, norm_const = self.modulator(s, shaping_probs)
        ptrs_x_mod = self.ptrs_adder(x_mod)
        # print(ptrs_x_mod.shape)

        if self.use_upsampling_filtering:
            x_up = self.upsampler(ptrs_x_mod)
            x_tx_filt = self.tx_filter(x_up)
            pn_out, _, _ = self.pn_tx(x_tx_filt)
            y_channel = self.channel(pn_out, No)
            y_channel_pn, _, _ = self.pn_rx(y_channel)
            y_rx_filt = self.rx_filter(y_channel_pn)
            y = self.downsampler(y_rx_filt)
            # print('y.shape: ',y.shape)
            _, y_data_vec = self.pn_compensate(y)
        else:
            # Transmit through the channel
            y = self.channel(x_mod, No)

        # Demodulate the received signal
        # logits = self.demodulator(y_data_vec, batch_size = self.batch_size, block_size = self.block_size)
        logits = self.demodulator(y_data_vec, No, norm_const, shaping_probs)
        # logits = self.demodulator_map(y_data_vec, No, norm_const, self.prior_values)

        loss_ordinary = tf.reduce_sum(self.loss_fn(bit_repr, logits))
        entropy_value = -tf.reduce_sum(shaping_probs * tf.math.log(shaping_probs))
        loss = loss_ordinary - entropy_value

        return loss, loss_ordinary, entropy_value, shaping_probs, norm_const

    @tf.function
    def compute_power_to_average_power_samples(self):
        s_power, shaping_probs_power = self.sampling(batch_size=self.papr_batch_size, block_size=self.block_size)
        x_power, _ = self.modulator(s_power, shaping_probs_power)
        x_up_power = self.upsampler(x_power)
        x_tx_filt_power = self.tx_filter(x_up_power)  # 'same' for discarding the edges
        p_x = tf.square(tf.abs(x_tx_filt_power))
        power_samples = p_x / tf.reduce_mean(p_x)
        entropy_value_power = -tf.reduce_sum(shaping_probs_power * tf.math.log(shaping_probs_power))
        return power_samples, entropy_value_power

    @tf.function
    def minimum_distance_penalty_complex(self, constellation, d_min=0.05):

        # tf.print('constellation :', constellation.shape)

        constellation = tf.squeeze(constellation)

        x_i = tf.expand_dims(constellation, 1)
        x_j = tf.expand_dims(constellation, 0)

        diff = x_i - x_j
        dist = tf.abs(diff)

        # tf.print('dist :', dist.shape)

        mask = 1 - tf.eye(tf.shape(constellation)[0])
        dist = dist * tf.cast(mask, dist.dtype)

        penalty = tf.square(tf.nn.relu(d_min - dist))
        penalty = penalty * tf.cast(mask, penalty.dtype)

        penalty = tf.reduce_sum(penalty) / 2.0

        return penalty


## Inner loop
@tf.function
def _inner_loop(lambd_papr, lambd_H, mu, nu, step, num_it, excess_power_target, entropy_target, papr_threshold, No):
    # We need to initialize c...
    excess_power = 0.0
    c_excess_power = excess_power
    L = 0.0
    lambda_dist = 10

    k = tf.constant(0, tf.int32)
    while tf.less(k, num_it):
        with tf.GradientTape() as tape:
            loss, loss_ordinary, entropy_value, _, norm_constellation = model(No)

            if use_upsampling_filtering:
                powers, entropy_value = model.compute_power_to_average_power_samples()
                excess_power = tf.reduce_mean(tf.nn.relu(powers[:, 64:-64] - papr_threshold))
                excess_power = tf.math.log(excess_power + 1e-12) / tf.math.log(10.)
                c_excess_power = excess_power - excess_power_target
                c_excess_power = tf.maximum(c_excess_power, lambd_papr / mu)
                dist_penalty = model.minimum_distance_penalty_complex(norm_constellation, d_min=0.1)
                excess_H = tf.maximum(entropy_value - entropy_target, lambd_H / nu) * 10

            # Augmented Lagrangian
            L = loss - lambd_papr * c_excess_power + 0.5 * mu * (
                tf.square(c_excess_power)) + lambda_dist * dist_penalty - lambd_H * excess_H + 0.5 * nu * (
                    tf.square(excess_H))

            # tf.print(dist_penalty)

        # Computing and applying gradients
        weights = model.trainable_weights
        grads = tape.gradient(L, weights)
        grads, glob_norm = tf.clip_by_global_norm(grads, 0.1)
        optimizer.apply_gradients(zip(grads, weights))
        # Print progress
        if k % 50 == 0:
            tf.print('Iteration', k, 'L', L / np.log(2), 'loss', loss / np.log(2), 'loss_ordinary',
                     loss_ordinary / np.log(2), 'entropy_value', entropy_value / np.log(2), 'glob_norm', glob_norm)

        k = k + 1
        step = step + 1

    if use_upsampling_filtering:
        powers, entropy_value = model.compute_power_to_average_power_samples()
        excess_power = tf.reduce_mean(tf.nn.relu(powers[:, 64:-64] - papr_threshold))
        excess_power = tf.math.log(excess_power + 1e-12) / tf.math.log(10.)
        c_excess_power = excess_power - excess_power_target
        c_excess_power = tf.maximum(c_excess_power, lambd_papr / mu)

    return c_excess_power, entropy_value, step


def run_experiment(snr_db, papr_threshold_db):
    print(f"\n===== Running SNR={snr_db} dB | PAPR={papr_threshold_db} dB =====\n")

    ebno_db = tf.random.uniform(shape=[batch_size], minval=8.0, maxval=16.0)
    No_ = sn.phy.utils.ebnodb2no(ebno_db, num_bits_per_symbol=bits_per_symbol, coderate=1)
    No_ = tf.expand_dims(No_, axis=-1)

    # No_ = tf.convert_to_tensor(1/(10**(snr_db/10)),tf.float32)
    papr_threshold_ = tf.math.pow(10.0, papr_threshold_db / 10.0)
    excess_power_target_ = -6.0  # B
    entropy_target_ = 4.66 * 0.693

    lambd_papr = tf.Variable(tf.constant(0.0, tf.float32), trainable=False, dtype=tf.float32)
    lambd_H = tf.Variable(tf.constant(0.0, tf.float32), trainable=False, dtype=tf.float32)
    mu = tf.Variable(0.01, trainable=False, dtype=tf.float32)
    nu = tf.Variable(0.01, trainable=False, dtype=tf.float32)
    step = tf.constant(0, tf.int32)

    for i in range(150):
        if i == 0:
            num_it = 500
        else:
            num_it = 35
        c_excess_power, entropy_value, step = _inner_loop(lambd_papr, lambd_H, mu, nu, step,
                                                          num_it, excess_power_target_,
                                                          entropy_target_, papr_threshold_, No_)

        excess_H = tf.maximum(entropy_value - entropy_target_, lambd_H / nu) * 10

        # Update Lagrange multipliers
        tf.print('-------------------------------------------------------')
        tf.print('-------------------------------------------------------')
        tf.print('Progress ', i, ' % ', 'c_excess_power', c_excess_power, 'c_excess_H', excess_H / 0.693, 'lambd_papr',
                 lambd_papr, 'mu', mu, 'nu', nu)
        tf.print('-------------------------------------------------------')
        tf.print('-------------------------------------------------------')
        lambd_papr = lambd_papr - mu * c_excess_power
        lambd_H = lambd_H - nu * excess_H

        # Update penalty parameter
        mu = mu * 1.003
        nu = nu * 1.003

    # ------------------
    # Save constellation
    # ------------------
    with tf.GradientTape() as tape:
        _, _, _, shaping_probs, _ = model(No_)

    points = (
            model.modulator.constellation_real.numpy()
            + 1j * model.modulator.constellation_imag.numpy()
    )
    probs = shaping_probs.numpy()

    entropy_finalized = entropy(probs)

    normalized_points = normalize_constellation(points, probs)

    exp_name = f"snr_{snr_db}_papr_{papr_threshold_db}"
    # np.save(f"{RESULTS_DIR}/{exp_name}_points.npy", normalized_points)
    # np.save(f"{RESULTS_DIR}/{exp_name}_probs.npy", probs)

    plt.figure()
    plot_constellation_wlabels(
        normalized_points,
        probs,
        f"PAPR={papr_threshold_db} dB"
    )
    plt.savefig(f"{RESULTS_DIR}/{exp_name}_constellation_entropy_{entropy_finalized}.png")
    plt.close()

    # ------------------
    # Compute & save CCDF
    # ------------------
    powers, _ = model.compute_power_to_average_power_samples()

    powers = tf.reshape(powers, [-1])
    powers_db = 10 * tf.math.log(powers + 1e-12) / tf.math.log(10.)

    x_sorted = np.sort(powers_db.numpy())
    ccdf = 1.0 - np.arange(1, len(x_sorted) + 1) / len(x_sorted)

    np.savez(
        f"{RESULTS_DIR}/{exp_name}.npz",
        snr_db=snr_db,
        entropy=entropy_finalized,
        papr_db=papr_threshold_db,
        constellation_points=normalized_points,
        constellation_probabilities=probs,
        ccdf_x_db=x_sorted,
        ccdf_y=ccdf
    )

    # np.save(f"{RESULTS_DIR}/{exp_name}_ccdf_x.npy", x_sorted)
    # np.save(f"{RESULTS_DIR}/{exp_name}_ccdf_y.npy", ccdf)

    print(f"Finished {exp_name}")


snr_list = [13]
papr_list = [6.4, 6.8, 7.2, 7.6, 8.0]

N = 32  # Number of classes (symbols)
bits_per_symbol = int(np.log2(N))
parity_k = 2
use_upsampling_filtering = True  # Remove 'false' this after testing without upsampling and filtering
constellation = symbol_points_qam32

tau = 10
# snr_db = 15


# Batch size
batch_size = 10
papr_batch_size = 30
block_size = 4096 - 32 * 4

# No = tf.convert_to_tensor(1/(10**(snr_db/10)),tf.float32)
ebno_db = tf.random.uniform(shape=[batch_size], minval=6.0, maxval=14.0)
No_ = sn.phy.utils.ebnodb2no(ebno_db, num_bits_per_symbol=bits_per_symbol, coderate=1)
No_ = tf.expand_dims(No_, axis=-1)

model = EndToEndSystem_bitwise(N, constellation, papr_batch_size,
                               batch_size, block_size,
                               bits_per_symbol, parity_k,
                               use_upsampling_filtering, tau)
_, _, _, _, _ = model(No_)
initial_weights = model.get_weights()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

RESULTS_DIR = "results_pn_entropy_220GHz_32QAM"
os.makedirs(RESULTS_DIR, exist_ok=True)

for snr_db in snr_list:
    for papr_db in papr_list:
        print('-------------------------------------------------------')
        print('PAPR :', papr_db, ' dB')
        print('-------------------------------------------------------')
        model.set_weights(initial_weights)
        optimizer = tf.keras.optimizers.Adam(0.001)
        run_experiment(snr_db, papr_db)
        tf.keras.backend.clear_session()
        gc.collect()

print("ALL EXPERIMENTS FINISHED")

for snr_db in snr_list:

    plt.figure(figsize=(6, 5))

    for papr_db in papr_list:
        exp_name = f"snr_{snr_db}_papr_{papr_db}"
        data = np.load(f"{RESULTS_DIR}/{exp_name}.npz")

        x = data["ccdf_x_db"]
        y = data["ccdf_y"]

        plt.plot(x, y, label=f"PAPR={papr_db} dB")

    plt.yscale("log")
    plt.xlabel("Normalized Power (dB)")
    plt.ylabel("CCDF = P(X > x)")
    plt.title(f"CCDF — SNR = {snr_db} dB")
    plt.grid(True, which="both")
    plt.xlim(2, 10)
    plt.ylim(1e-5, 1)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/ccdf_snr_{snr_db}.png", dpi=300)
    # plt.show()
