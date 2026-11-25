import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np
from sionna.phy.utils import complex_normal
import matplotlib.pyplot as plt
import sionna as sn
from scipy.io import loadmat
import tensorflow_probability as tfp

def nextpow2(N):
    """ Function for finding the next power of 2 """
    n = 1
    while n < N: n *= 2
    return n

class PhaseNoise(Layer):
    def __init__(self, model_type='10', bandwidth=7e9, fc=140e9, n_samples=1024, **kwargs):
        super().__init__()
        self._real_dtype = tf.dtypes.as_dtype(self.dtype).real_dtype
        self.model_type = model_type
        self.bandwidth = bandwidth
        self.fc = fc
        self.n_samples = n_samples

        samples = 2 * nextpow2(n_samples)
        self.samples = samples  # upper limit

        fvec = (self.bandwidth / 2) * np.linspace(0, 1, self.samples)

        # 3GPP TR38.803 v14.2.0 70GHz PLL phase noise model Table 6.1.10.1-3
        if self.model_type == '10':

            fref = 70e9  # 70 GHz reference
            psd0 = 8894  # 39.49 dB
            fz = [3e3, 396e3, 754e6]
            az = [2.37, 2.7, 2.53]
            fp = [1, 1.55e6, 30e6]
            ap = [3.3, 3.3, 1]

            lf = psd0 * np.ones([1, self.samples])  # PN PSD
            for n in range(3):
                lf = lf * (1 + np.power((fvec / fz[n]), az[n])) / (1 + np.power((fvec / fp[n]), ap[n]))
            # scale to carrier frq
            lf = lf * np.square((self.fc / fref))

        # 3GPP TR38.803 v14.2.0 UE model 1 Table 6.1.11.2-1
        if self.model_type == '11':
            # loop bandwidth
            LBW = 187e3  # Hz

            # all vectors contain values for and in order of
            # REF clk  PLL  VCO 1/f^2  VCO 1/f^3
            # Figure Of Merit
            FOM = [-215, -240, -175, -130]
            # zero-point frequency
            fz = [float('inf'), 1e4, 50.3e6, float('inf')]
            # Power consumption (mW)
            P = [10, 20, 20, 20]
            # zero-point power
            k = [2, 1, 2, 3]

            f_low = np.take(fvec, np.where(fvec <= LBW))
            f_high = np.take(fvec, np.where(fvec > LBW))

            PSD0 = FOM + 20 * np.log10(self.fc) - 10 * np.log10(
                P)  # x dB, actually P/1mW

            fn = lambda f, psd0, fz, k: np.power(10.0, psd0 / 10.0) * (
                    (1.0 + np.power((f / fz), k)) / (1.0 + np.power(f, k)))

            lf = np.concatenate([fn(f_low, PSD0[0], fz[0], k[0]) + fn(f_low, PSD0[1], fz[1], k[1]),
                                 fn(f_high, PSD0[2], fz[2], k[2]) + fn(f_high, PSD0[3], fz[3], k[3])], axis=1)

        # 3GPP TR38.803 v14.2.0 BS model 2
        if self.model_type == '12':
            # loop bandwidth
            LBW = 112e3  # Hz

            # all vectors contain values for and in order of
            # REF clk  PLL  VCO 1/f^2  VCO 1/f^3
            # Figure Of Merit
            FOM = [-240, -245, -187, -130]
            # zero-point frequency
            fz = [float('inf'), 1e4, 8e6, float('inf')]
            # Power consumption (mW)
            P = [10, 20, 50, 50]
            # zero-point power
            k = [2, 1, 2, 3]

            f_low = np.take(fvec, np.where(fvec <= LBW))
            f_high = np.take(fvec, np.where(fvec > LBW))

            PSD0 = FOM + 20 * np.log10(self.fc) - 10 * np.log10(P)  # x dB, actually P/1mW

            fn = lambda f, psd0, fz, k: np.power(10.0, psd0 / 10.0) * (
                    (1.0 + np.power((f / fz), k)) / (1.0 + np.power(f, k)))

            lf = np.concatenate([fn(f_low, PSD0[0], fz[0], k[0]) + fn(f_low, PSD0[1], fz[1], k[1]),
                                 fn(f_high, PSD0[2], fz[2], k[2]) + fn(f_high, PSD0[3], fz[3], k[3])], axis=1)


        if self.model_type == '13':
            # Texas Instruments LMX2595 at 20GHz carrier freq
            # Fig 14 of data sheet revised April 2019

            # measured PN PSD is approximated by polynomials
            # (similar to 3GPP case 10 with n,m = 1,2)
            # f_meas = [1e2 1e3 1e4 1e5 1e6 10e6 40e6 95e6 100e6 fp(end)];
            # psd_meas = [-82.8 -92.2 -100.9 -104.1 -116.5 -140.5 -147.2 -147.7 -147.7 -147.7];

            fref = 20e9  # 70 GHz reference
            psd0 = 10 ** ((-72.0) / 10)  # 39.49 dB
            fz = [3e4, 1.75e7]
            az = [1.4, 2.55]
            fp = [1e1, 3e5]
            ap = [1, 2.95]

            lf = psd0 * np.ones([1, self.samples])  # PN PSD
            for n in range(2):
                lf = lf * (1 + np.power((fvec / fz[n]), az[n])) / (1 + np.power((fvec / fp[n]), ap[n]))
            # scale to carrier frq
            lf = lf * np.square((self.fc / fref))
        # NEU 130 Ghz
        if self.model_type == '16':

            fref = 130e9  # 70 GHz reference
            psd0 = 10 ** ((-66.5) / 10)  # 39.49 dB
            fz = [0.01e6, 6.0e6, 10.0e6]
            az = [2.0,2.0,2.0]
            fp = [0.005e6,0.03e6,5.0e6]
            ap = [2.0,2.0,2.0]

            lf = psd0 * np.ones([1, self.samples])  # PN PSD
            for n in range(3):
                lf = lf * (1 + np.power((fvec / fz[n]), az[n])) / (1 + np.power((fvec / fp[n]), ap[n]))
            # scale to carrier frq
            #lf = lf * np.square((self.fc / fref))

        # remove DC
        lf = np.concatenate([[[0]], lf[:, 1:]], axis=-1)
        self.lf = tf.convert_to_tensor(lf, dtype=tf.float32)


    @tf.function
    def call(self, x):

        # White complex Gaussian sequence
        src_t = complex_normal([tf.shape(x)[0], self.samples])
        src_f = tf.signal.fft(src_t)  # fft(src_t)

        # Apply PSD weights
        f_a = tf.complex(tf.sqrt(self.lf), tf.cast(0.0, tf.float32)) * src_f

        # Symmetric spectrum(PSD is also even symmetric) --> real time - domain process
        f_a = tf.concat([f_a, tf.math.conj(tf.reverse(f_a[:, 1:], axis=[-1]))], axis=-1)
        theta_t = tf.signal.ifft(f_a)  # ifft(f_a)

        # Normalize to have correct variance
        theta_norm = theta_t * tf.cast(tf.sqrt(2 / (1 / self.bandwidth)), dtype=tf.complex64)

        # Take the required length from the middle (fftshift)
        required_samples_start_idx = tf.cast(self.samples / 2, dtype=tf.int32)

        pn_var = tf.math.reduce_variance(
            theta_norm[:, required_samples_start_idx:required_samples_start_idx + self.n_samples])

        phase_noise = tf.math.exp(
            1j * theta_norm[:, required_samples_start_idx:required_samples_start_idx + self.n_samples])
        y = x * phase_noise

        return y, pn_var, phase_noise

    def plot_psd(self):
        # Plot the PSD
        fig = plt.figure()
        fvec = (self.bandwidth / 2) * np.linspace(0, 1, self.samples)
        PSD = 10 * np.log10(np.squeeze(self.lf.numpy()))

        # Main plot
        ax1 = fig.add_subplot(111)
        ax1.semilogx(fvec, PSD)
        ax1.grid(True)
        ax1.set_xlabel('Frequency offset [Hz]')
        ax1.set_ylabel('Phase noise PSD [dBc/Hz]')
        ax1.set_xlim([1e2, 1e10])
        plt.tight_layout()  # Adjust layout for subplots
        #return ax
        plt.show()

class PhaseNoise_Tx:
    def __init__(self, fc, bandwidth, samples,
                 n_iter=1,
                 psd_db=-72.0,
                 fref=20e9,
                 fz=[3e4, 1.75e7],
                 az=[1.4, 2.55],
                 fp=[1e1, 3e5],
                 ap=[1, 2.95]):

        self.fc = fc
        self.bandwidth = bandwidth
        self.samples = samples
        self.n_iter = n_iter
        fvec = (bandwidth / 2) * np.linspace(0, 1, 1 + samples//2)
        fvec = fvec[1:]

        # ----- Build PSD -----
        psd0 = 10 ** (psd_db / 10)

        lf = psd0 * np.ones([1, len(fvec)])

        for n in range(2):
            lf = lf * (1 + np.power((fvec / fz[n]), az[n])) / \
                      (1 + np.power((fvec / fp[n]), ap[n]))

        # scale to carrier frequency
        lf = lf * np.square((fc / fref))

        # Tensor version
        self.lf = tf.constant(lf, dtype=tf.float64)

        self.delta_f = bandwidth / tf.cast(samples, tf.float32)

    def _generate_pn(self, batch_size):
        # Complex Gaussian source in frequency domain
        src_f = (tf.complex(
                    tf.random.normal([batch_size, self.samples]),
                    tf.random.normal([batch_size, self.samples])
                 )
                 * tf.sqrt(tf.cast(self.delta_f, tf.complex64))
                 * self.samples)

        # Mirror PSD
        psd_squared = tf.concat([
            tf.math.conj(tf.reverse(self.lf, axis=[-1])),
            self.lf
        ], axis=-1)

        f_a = tf.cast(src_f, tf.complex128) * tf.complex(
                    tf.sqrt(psd_squared),
                    tf.cast(0.0, tf.float64)
              )

        theta_use = tf.signal.ifft(tf.signal.fftshift(f_a))

        phase_noise = tf.exp(
            1j * tf.cast(tf.sqrt(2.0), tf.complex128) *
            tf.complex(tf.math.real(theta_use),
                       tf.cast(0.0, tf.float64))
        )

        return tf.cast(phase_noise, tf.complex64)

    # ------------------------------------------------------------

    def __call__(self, symbols):
        """
        symbols : [batch_size, samples] complex64
        returns : phase-noise corrupted symbols
        """

        batch_size = tf.shape(symbols)[0]

        for itervalue in range(self.n_iter):
            # tf.print("iter: ", itervalue)

            phase_noise = self._generate_pn(batch_size)

            # Remove carrier if needed (same as your code)
            pn_without_carrier = phase_noise - tf.complex(1.0, 0.0)

            # Corrupt symbols

            print(phase_noise.shape)
            symbols = symbols * phase_noise

        return symbols, pn_without_carrier

class PhaseNoise_Rx:
    def __init__(self, fc, bandwidth, samples,
                 n_iter=1,
                 fom_vec=[-215, -240, -175, -130],
                 fz_vec=[np.inf, 1e4, 50.30e6, np.inf],
                 p_vec=[10, 20, 20, 20],
                 k_vec=[2, 1, 2, 3]):

        self.fc = fc
        self.bandwidth = bandwidth
        self.samples = samples
        self.n_iter = n_iter
        fvec = (bandwidth / 2) * np.linspace(0, 1, 1 + samples//2)
        fvec = fvec[1:]

        # ----- Build PSD -----
        fom = np.asarray(fom_vec)[:, None]
        fz  = np.asarray(fz_vec)[:, None]
        p   = np.asarray(p_vec)[:, None]
        k   = np.asarray(k_vec)[:, None]

        PSD0 = fom + 20 * np.log10(fc) - 10 * np.log10(p)
        S_ref = PSD0 + 10*np.log10((1 + (fvec / fz)**k) / (1 + fvec**k))
        f_loop = 187e3

        S_lin = 10**(S_ref / 10)
        S_out = np.where(
            fvec < f_loop,
            S_lin[0] + S_lin[1],
            S_lin[2] + S_lin[3]
        )
        # S_out_dB = 10 * np.log10(S_out)
        # Tensor version
        self.lf = tf.constant(S_out, dtype=tf.float64)

        self.delta_f = bandwidth / tf.cast(samples, tf.float32)

    def _generate_pn(self, batch_size):
        # Complex Gaussian source in frequency domain
        src_f = (tf.complex(
                    tf.random.normal([batch_size, self.samples]),
                    tf.random.normal([batch_size, self.samples])
                 )
                 * tf.sqrt(tf.cast(self.delta_f, tf.complex64))
                 * self.samples)

        # Mirror PSD
        psd_squared = tf.concat([
            tf.math.conj(tf.reverse(self.lf, axis=[-1])),
            self.lf
        ], axis=-1)

        f_a = tf.cast(src_f, tf.complex128) * tf.complex(
                    tf.sqrt(psd_squared),
                    tf.cast(0.0, tf.float64)
              )

        theta_use = tf.signal.ifft(tf.signal.fftshift(f_a))

        phase_noise = tf.exp(
            1j * tf.cast(tf.sqrt(2.0), tf.complex128) *
            tf.complex(tf.math.real(theta_use),
                       tf.cast(0.0, tf.float64))
        )

        return tf.cast(phase_noise, tf.complex64)

    # ------------------------------------------------------------

    def __call__(self, symbols):
        """
        symbols : [batch_size, samples] complex64
        returns : phase-noise corrupted symbols
        """

        batch_size = tf.shape(symbols)[0]

        for itervalue in range(self.n_iter):
            # tf.print("iter: ", itervalue)

            phase_noise = self._generate_pn(batch_size)

            # Remove carrier if needed (same as your code)
            pn_without_carrier = phase_noise - tf.complex(1.0, 0.0)

            # Corrupt symbols

            print(phase_noise.shape)
            symbols = symbols * phase_noise

        return symbols, pn_without_carrier


class PTRS_Adder:
    def __init__(self, path_to_ptrs_mat , num_ptrs_blocks, symbols_per_ptrs_block, block_size_ptrs, batch_size):
        data = loadmat(path_to_ptrs_mat)
        self.pilots = tf.cast(tf.reshape(tf.convert_to_tensor(data["pilots"]),[num_ptrs_blocks*symbols_per_ptrs_block]), dtype = tf.complex64)

        self.ptrs_indices, self.data_indices = self.get_ptrs_block_indices(block_size_ptrs,num_ptrs_blocks,symbols_per_ptrs_block)
        # ptrs_indices_tensor = tf.convert_to_tensor(ptrs_indices)

        ptrs_zero_row = tf.zeros_like(self.ptrs_indices)
        ptrs_indices_2d = tf.transpose(tf.stack([ptrs_zero_row, self.ptrs_indices], axis=1),[0,1])
        self.data_indices_exp = tf.expand_dims(self.data_indices, axis=1)

        rgrid = tf.zeros([1,block_size_ptrs], dtype=tf.complex64)
        rgrid = tf.tensor_scatter_nd_update(
            rgrid,
            ptrs_indices_2d,
            self.pilots
        )
        self.resource_grid = tf.tile(rgrid, [batch_size, 1])

    def __call__(self, x_mod):

        # print('data_indices_exp :',self.data_indices_exp.shape)
        # print('resource grid :',tf.transpose(self.resource_grid,[1, 0]).shape)
        # print('x_mod :', tf.transpose(x_mod,[1, 0]).shape)

        x_mod_w_ptrs = tf.transpose(tf.tensor_scatter_nd_update(
            tf.transpose(self.resource_grid,[1, 0]),
            self.data_indices_exp,
            tf.transpose(x_mod,[1, 0])
        ),[1, 0])

        return x_mod_w_ptrs

    def get_ptrs_block_indices(self, block_length, ptrs_blocks_per_block, samples_per_block):
        edge = np.mod(block_length-ptrs_blocks_per_block*samples_per_block,ptrs_blocks_per_block-1)
        block_distance = (block_length-samples_per_block*ptrs_blocks_per_block-edge)/(ptrs_blocks_per_block-1)
        ptrs_block_indices = np.arange(np.floor(edge/2), np.floor(edge/2)+samples_per_block)

        for n in range(1,ptrs_blocks_per_block):
            ptrs_block_indices = np.concatenate([ptrs_block_indices, np.arange(ptrs_block_indices[-1]+block_distance+1, ptrs_block_indices[-1]+samples_per_block+block_distance+1)])

        all_indices = np.arange(block_length)
        data_indices = np.setdiff1d(all_indices, ptrs_block_indices)

        return ptrs_block_indices.astype(int), data_indices.astype(int)


class PN_Compensation:
    def __init__(self, ptrs_indices, data_indices, block_size_ptrs, pilots, batch_size):
        self.x_dense = tf.cast(tf.range(0, block_size_ptrs),tf.float32)
        self.ptrs_avg_indices = tf.cast(tf.reduce_mean((tf.reshape(ptrs_indices,[-1, 32, 4])), axis =-1), tf.float32)
        self.pilots = pilots
        self.ptrs_indices = ptrs_indices
        self.data_indices = data_indices
        self.batch_size = batch_size

    def __call__(self, y_down):
        y_ptrs = tf.gather(y_down, self.ptrs_indices, axis=1)
        phase_estimate = y_ptrs * tf.math.conj(self.pilots)/tf.cast(tf.square(tf.abs(self.pilots)), tf.complex64)
        phase_estimate_mean = tf.reduce_mean(tf.reshape(phase_estimate,[self.batch_size, 32,4]), axis=-1)
        phase_estimate_interp_r = tfp.math.interp_regular_1d_grid(x=self.x_dense, x_ref_min = self.ptrs_avg_indices[0,0]
                                                                ,x_ref_max = self.ptrs_avg_indices[0,-1],
                                                                 y_ref= tf.math.real(phase_estimate_mean))
        phase_estimate_interp_i = tfp.math.interp_regular_1d_grid(x=self.x_dense, x_ref_min = self.ptrs_avg_indices[0,0]
                                                                ,x_ref_max = self.ptrs_avg_indices[0,-1],
                                                                 y_ref= tf.math.imag(phase_estimate_mean))
        y_wpn_free = y_down / tf.complex(phase_estimate_interp_r,phase_estimate_interp_i)
        y_data = tf.gather(y_wpn_free, self.data_indices, axis=1)
        return y_wpn_free, y_data
