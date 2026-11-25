from sionna.phy.block import Block, Object
from sionna.phy.mapping import Constellation, SymbolLogits2LLRs
from sionna.phy.utils import split_dim, expand_to_rank
from sionna.phy import config
from tensorflow.keras import Layer,Input
from tensorflow.keras.layers import Dense
from itertools import combinations
from math import comb 

import numpy as np
import tensorflow as tf

PI = tf.constant(np.pi, tf.float32)

def shaping_encoder_func_adjusted(ns: int, ks: int, weight_counts):
    M = 1 << ks  # 2**ks


    if sum(weight_counts.values()) != M:
        raise ValueError(
            f"Selected codewords must sum to M={M}, "
            f"but got {sum(weight_counts.values())}"
        )

    codebook = []

    # Build the codebook according to the specified weight distribution.
    for weight, count in weight_counts.items():
        available = list(combinations(range(ns), weight))

        if count > len(available):
            raise ValueError(
                f"Cannot select {count} codewords of weight {weight}; "
                f"only {len(available)} are available."
            )

        for combo in available[:count]:
            cw = np.zeros(ns, dtype=np.uint8)
            cw[list(combo)] = 1
            codebook.append(cw)

    codebook = np.vstack(codebook)  # Shape: (M, ns)

    # Average probability of a zero bit.
    p0 = 1.0 - codebook.sum() / (M * ns)

    # Label books.
    keybook = np.zeros((M, ks), dtype=np.uint8)
    shapingenclabels = []
    shapingdeclabels = []

    encoder_map = {}
    decoder_map = {}

    for i in range(M):
        # Input bits in left-MSB order.
        input_bits = np.array(
            list(np.binary_repr(i, width=ks)),
            dtype=np.uint8
        )

        keybook[i] = input_bits

        encoder_map[tuple(input_bits)] = codebook[i]
        decoder_map[tuple(codebook[i])] = input_bits

        shapingenclabels.append(
            ''.join(map(str, input_bits.tolist()))
        )
        shapingdeclabels.append(
            ''.join(map(str, codebook[i].tolist()))
        )

    return (
        codebook,
        keybook,
        encoder_map,
        decoder_map,
        p0,
        shapingenclabels,
        shapingdeclabels
    )

def shaping_encoder_func(ns: int, ks: int):

    M = 1 << ks                      # 2^ks
    codebook  = []
    weight    = 0
    # -------- build codewords sorted by increasing Hamming weight -------- #
    while len(codebook) < M:
        for combo in combinations(range(ns), weight):
            cw = np.zeros(ns, dtype=np.uint8)
            cw[list(combo)] = 1
            codebook.append(cw)
            if len(codebook) == M:
                break
        weight += 1

    codebook = np.vstack(codebook)           # (M, ns)
    # -------------------------- statistics -------------------------------- #
    p0 = 1.0 - codebook.sum() / (M * ns)
    # --------------------------- label books ------------------------------ #
    keybook  = np.zeros((M, ks), dtype=np.uint8)
    shapingenclabels = []
    shapingdeclabels = []
    encoder_map = {}
    decoder_map = {}
    for i in range(M):
        # input bits (ks) : left‑MSB order
        input_bits = np.array(list(np.binary_repr(i, width=ks)), dtype=np.uint8)
        keybook[i] = input_bits

        # dictionaries use tuples as hashable keys
        encoder_map[tuple(input_bits)] = codebook[i]

        output_bits = codebook[i]
        decoder_map[tuple(output_bits)] = input_bits

        shapingenclabels.append(''.join(map(str, input_bits.tolist())))
        shapingdeclabels.append(''.join(map(str, output_bits.tolist())))

    return (codebook,
            keybook,
            encoder_map,
            decoder_map,
            p0,
            shapingenclabels,
            shapingdeclabels)

def wrap_to_pi(angle):
    return tf.math.floormod(angle + PI, 2.0 * PI) - PI

def low_pn_approx_pdf(y, x, sigma_r, sigma_p):

    y = tf.cast(y, tf.complex64)
    x = tf.cast(x, tf.complex64)

    r = tf.abs(y)
    theta = tf.math.angle(y)

    l = tf.abs(x)
    alpha = tf.math.angle(x)
    alpha_c = tf.complex(alpha,0.0)

    r = tf.cast(r, tf.float32)
    l = tf.cast(l, tf.float32)

    sigma_r = tf.cast(sigma_r, tf.float32)
    sigma_p = tf.cast(sigma_p, tf.float32)

    complx_number = y*tf.exp(-1j*alpha_c)

    log_norm = -tf.math.log(2.0 * PI * sigma_r * tf.sqrt((l**2)*sigma_p**2 + sigma_r**2))

    log_real = -(tf.abs(tf.math.real(complx_number) - l) ** 2) / (2.0 * sigma_r ** 2)
    log_imag = -(tf.math.imag(complx_number) ** 2) / (2.0 * ((l**2)*sigma_p**2 + sigma_r**2))

    return log_norm + log_real + log_imag

def log_polar_approx_pdf(y, x, sigma_r, sigma_p, eps=1e-8):

    y = tf.cast(y, tf.complex64)
    x = tf.cast(x, tf.complex64)

    r = tf.abs(y)
    theta = tf.math.angle(y)

    l = tf.abs(x)
    alpha = tf.math.angle(x)

    dtheta = wrap_to_pi(theta - alpha)

    r = tf.cast(r, tf.float32)
    l = tf.cast(l, tf.float32)
    dtheta = tf.cast(dtheta, tf.float32)

    sigma_r = tf.cast(sigma_r, tf.float32)
    sigma_p = tf.cast(sigma_p, tf.float32)
    eps = tf.cast(eps, tf.float32)

    log_norm = -tf.math.log(2.0 * PI * sigma_r * tf.sqrt((l**2)*sigma_p**2 + sigma_r**2))
    # log_jac = -tf.math.log(tf.maximum(r, eps))
    log_radial = -(tf.abs(r - l) ** 2) / (2.0 * sigma_r ** 2)
    log_angular = -(dtheta ** 2) / (2.0 * (sigma_p**2 + (1/(l**2))*sigma_r**2))

    return log_norm + log_radial + log_angular

def log_polar_approx_pdf(y, x, sigma_r, sigma_p, eps=1e-8):

    y = tf.cast(y, tf.complex64)
    x = tf.cast(x, tf.complex64)

    r = tf.abs(y)
    theta = tf.math.angle(y)

    l = tf.abs(x)
    alpha = tf.math.angle(x)

    dtheta = wrap_to_pi(theta - alpha)

    r = tf.cast(r, tf.float32)
    l = tf.cast(l, tf.float32)
    dtheta = tf.cast(dtheta, tf.float32)

    sigma_r = tf.cast(sigma_r, tf.float32)
    sigma_p = tf.cast(sigma_p, tf.float32)
    eps = tf.cast(eps, tf.float32)

    log_norm = -tf.math.log(2.0 * PI * sigma_r * tf.sqrt((l**2)*sigma_p**2 + sigma_r**2))
    # log_jac = -tf.math.log(tf.maximum(r, eps))
    log_radial = -(tf.abs(r - l) ** 2) / (2.0 * sigma_r ** 2)
    log_angular = -(dtheta ** 2) / (2.0 * (sigma_p**2 + (1/(l**2))*sigma_r**2))

    return log_norm + log_radial + log_angular

def log_gaussian_pdf(y, x, sigma_r, eps=1e-8):
    # y, x: complex tensors (same shape)
    # sigma2: noise variance (scalar or broadcastable tensor), E[|n|^2]
    y = tf.cast(y, tf.complex64)
    x = tf.cast(x, tf.complex64)
    sigma2 = tf.cast(sigma_r**2, tf.float32)

    diff = y - x
    sq_dist = tf.square(tf.abs(diff))  # |y - x|^2

    log_norm = -tf.math.log(tf.cast(PI, tf.float32) * sigma2)
    log_likelihood = log_norm - sq_dist / sigma2

    return log_likelihood

class Demapper_optimized(Block):

    def __init__(self,
                 demapping_method,
                 constellation_type=None,
                 num_bits_per_symbol=None,
                 constellation=None,
                 hard_out=False,
                 precision=None,
                 pn_var = None,
                 condition = None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)

        # Create constellation object
        self._constellation = Constellation.check_or_create(
                                constellation_type=constellation_type,
                                num_bits_per_symbol=num_bits_per_symbol,
                                constellation=constellation,
                                precision=precision)

        self.num_bits_per_symbol = self._constellation.num_bits_per_symbol

        self._logits2llrs = SymbolLogits2LLRs(demapping_method,
                                              self.num_bits_per_symbol,
                                              hard_out=hard_out,
                                              precision=precision,
                                              **kwargs)

        self._no_threshold = tf.cast(np.finfo(self.rdtype.as_numpy_dtype).tiny,
                                     self.rdtype)

        self.condition = condition
        self.pn_var = pn_var

    def call(self, y, no, const_points, prior=None):

        self.constellation_points = const_points

        # Reshape constellation points to [1,...1,num_points]
        points_shape = [1]*y.shape.rank + self.constellation_points.shape
        points = tf.reshape(self.constellation_points, points_shape)

        if self.condition == 'awgn':
            # Compute squared distances from y to all points
            # shape [...,n,num_points]
            squared_dist = tf.pow(tf.abs(tf.expand_dims(y, axis=-1) - points), 2)
    
            # Add a dummy dimension for broadcasting. This is not needed when no
            # is a scalar, but also does not do any harm.
            no = tf.expand_dims(no, axis=-1)
            # Deal with zero or very small values.
            no = tf.math.maximum(no, self._no_threshold)
            # Compute exponents
            exponents = -squared_dist/no

        elif self.condition == 'lowpn':
            sigma_r = tf.math.sqrt(no)
            sigma_p = self.pn_var
            y_exp = tf.expand_dims(y, axis=2)          # [B,T,1]
            sym_exp = tf.expand_dims(tf.expand_dims(const_points, axis=0), axis=0)  # [1,1,M]
            exponents = low_pn_approx_pdf(y_exp, sym_exp, sigma_r, sigma_p)   # [B,T,M]

        elif self.condition == 'highsnr':
            sigma_r = tf.math.sqrt(no)
            sigma_p = self.pn_var
            y_exp = tf.expand_dims(y, axis=2)          # [B,T,1]
            sym_exp = tf.expand_dims(tf.expand_dims(const_points, axis=0), axis=0)  # [1,1,M]
            # exponents = log_gaussian_pdf(y_exp, sym_exp, sigma_r, eps=1e-8)
            exponents = log_polar_approx_pdf(y_exp, sym_exp, sigma_r, sigma_p)   # [B,T,M]

        llr = self._logits2llrs(exponents, prior)

        # Reshape LLRs to [...,n*num_bits_per_symbol]
        out_shape = tf.concat([tf.shape(y)[:-1],
                               [y.shape[-1] * \
                                self.num_bits_per_symbol]], 0)
        llr_reshaped = tf.reshape(llr, out_shape)

        return llr_reshaped

class Demodulator_bitwise_awgn(Layer):
    def __init__(self, m, batch_size):
        super(Demodulator_bitwise_awgn, self).__init__()
        self.dense1 = Dense(128, activation='relu')  # First Dense Layer
        self.dense2 = Dense(128, activation='relu')  # Second Dense Layer
        self.dense3 = Dense(m, activation='linear') # Output Dense Layer
        self.m = m
        self.batch_size = batch_size


    def call(self, y, prior, ebno_db):
        # Ensure complex-valued input is split into real and imaginary parts

        ebno_db = tf.broadcast_to(ebno_db, (y.shape[0], 1))

        y_reshaped = tf.reshape(y,[self.batch_size * y.shape[1], 1])
        prior = tf.reshape(prior, [self.batch_size * y.shape[1], self.m])
        # max_abs = tf.reduce_max(tf.abs(prior)) + 1e-8  # Add epsilon to avoid divide-by-zero
        # prior = prior / max_abs
        y_real = tf.math.real(y_reshaped)
        y_imag = tf.math.imag(y_reshaped)

        brd_snr_tensor = tf.reshape(tf.tile(ebno_db,(1,y.shape[1])), (-1, 1))
        
        # Concatenate real, imaginary, and SNR tensors along the last dimension
        stack_input = tf.concat([y_real, y_imag, prior, brd_snr_tensor], axis=1) 

        # Pass the stacked input through the dense layers
        x = self.dense1(stack_input)
        x = self.dense2(x)
        logits = self.dense3(x)
        logits_step1 = tf.reshape(logits,[self.batch_size, y.shape[1],self.m])
        logits = tf.reshape(logits_step1,[self.batch_size, y.shape[1]*self.m])

        return logits


class Demodulator_bitwise(Layer):
    def __init__(self, m, batch_size, RBF_blocksize):
        super(Demodulator_bitwise, self).__init__()
        self.dense1 = Dense(128, activation='relu')  # First Dense Layer
        self.dense2 = Dense(128, activation='relu')  # Second Dense Layer
        self.dense3 = Dense(m, activation='linear') # Output Dense Layer
        self.m = m
        self.batch_size = batch_size
        self.RBF_blocksize = RBF_blocksize


    def call(self, y, prior, h_hat, ebno_db, sigma2_eff):
        # Ensure complex-valued input is split into real and imaginary parts
        # ebno_db = tf.broadcast_to(ebno_db, (y.shape[0], 1))

        brd_h_tensor = tf.reshape(tf.tile(h_hat,(1,1,self.RBF_blocksize)), (-1, 1))
        brd_snr_tensor = tf.reshape(tf.tile(ebno_db,(1,y.shape[1])), (-1, 1))
        # print('brd_h_tensor.shape',brd_h_tensor.shape)

        h_real = tf.math.real(brd_h_tensor)
        h_imag = tf.math.imag(brd_h_tensor)

        # print('h_real.shape',h_real.shape)

        y_reshaped = tf.reshape(y,[self.batch_size * y.shape[1], 1])
        sigma2_eff_reshaped = tf.reshape(sigma2_eff,[self.batch_size * y.shape[1], 1])
        ebno_db_eff_reshaped = tf.reshape(ebno_db,[self.batch_size * y.shape[1], 1])        
        prior = tf.reshape(prior, [self.batch_size * y.shape[1], self.m])
        # max_abs = tf.reduce_max(tf.abs(prior)) + 1e-8  # Add epsilon to avoid divide-by-zero
        # prior = prior / max_abs
        y_real = tf.math.real(y_reshaped)
        y_imag = tf.math.imag(y_reshaped)

        # Concatenate real, imaginary, and SNR tensors along the last dimension
        stack_input = tf.concat([y_real, y_imag, prior, ebno_db_eff_reshaped], axis=1) 

        # Pass the stacked input through the dense layers
        x = self.dense1(stack_input)
        x = self.dense2(x)
        logits = self.dense3(x)
        logits_step1 = tf.reshape(logits,[self.batch_size, y.shape[1],self.m])
        logits = tf.reshape(logits_step1,[self.batch_size, y.shape[1]*self.m])

        return logits

class Shaping_encoder(Block):
    # pylint: disable=line-too-long
    
    def __init__(self,
                 codebook,
                 ks=None,
                 precision=None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)

        self._ks = ks
        self._codebook = codebook

    def call(self, bits):

        bits = tf.cast(bits, dtype=tf.int32)
        n1 = int(bits.shape[-1]/self._ks)
        new_shape = [n1 , self._ks]
        bits = split_dim(bits, new_shape, axis=tf.rank(bits)-1)
        test_bit_positions = tf.cast(tf.range(self._ks-1, -1, -1), dtype=tf.int32)
        shifted_bits = tf.bitwise.left_shift(bits, test_bit_positions)
        int_rep = tf.reduce_sum(shifted_bits, axis=-1)
        x = tf.gather(self._codebook, int_rep, axis=0)
        # print(x.shape)
        x1 = tf.reshape(x, [-1, x.shape[1]*x.shape[2]])  
        # print(x1.shape)

        return x1

class Shaping_decoder_forward(Block):
    # pylint: disable=line-too-long
    def __init__(self,
                 method,
                 ks,
                 codebook,
                 *,
                 hard_out=False,
                 precision=None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)
        assert method in ("app","maxlog"), "Unknown demapping method"
        self._method = method
        self._hard_out = hard_out
        self._ks = ks
        cb_size = int(2**ks)

        # Array composed of binary representations of all symbols indices
        a = np.zeros([cb_size, ks])
        for i in range(0, cb_size):
            a[i,:] = np.array(list(np.binary_repr(i, ks)),
                              dtype=np.int32)

        # Compute symbol indices for which the bits are 0 or 1
        c0 = np.zeros([int(cb_size/2), ks])
        c1 = np.zeros([int(cb_size/2), ks])

        for i in range(ks-1,-1,-1):
            c0[:,i] = np.where(a[:,i]==0)[0]
            c1[:,i] = np.where(a[:,i]==1)[0]
        self._c0 = tf.constant(c0, dtype=tf.int32) # Symbols with ith bit=0
        self._c1 = tf.constant(c1, dtype=tf.int32) # Symbols with ith bit=1

        # [num_points, num_bits_per_symbol]
        a = 2*a-1
        self._a = tf.constant(a, dtype=self.rdtype)
        self._codebook = tf.constant(2*np.array(codebook, dtype=np.int32)-1, dtype=self.rdtype)

        # Determine the reduce function for LLR computation
        if self._method == "app":
            self._reduce = tf.reduce_logsumexp
        else:
            self._reduce = tf.reduce_max

    @property
    def k_s(self):
        return self._ks

    def call(self, llr_a_c, llr_a_d):

        llr_a_d = tf.expand_dims(llr_a_d, axis=-2)
        llr_a_c = tf.expand_dims(llr_a_c, axis=-2)        
        a = expand_to_rank(self._a, tf.rank(llr_a_d), axis=0)
        codebook = expand_to_rank(self._codebook, tf.rank(llr_a_c), axis=0)
        exp_ps_c = tf.reduce_sum(tf.math.log_sigmoid(codebook*llr_a_c), axis=-1)
        log_sigmoid_sum = tf.math.log_sigmoid(a*llr_a_d)
        exp_ps = tf.reduce_sum(log_sigmoid_sum, axis=-1)# - log_sigmoid_sum
        exp_ps_expanded = tf.expand_dims(exp_ps, axis=-1)
        exp_ps = exp_ps_expanded - log_sigmoid_sum

        # Transpose c0 to match tf.gather format
        indices0 = tf.transpose(self._c0)
        indices1 = tf.transpose(self._c1)

        gathered_per_feature0 = []  # Will hold 5 tensors of shape (6, 6, 16)
        gathered_per_feature1 = []  # Will hold 5 tensors of shape (6, 6, 16)
        
        for i in range(self._ks):
            gathered0 = tf.gather(exp_ps[..., i], indices0[i], axis=2)  # shape: (6, 6, 16)
            gathered_per_feature0.append(gathered0)
            gathered1 = tf.gather(exp_ps[..., i], indices1[i], axis=2)  # shape: (6, 6, 16)
            gathered_per_feature1.append(gathered1)
            
        # Stack the result back along last axis
        exp_ps0 = tf.stack(gathered_per_feature0, axis=-1)  # shape: (6, 6, 16, 5)
        exp_ps1 = tf.stack(gathered_per_feature1, axis=-1)  # shape: (6, 6, 16, 5)
        
        exp_ps0_c = tf.gather(exp_ps_c, self._c0, axis=-1)
        exp_ps1_c = tf.gather(exp_ps_c, self._c1, axis=-1)

        llr = self._reduce(exp_ps1 + exp_ps1_c, axis=-2)\
                - self._reduce(exp_ps0 + exp_ps0_c, axis=-2)

        llr = tf.reshape(llr, [llr.shape[0], -1])

        if self._hard_out:
            return hard_decisions(llr)
        else:
            return llr


class Shaping_decoder_backward(Block):
    # pylint: disable=line-too-long
    def __init__(self,
                 method,
                 ks,
                 ns,
                 codebook,
                 *,
                 hard_out=False,
                 precision=None,
                 **kwargs):
        super().__init__(precision=precision, **kwargs)
        assert method in ("app","maxlog"), "Unknown demapping method"
        self._method = method
        self._hard_out = hard_out
        self._ns = ns
        self._ks = ks
        cb_size = int(2**ks)

        np_codebook = np.array(codebook, dtype=np.int32)

        # Array composed of binary representations of all symbols indices
        a = np.zeros([cb_size, ks])
        for i in range(0, cb_size):
            a[i,:] = np.array(list(np.binary_repr(i, ks)),
                              dtype=np.int32)

        c0 = []
        c1 = []
        for i in range(ns):
            c0_i = np.where(np_codebook[:, i] == 0)[0]
            c1_i = np.where(np_codebook[:, i] == 1)[0]
            c0.append(c0_i)
            c1.append(c1_i)

        self._c0 = c0
        self._c1 = c1

        # [num_points, num_bits_per_symbol]
        a = 2*a-1
        self._a = tf.constant(a, dtype=self.rdtype)
        self._codebook = tf.constant(2*np_codebook-1, dtype=self.rdtype)

        # Determine the reduce function for LLR computation
        if self._method == "app":
            self._reduce = tf.reduce_logsumexp
        else:
            self._reduce = tf.reduce_max

    @property
    def k_s(self):
        return self._ks

    def call(self, llr_a_c, llr_a_d):

        llr_a_d = tf.expand_dims(llr_a_d, axis=-2)
        llr_a_c = tf.expand_dims(llr_a_c, axis=-2)        
        a = expand_to_rank(self._a, tf.rank(llr_a_d), axis=0)
        
        codebook = expand_to_rank(self._codebook, tf.rank(llr_a_c), axis=0)
        
        exp_ps_d = tf.reduce_sum(tf.math.log_sigmoid(a*llr_a_d), axis=-1)
        # print('exp_ps_d.shape',exp_ps_d.shape)
        log_sigmoid_sum = tf.math.log_sigmoid(codebook*llr_a_c)
        exp_ps = tf.reduce_sum(log_sigmoid_sum, axis=-1)# - log_sigmoid_sum
        exp_ps_expanded = tf.expand_dims(exp_ps, axis=-1)

        exp_ps = exp_ps_expanded - log_sigmoid_sum

        indices0 = tf.ragged.constant(self._c0, dtype=tf.int32)  # shape: (ks, None)
        indices1 = tf.ragged.constant(self._c1, dtype=tf.int32)  # shape: (ks, None)

        gathered_per_feature0 = []  # Will hold 5 tensors of shape (6, 6, 16)
        gathered_per_feature1 = []  # Will hold 5 tensors of shape (6, 6, 16)

        codebook_gathered_per_feature0 = []  # Will hold 5 tensors of shape (6, 6, 16)
        codebook_gathered_per_feature1 = []  # Will hold 5 tensors of shape (6, 6, 16)

        for i in range(self._ns):

            codebook_gathered0 = tf.gather(exp_ps_d, indices0[i], axis=2)  # shape: (6, 6, 16)
            codebook_gathered1 = tf.gather(exp_ps_d, indices1[i], axis=2)  # shape: (6, 6, 16)

            gathered0 = tf.gather(exp_ps[..., i], indices0[i], axis=2)  # shape: (6, 6, 16)
            gathered1 = tf.gather(exp_ps[..., i], indices1[i], axis=2)  # shape: (6, 6, 16)
                        
            summed0 = self._reduce(gathered0+codebook_gathered0, axis=-1, keepdims=True)
            summed1 = self._reduce(gathered1+codebook_gathered1, axis=-1, keepdims=True)
            summed1 = tf.maximum(summed1, -27.6310)  # Avoid -inf in log
            
            gathered_per_feature0.append(summed0)
            gathered_per_feature1.append(summed1)
            
        # Stack the result back along last axis
        exp_ps0 = tf.stack(gathered_per_feature0, axis=-1)  # shape: (6, 6, 16, 5)
        exp_ps1 = tf.stack(gathered_per_feature1, axis=-1)  # shape: (6, 6, 16, 5)
        
        llr = exp_ps1 - exp_ps0

        llr = tf.reshape(llr, [llr.shape[0], -1])

        if self._hard_out:
            return hard_decisions(llr)
        else:
            return llr

class Modulator(Layer):

    def __init__(self, constellation, p0, num_bits_per_symbol, **kwargs):
        super(Modulator, self).__init__(**kwargs)

        symbol_points = constellation.points

        self.num_bits_per_symbol = num_bits_per_symbol #np.log2(symbol_points.shape[0])
        # print(self.num_bits_per_symbol)

        self._bit_positions = tf.cast(tf.range(self.num_bits_per_symbol-1, -1, -1), dtype=tf.int32)

        self.constellation_real_init = np.real(symbol_points)
        self.constellation_imag_init = np.imag(symbol_points)

        self.constellation = tf.complex(self.constellation_real_init,self.constellation_imag_init)

        labels = np.arange(self.constellation.shape[0])
        # Generate binary labels
        bin_labels = ((labels[:, None] & (1 << np.arange(self.num_bits_per_symbol - 1, -1, -1))) > 0).astype(int)
        first_bits = bin_labels[:, 0]  # First (MSB) bit
        # fifth_bits = bin_labels[:, 4]  # Fifth (MSB) bit
        
        self.p0 = tf.cast(p0,tf.float32)
        # self.g = 2
        # self.p0 = p0

        # Assign probabilities based on MSB
        self.prob = tf.where(first_bits == 0, self.p0 / (2 ** (self.num_bits_per_symbol - 1)), (1 - self.p0) / (2 ** (self.num_bits_per_symbol - 1)))

        # prob1 = tf.where(first_bits == 0, self.p0 , (1 - self.p0))
        # prob2 = tf.where(fifth_bits == 0, self.p0 , (1 - self.p0))
        # self.prob = tf.multiply(prob1,prob2)
        # self.prob = self.prob/(2 ** (self.num_bits_per_symbol - self.g))
    

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

    def call(self, bits):

        self.constellation = tf.complex(self.constellation_real,self.constellation_imag)

        normalization_constant = tf.sqrt(
            tf.reduce_sum(self.prob * tf.abs(self.constellation) ** 2)
        )
        # print('normalization_constant',normalization_constant.shape)

        normalization_constant = tf.expand_dims(tf.complex(normalization_constant,0.0),-1)

        self.constellation  = self.constellation / normalization_constant

        # Convert to int32
        bits = tf.cast(bits, dtype=tf.int32)

        # Reshape last dimensions to the desired format
        # n1 = int(bits.shape[-1]/self.num_bits_per_symbol)
        # new_shape = [n1 , self.num_bits_per_symbol]
        # bits = split_dim(bits, new_shape, axis=tf.rank(bits)-1)

        # Use bitwise left shift to compute powers of two
        shifted_bits = tf.bitwise.left_shift(bits, self._bit_positions)

        # Compute the integer representation using bitwise operations
        int_rep = tf.reduce_sum(shifted_bits, axis=-1)

        # Map integers to constellation symbols
        x = tf.gather(self.constellation, int_rep, axis=0)

        return x, self.constellation

class Modulator_g2(Layer):

    def __init__(self, constellation, p0, num_bits_per_symbol, **kwargs):
        super(Modulator_g2, self).__init__(**kwargs)

        symbol_points = constellation.points

        self.num_bits_per_symbol = num_bits_per_symbol #np.log2(symbol_points.shape[0])
        # print(self.num_bits_per_symbol)

        self._bit_positions = tf.cast(tf.range(self.num_bits_per_symbol-1, -1, -1), dtype=tf.int32)

        self.constellation_real_init = np.real(symbol_points)
        self.constellation_imag_init = np.imag(symbol_points)

        self.constellation = tf.complex(self.constellation_real_init,self.constellation_imag_init)

        labels = np.arange(self.constellation.shape[0])
        # Generate binary labels
        bin_labels = ((labels[:, None] & (1 << np.arange(self.num_bits_per_symbol - 1, -1, -1))) > 0).astype(int)
        first_bits = bin_labels[:, 0]  # First (MSB) bit
        fifth_bits = bin_labels[:, 5]  # Sixth (MSB) bit
        
        self.p0 = tf.cast(p0,tf.float32)
        self.g = 2
        # self.p0 = p0

        # Assign probabilities based on MSB
        # self.prob = tf.where(first_bits == 0, self.p0 / (2 ** (self.num_bits_per_symbol - 1)), (1 - self.p0) / (2 ** (self.num_bits_per_symbol - 1)))

        prob1 = tf.where(first_bits == 0, self.p0 , (1 - self.p0))
        prob2 = tf.where(fifth_bits == 0, self.p0 , (1 - self.p0))
        self.prob = tf.multiply(prob1,prob2)
        self.prob = self.prob/(2 ** (self.num_bits_per_symbol - self.g))
    

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

    def call(self, bits):

        self.constellation = tf.complex(self.constellation_real,self.constellation_imag)

        normalization_constant = tf.sqrt(
            tf.reduce_sum(self.prob * tf.abs(self.constellation) ** 2)
        )
        # print('normalization_constant',normalization_constant.shape)

        normalization_constant = tf.expand_dims(tf.complex(normalization_constant,0.0),-1)

        self.constellation  = self.constellation / normalization_constant

        # Convert to int32
        bits = tf.cast(bits, dtype=tf.int32)

        # Reshape last dimensions to the desired format
        # n1 = int(bits.shape[-1]/self.num_bits_per_symbol)
        # new_shape = [n1 , self.num_bits_per_symbol]
        # bits = split_dim(bits, new_shape, axis=tf.rank(bits)-1)

        # Use bitwise left shift to compute powers of two
        shifted_bits = tf.bitwise.left_shift(bits, self._bit_positions)

        # Compute the integer representation using bitwise operations
        int_rep = tf.reduce_sum(shifted_bits, axis=-1)

        # Map integers to constellation symbols
        x = tf.gather(self.constellation, int_rep, axis=0)

        return x, self.constellation

class Shaping_dec_forward_neural(Layer):
    def __init__(self, ks, ns, batch_size):
        super(Shaping_dec_forward_neural, self).__init__()
        self.dense1 = Dense(128, activation='relu')  # First Dense Layer
        self.dense2 = Dense(128, activation='relu')  # Second Dense Layer
        self.dense3 = Dense(ks, activation='linear') # Output Dense Layer
        self.ks = ks
        self.ns = ns
        self.batch_size = batch_size

    def call(self, llr_a_c, llr_a_d):
        # Ensure complex-valued input is split into real and imaginary parts

        llr_a_c_reshaped = tf.reshape(llr_a_c,[self.batch_size * llr_a_c.shape[1], self.ns])
        llr_a_d_reshaped = tf.reshape(llr_a_d, [self.batch_size * llr_a_c.shape[1], self.ks])

        stack_input = tf.concat([llr_a_c_reshaped, llr_a_d_reshaped], axis=1)

        x = self.dense1(stack_input)
        x = self.dense2(x)
        logits = self.dense3(x)

        logits_step1 = tf.reshape(logits,[self.batch_size, llr_a_c.shape[1], self.ks])
        logits = tf.reshape(logits_step1,[self.batch_size, llr_a_c.shape[1]*self.ks])

        return logits

class Shaping_dec_backward_neural(Layer):
    def __init__(self, ks, ns, batch_size):
        super(Shaping_dec_backward_neural, self).__init__()
        self.dense1 = Dense(128, activation='relu')  # First Dense Layer
        self.dense2 = Dense(128, activation='relu')  # Second Dense Layer
        self.dense3 = Dense(ns, activation='linear') # Output Dense Layer
        self.ks = ks
        self.ns = ns
        self.batch_size = batch_size

    def call(self, llr_a_c, llr_a_d):
        # Ensure complex-valued input is split into real and imaginary parts

        llr_a_c_reshaped = tf.reshape(llr_a_c,[self.batch_size * llr_a_c.shape[1], self.ns])
        llr_a_d_reshaped = tf.reshape(llr_a_d, [self.batch_size * llr_a_c.shape[1], self.ks])

        stack_input = tf.concat([llr_a_c_reshaped, llr_a_d_reshaped], axis=1)

        x = self.dense1(stack_input)
        x = self.dense2(x)
        logits = self.dense3(x)

        logits_step1 = tf.reshape(logits,[self.batch_size, llr_a_c.shape[1], self.ns])
        logits = tf.reshape(logits_step1,[self.batch_size, llr_a_c.shape[1]*self.ns])

        return logits

def soft_round(x, alpha, eps=1e-3):
  """Differentiable approximation to `round`.

  Larger alphas correspond to closer approximations of the round function.
  If alpha is close to zero, this function reduces to the identity.

  This is described in Sec. 4.1. in the paper
  > "Universally Quantized Neural Compression"<br />
  > Eirikur Agustsson & Lucas Theis<br />
  > https://arxiv.org/abs/2006.09952

  Args:
    x: `tf.Tensor`. Inputs to the rounding function.
    alpha: Float or `tf.Tensor`. Controls smoothness of the approximation.
    eps: Float. Threshold below which `soft_round` will return identity.

  Returns:
    `tf.Tensor`
  """
  # This guards the gradient of tf.where below against NaNs, while maintaining
  # correctness, as for alpha < eps the result is ignored.
  alpha_bounded = tf.maximum(alpha, eps)

  m = tf.floor(x) + .5
  r = x - m
  z = tf.tanh(alpha_bounded / 2.) * 2.
  y = m + tf.tanh(alpha_bounded * r) / z

  # For very low alphas, soft_round behaves like identity
  return tf.where(alpha < eps, x, y, name="soft_round")

class Shaping_encoder_neural(Layer):

    def __init__(self,
                 ns,
                 ks,
                 batch_size):
        super(Shaping_encoder_neural, self).__init__()
        # super().__init__(precision=precision, **kwargs)

        self._ks = ks
        self._ns = ns
        self.batch_size = batch_size
        self.identity_matrix = np.identity(2**ks)
        self.nn_model = tf.keras.Sequential([
            Input(shape=(2**ks,)),
            Dense(32, activation='relu'),
            Dense(32, activation='relu'),
            Dense(ns, activation=None)  # output is 4 bits
        ])
        self.nn_model.trainable = False 

    def call(self, bits):
        bits = tf.cast(bits, dtype=tf.int32)
        n1 = int(bits.shape[-1] / self._ks)
        new_shape = [n1, self._ks]
        bits = split_dim(bits, new_shape, axis=tf.rank(bits) - 1)
        test_bit_positions = tf.cast(tf.range(self._ks - 1, -1, -1), dtype=tf.int32)
        shifted_bits = tf.bitwise.left_shift(bits, test_bit_positions)
        int_rep = tf.reduce_sum(shifted_bits, axis=-1)
        x = tf.gather(self.identity_matrix, int_rep, axis=0)
        # print(x.shape)

        x_reshaped = tf.reshape(x, [self.batch_size * x.shape[1], x.shape[2]])
        y = self.nn_model(x_reshaped)  # shape: (2*70, 4)
        y = tf.sigmoid(y)
        y = tf.reshape(y, [self.batch_size, x.shape[1], self._ns])  # back to (2, 70, 4)

        x1 = tf.reshape(y, [self.batch_size, x.shape[1] * self._ns])
        # x1_soft_round = soft_round(x1,10.0)
        
        # p0 = tf.reduce_mean(x1)
        
        binary = tf.round(x1)
        # p0 = 1-tf.reduce_mean(tf.cast(x1_soft_round, tf.float32))
        p0 = 1-tf.reduce_mean(tf.cast(binary, tf.float32))
        # p0 = tf.convert_to_tensor([0.75])
        # print('p0',p0.numpy())
        # print('p0_test',p0_test.numpy())
        return x1 + tf.stop_gradient(binary - x1), p0

        # return x1_soft_round, p0

class Pilot_RBF(Layer):
    def __init__(self,  RBF_blocksize, pilot_symbol, K_factor=0.0):
        super(Pilot_RBF, self).__init__()
        self.pilot_symbol = pilot_symbol
        self.RBF_blocksize = RBF_blocksize
        self.K_factor = K_factor  # Adjustable K (0=Rayleigh, inf=AWGN/LoS)

    def call(self, x_mod, batch_size):
        
        x_mod_reshaped = tf.reshape(x_mod, [batch_size, int(x_mod.shape[1]/self.RBF_blocksize), self.RBF_blocksize])
        pilots = tf.fill([batch_size, x_mod_reshaped.shape[1], 1], tf.cast(self.pilot_symbol, x_mod.dtype))
        x_with_pilots = tf.concat([x_mod_reshaped, pilots], axis=-1)

        std = tf.cast(tf.sqrt(0.5), dtype=tf.float32)
        h_real = config.tf_rng.normal(shape=[batch_size,
                                             x_with_pilots.shape[1],
                                             1],
                                      stddev=std,
                                      dtype = tf.float32)
        h_img = config.tf_rng.normal(shape=[batch_size,
                                            x_with_pilots.shape[1],
                                            1],
                                     stddev=std,
                                     dtype = tf.float32)

        h_nlos = tf.complex(h_real, h_img)

        # LoS component (unit phasor, can be generalized)
        h_los = tf.complex(tf.ones_like(h_real), tf.zeros_like(h_real))

        # Combine to form Rician fading
        K = tf.cast(self.K_factor, tf.complex64)
        h = tf.sqrt(K/(K+1.0)) * h_los + tf.sqrt(1.0/(K+1.0)) * h_nlos

        x_mod_h = tf.multiply(x_with_pilots,h)
        return x_mod_h, h

class Ch_est_equalize(Layer):
    def __init__(self,  pilot_rbf, perfect_csi):
        super(Ch_est_equalize, self).__init__()
        self.RBF_blocksize = pilot_rbf.RBF_blocksize
        self.perfect_csi = perfect_csi

    def call(self, y, No, h_exact):

        # Channel estimation step

        if self.perfect_csi:
            h_hat = h_exact
        else:
            h_hat = tf.expand_dims(y[:, :, -1], axis=-1) / (1 + tf.expand_dims(tf.cast(No,tf.complex64), axis=-1))

        # Equalization step
        h_conj = tf.math.conj(h_hat)
        denominator = tf.cast(h_hat * h_conj,tf.complex64)

        y_equalized = y[:, :, :self.RBF_blocksize]*h_conj/denominator
        # y_equalized = tf.squeeze(y_equalized, axis=1)
        y_equalized = tf.reshape(y_equalized, [y_equalized.shape[0],-1])

        return y_equalized, h_hat

class Ch_est_equalize_corrected(Layer):
    def __init__(self,  pilot_rbf, perfect_csi):
        super(Ch_est_equalize_corrected, self).__init__()
        self.RBF_blocksize = pilot_rbf.RBF_blocksize
        self.perfect_csi = perfect_csi

    def call(self, y, No, h_exact):

        # Channel estimation step

        if self.perfect_csi:
            h_hat = h_exact
        else:
            h_hat = tf.expand_dims(y[:, :, -1], axis=-1) / (1 + tf.expand_dims(tf.cast(No,tf.complex64), axis=-1))

        # Equalization step
        No = tf.expand_dims(No, axis=-1)
        h_conj = tf.math.conj(h_hat)
        denominator = (tf.cast(h_hat * h_conj,tf.complex64) + tf.cast(No,tf.complex64))

        rho = tf.abs(h_hat)**2 / (tf.abs(h_hat)**2 + No)
        epsilon2 = rho - tf.square(rho)

        y_equalized = y[:, :, :self.RBF_blocksize]*h_conj/denominator

        rho_tiled = tf.broadcast_to(rho, tf.shape(y_equalized))
        epsilon2_tiled = tf.broadcast_to(epsilon2, tf.shape(y_equalized))

        # print('y_equalized',y_equalized.shape)
        # y_equalized = tf.squeeze(y_equalized, axis=1)
        y_equalized = tf.reshape(y_equalized, [y_equalized.shape[0],-1])
        rho_tiled = tf.reshape(rho_tiled, [rho_tiled.shape[0],-1])
        epsilon2_tiled = tf.reshape(epsilon2_tiled, [epsilon2_tiled.shape[0],-1])
        
        return y_equalized, h_hat, rho_tiled, epsilon2_tiled
