import tensorflow as tf
import numpy as np
from collections import defaultdict

@tf.function
def bits_to_symbol_indices(bits, lut_tf, m):
    weights = tf.cast(2 ** tf.range(m-1, -1, -1), tf.int32)
    vals = tf.reduce_sum(bits * weights, axis=1)   # [B]
    return tf.gather(lut_tf, vals)                 # [B]

@tf.function
def add_awgn(signal, ebno_db, bits_per_symbol):
    ebno_lin = tf.pow(10.0, ebno_db / 10.0)
    esno_lin = ebno_lin * tf.cast(bits_per_symbol, tf.float32)
    sigma = tf.sqrt(1.0 / (2*esno_lin))
    noise_r = tf.random.normal(tf.shape(signal), dtype=tf.float32)
    noise_i = tf.random.normal(tf.shape(signal), dtype=tf.float32)
    noise = tf.complex(noise_r, noise_i) * tf.cast(sigma, tf.complex64)
    return signal + noise

@tf.function
def cm_batch_contribution(y, x_sent, noise_var, m, symbols):
    # y: [B], x_sent: [B], symbols: [M]
    # Compute sum_{x'} exp(-|y-x'|^2/(2 noise_var)) for all x' in constellation
    y_exp = tf.expand_dims(y, axis=1)                     # [B,1]
    sym_exp = tf.expand_dims(symbols, axis=0)            # [1,M]
    d2_all = tf.abs(y_exp - sym_exp) ** 2                # [B,M]
    exponents_all = -tf.cast(d2_all, tf.float32) / (2.0 * noise_var)
    sumprob = tf.reduce_sum(tf.exp(exponents_all), axis=1)  # [B]

    d2_sent = tf.abs(y - x_sent) ** 2
    eucl_term = -tf.cast(d2_sent, tf.float32) / (2.0 * noise_var)

    # MATLAB expression:
    # m - log2(prob(...)/exp(eucl_dist_norm(...)))
    # Here prob(...,ii=0,...) == sum over all symbols
    vals = tf.cast(m, tf.float32) - tf.math.log(sumprob / tf.exp(eucl_term)) / tf.math.log(2.0)
    return tf.reduce_sum(vals)

def estimate_cm_capacity(esno_db_vec, n_samples, batch_size, m, lut_tf, symbols):
    ccCap = np.zeros(len(esno_db_vec), dtype=np.float32)
    shannoncap = np.zeros(len(esno_db_vec), dtype=np.float32)

    num_batches = int(np.ceil(n_samples / batch_size))

    for n, esno_db in enumerate(esno_db_vec):
        ebno_db = esno_db - 10 * np.log10(m)
        esno_lin = 10 ** (esno_db / 10.0)
        noise_var = np.float32(1.0 / (2*esno_lin))
        shannoncap[n] = 0.5*np.log2(1.0 + 2*esno_lin)

        total = 0.0
        done = 0

        for _ in range(num_batches):
            B = min(batch_size, n_samples - done)
            if B <= 0:
                break

            bits = tf.random.uniform((B, m), minval=0, maxval=2, dtype=tf.int32)
            idx = bits_to_symbol_indices(bits, lut_tf, m)
            x_sent = tf.gather(symbols, idx)                            # [B]
            y = add_awgn(x_sent, tf.constant(ebno_db, tf.float32), m)   # [B]

            batch_sum = cm_batch_contribution(
                y,
                x_sent,
                tf.constant(noise_var, dtype=tf.float32),
                m,
                symbols
            )
            total += float(batch_sum.numpy())
            done += B

        ccCap[n] = total / n_samples
        print(f"Es/No = {esno_db:.1f} dB, CM capacity = {ccCap[n]:.6f}")

    return ccCap, shannoncap

def gray_bitcount(x):
    return bin(x).count("1")

def bold_d_and_w_ordered(constellation):
    x = np.asarray(constellation)
    M = len(x)
    m = int(np.log2(M))
    if 2**m != M:
        raise ValueError("Constellation size must be a power of 2.")

    dist_to_total = defaultdict(int)
    dist_to_h1 = defaultdict(int)

    for i in range(M):
        for j in range(i + 1, M):
            if np.iscomplexobj(x):
                dist = np.abs(x[i] - x[j])
            else:
                dist = np.linalg.norm(x[i] - x[j])

            dist = float(np.round(dist, 12))
            dist_to_total[dist] += 1

            if gray_bitcount(i ^ j) == 1:
                dist_to_h1[dist] += 1

    d = np.array(sorted(dist_to_total.keys()))
    w = np.array([dist_to_h1[di] for di in d], dtype=int)
    pair_counts = np.array([dist_to_total[di] for di in d], dtype=int)

    return d, w, pair_counts


@tf.function
def bits_to_symbol_indices_binary(bits, idx_pair):
    # bits shape: [B, 1], 0 or 1
    # idx_pair: [2] containing the two symbol indices to use
    idx = tf.gather(idx_pair, bits)  # [B]
    return idx


@tf.function
def cm_batch_contribution_pair(y, x_sent, noise_var, idx_pair, symbols):
    # y: [B], x_sent: [B], only two symbols in constellation
    y_exp = y          # [B,1]
    # print('y_exp.shape :',y_exp[:5,0])
    sym_pair = tf.gather(symbols, idx_pair)    # [2]
    sym_exp = tf.expand_dims(sym_pair, axis=0) # [1,2]
    d2_all = tf.abs(y_exp - sym_exp) ** 2      # [B,2]

    # print("first 5 d2_all =", d2_all.shape)
    exponents_all = -tf.cast(d2_all, tf.float32) / (2.0 * noise_var)
    sumprob = tf.expand_dims(tf.reduce_sum(tf.exp(exponents_all), axis=1),axis=1)  # [B]

    d2_sent = tf.abs(y - x_sent) ** 2
    eucl_term = -tf.cast(d2_sent, tf.float32) / (2.0 * noise_var)

    # print("first 5 sumprob =", sumprob.shape)
    # print("first 5 eucl_term =", eucl_term.shape)

    m = 1
    vals = tf.cast(m, tf.float32) - tf.math.log(sumprob / tf.exp(eucl_term)) / tf.math.log(2.0)
    # print('symbols :', symbols)

    return tf.reduce_sum(vals)


def estimate_cm_capacity_pair(esno_db_vec, n_samples, batch_size, idx_pair, symbols):
    """
    idx_pair: [2] tensor/array containing the two symbol indices to transmit
    """
    ccCap = np.zeros(len(esno_db_vec), dtype=np.float32)
    shannoncap = np.zeros(len(esno_db_vec), dtype=np.float32)
    num_batches = int(np.ceil(n_samples / batch_size))
    
    idx_pair_tf = tf.constant(idx_pair, dtype=tf.int32)

    for n, esno_db in enumerate(esno_db_vec):
        ebno_db = esno_db - 10 * np.log10(1)  # m=1 for binary
        esno_lin = 10 ** (esno_db / 10.0)
        noise_var = np.float32(1.0 / (2*esno_lin))
        shannoncap[n] = 0.5*np.log2(1.0 + 2*esno_lin)  # Shannon for binary

        total = 0.0
        done = 0

        for _ in range(num_batches):
            B = min(batch_size, n_samples - done)
            if B <= 0:
                break

            # Binary bits: only 1 bit per symbol
            bits = tf.random.uniform((B, 1), minval=0, maxval=2, dtype=tf.int32)
            idx = bits_to_symbol_indices_binary(bits, idx_pair_tf)
            # print('idx.shape :',idx.shape)
            # print('idx :',idx[:20,0])
            x_sent = tf.gather(symbols, idx)                            # [B]
            y = add_awgn(x_sent, tf.constant(ebno_db, tf.float32), 1)   # [B], bits_per_symbol=1
            # print('y shape :', y.shape)

            batch_sum = cm_batch_contribution_pair(
                y,
                x_sent,
                tf.constant(noise_var, dtype=tf.float32),
                idx_pair_tf,
                symbols
            )
            total += float(batch_sum.numpy())
            done += B

        ccCap[n] = total / n_samples
        print(f"Es/No = {esno_db:.1f} dB, Pair MI = {ccCap[n]:.6f}")

    return ccCap, shannoncap

@tf.function
def add_awgn_get_sigma(signal, ebno_db, bits_per_symbol):
    ebno_lin = tf.pow(10.0, ebno_db / 10.0)
    esno_lin = ebno_lin * tf.cast(bits_per_symbol, tf.float32)
    sigma2 = 1.0 / (2*esno_lin)
    sigma = tf.sqrt(sigma2)
    noise = sigma * tf.random.normal(tf.shape(signal), dtype=tf.float32)
    return signal + noise, sigma2

@tf.function
def bicm_batch_contribution(y, tx_bits, noise_var, symbols, m, masks):
    # y: [B]
    # tx_bits: [B, m]
    # noise_var: scalar

    y_exp = y[:, None]                      # [B,1]
    s_exp = symbols[None, :]                # [1,M]

    # Real 1-D distance
    d2 = tf.square(y_exp - s_exp)           # [B,M]

    # metric proportional to log p(y|x), constants omitted because they cancel
    metric = -d2 / (2.0 * noise_var)        # [B,M]

    # full constellation log-sum-exp, shape [B]
    lse_all = tf.math.reduce_logsumexp(metric, axis=1)

    out = tf.TensorArray(dtype=tf.float32, size=m)
    for i in tf.range(m):
        b_i = tx_bits[:, i]  # [B], 0/1

        # select proper mask for each sample according to transmitted bit b_i
        # masks[i] has shape [2,M], gather -> [B,M]
        mask_i = tf.gather(masks[i], b_i)   # [B,M]

        neg_inf = tf.constant(-1e30, dtype=tf.float32)
        metric_masked = tf.where(mask_i > 0, metric, neg_inf)

        # subset log-sum-exp, shape [B]
        lse_subset = tf.math.reduce_logsumexp(metric_masked, axis=1)

        # MATLAB: llccap += log2( prob_all / prob_subset )
        # BICM_i = 1 - E[ log2(prob_all/prob_subset) ]
        term = (lse_all - lse_subset) /tf.math.log(tf.constant(2.0, tf.float32))
        out = out.write(i, 1.0 - tf.reduce_mean(term))
        # out.append(1.0 - tf.reduce_mean(term))

    return out.stack() # tf.stack(out, axis=0)  # [m]

def estimate_bicm_capacity(esno_db_vec, n_samples, batch_size, symbols, lut_tf, m, masks):
    bicmCap = np.zeros((m, len(esno_db_vec)), dtype=np.float32)
    num_batches = int(np.ceil(n_samples / batch_size))

    for n, esno_db in enumerate(esno_db_vec):
        ebno_db = esno_db - 10*np.log10(m)

        acc = np.zeros(m, dtype=np.float64)
        done = 0

        for _ in range(num_batches):
            B = min(batch_size, n_samples - done)
            if B <= 0:
                break

            bits = tf.random.uniform((B, m), minval=0, maxval=2, dtype=tf.int32)
            idx = bits_to_symbol_indices(bits, lut_tf, m)
            x = tf.gather(symbols, idx)  # [B]

            y, noise_var = add_awgn_get_sigma(x, tf.constant(ebno_db, tf.float32), m)

            batch_bicm = bicm_batch_contribution(y, bits, noise_var, symbols, m, masks)  # [m]
            acc += batch_bicm.numpy() * B
            done += B

        bicmCap[:, n] = acc / n_samples
        print(f"Es/No={esno_db:.1f} dB, BICM per bit = {bicmCap[:, n]}")

    return bicmCap