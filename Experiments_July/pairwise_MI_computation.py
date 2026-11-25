
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import itertools
import importlib
import Mi_functions
importlib.reload(Mi_functions)
from Mi_functions import bits_to_symbol_indices, gray_bitcount,\
    bold_d_and_w_ordered, bits_to_symbol_indices_binary, cm_batch_contribution_pair, estimate_cm_capacity_pair,\
    add_awgn_get_sigma, bicm_batch_contribution

#add_awgn, cm_batch_contribution, estimate_cm_capacity,

LOG2 = tf.math.log(tf.constant(2.0, tf.float32))
PI = tf.constant(np.pi, tf.float32)


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
def add_phase_noise_and_awgn(signal, ebno_db, bits_per_symbol, phase_noise_var):
    ebno_db = tf.cast(ebno_db, tf.float32)
    bits_per_symbol = tf.cast(bits_per_symbol, tf.float32)
    phase_noise_var = tf.cast(phase_noise_var, tf.float32)

    # Phase noise: phi ~ N(0, phase_noise_var)
    phi = tf.random.normal(tf.shape(signal), stddev=tf.sqrt(phase_noise_var), dtype=tf.float32)
    phase_rot = tf.complex(tf.cos(phi), tf.sin(phi))
    signal_pn = signal * phase_rot

    # AWGN from Eb/N0
    ebno_lin = tf.pow(10.0, ebno_db / 10.0)
    esno_lin = ebno_lin * bits_per_symbol
    sigma = tf.sqrt(1.0 / (2.0 * esno_lin))

    noise_r = tf.random.normal(tf.shape(signal), dtype=tf.float32)
    noise_i = tf.random.normal(tf.shape(signal), dtype=tf.float32)
    noise = tf.complex(noise_r, noise_i) * tf.cast(sigma, tf.complex64)

    return signal_pn + noise


@tf.function
def sample_pair_indices(bits, idx_pair, q_pair):
    # bits: [B,1] in {0,1}
    # q_pair: [2] effective probabilities summing to 1
    u = tf.random.uniform(tf.shape(bits), dtype=tf.float64)
    bits_eff = tf.cast(u < q_pair[1], tf.int32)  # 1 with prob q1, else 0
    idx = tf.gather(idx_pair, bits_eff)
    return idx, bits_eff


@tf.function
def bits_to_symbol_indices_binary(bits, idx_pair):
    # bits shape: [B, 1], 0 or 1
    # idx_pair: [2] containing the two symbol indices to use
    idx = tf.gather(idx_pair, bits)  # [B]
    return idx

@tf.function
def wrap_to_pi(angle):
    return tf.math.floormod(angle + PI, 2.0 * PI) - PI

@tf.function
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

@tf.function
def cm_batch_contribution_pair_polar(y, x_sent, sigma_r, sigma_p, idx_pair, symbols):
    sym_pair = tf.gather(symbols, idx_pair)  # [2]
    sym_exp = tf.expand_dims(sym_pair, axis=0)  # [1,2]
    y_exp = y

    logp_all = log_polar_approx_pdf(y_exp, sym_exp, sigma_r, sigma_p)  # [B,2]
    log_sumprob = tf.expand_dims(tf.reduce_logsumexp(logp_all, axis=1), axis=1)  # [B]
    logp_sent = log_polar_approx_pdf(y, x_sent, sigma_r, sigma_p)  # [B]

    vals = tf.cast(1, tf.float32) - (log_sumprob - logp_sent) / LOG2
    return tf.reduce_sum(vals)

@tf.function
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


@tf.function
def cm_batch_contribution_pair_low_pn(y, x_sent, sigma_r, sigma_p, idx_pair, symbols):
    sym_pair = tf.gather(symbols, idx_pair)  # [2]
    sym_exp = tf.expand_dims(sym_pair, axis=0)  # [1,2]
    y_exp = y

    logp_all = low_pn_approx_pdf(y_exp, sym_exp, sigma_r, sigma_p)  # [B,2]
    log_sumprob = tf.expand_dims(tf.reduce_logsumexp(logp_all, axis=1), axis=1)  # [B]
    logp_sent = low_pn_approx_pdf(y, x_sent, sigma_r, sigma_p)  # [B]

    vals = tf.cast(1, tf.float32) - (log_sumprob - logp_sent) / LOG2
    return tf.reduce_sum(vals)

@tf.function
def log_p_y_given_x_mc(y, x, noise_var, phase_noise_var, n_phi_mc):
    y = tf.cast(y, tf.complex64)                         # [B] or [B,M]
    x = tf.cast(x, tf.complex64)                         # same shape as y
    noise_var = tf.cast(2*noise_var, tf.float32)
    phase_noise_var = tf.cast(phase_noise_var, tf.float32)

    shape_y = tf.shape(y)
    phi_shape = tf.concat([[n_phi_mc], shape_y], axis=0)   # [K,...]

    phi = tf.random.normal(phi_shape,
                           stddev=tf.sqrt(phase_noise_var),
                           dtype=tf.float32)               # [K,...]
    rot = tf.complex(tf.cos(phi), tf.sin(phi))             # [K,...]

    x_exp = tf.expand_dims(x, axis=0)                      # [1,...]
    y_exp = tf.expand_dims(y, axis=0)                      # [1,...]

    mu = x_exp * rot                                       # [K,...]
    d2 = tf.abs(y_exp - mu) ** 2                           # [K,...]
    d2 = tf.cast(d2, tf.float32)

    # CN(mu, noise_var): p(y|x,phi)=1/(pi*noise_var) exp(-|y-mu|^2/noise_var)
    log_pdf_given_phi = -tf.math.log(np.pi * noise_var) - d2 / noise_var   # [K,...]

    # log E_phi[p(y|x,phi)] = logsumexp(log p(y|x,phi_k)) - log(K)
    return tf.reduce_logsumexp(log_pdf_given_phi, axis=0) - tf.math.log(tf.cast(n_phi_mc, tf.float32))

@tf.function
def cm_batch_contribution_pair_exact(y, x_sent, noise_var, phase_noise_var, idx_pair, symbols, q_pair):
    x_sent = tf.squeeze(x_sent)
    sym_pair = tf.gather(symbols, idx_pair)  # [2]
    # print('sym_pair.shape:', sym_pair.shape)
    sym_exp = tf.expand_dims(sym_pair, axis=0)  # [1,2]
    # print('sym_exp.shape:', sym_exp.shape)
    # print('x_sent.shape:', x_sent.shape)
    y_exp = y

    sym_exp = tf.broadcast_to(
        tf.expand_dims(sym_pair, 0),
        [tf.shape(y_exp)[0], tf.shape(sym_pair)[0]]
    )  # [2,M]
    # print('sym_exp.shape:', sym_exp.shape)
    y_grid = tf.broadcast_to(y_exp, tf.shape(sym_exp))  # [B,2]
    # print('y_grid.shape:', y_grid.shape)
    logp_all = log_p_y_given_x_mc(
        y_grid,
        sym_exp,
        noise_var,
        phase_noise_var,
        1024
    )  # [B,M]

    log_num = log_p_y_given_x_mc(
        tf.squeeze(y),
        x_sent,
        noise_var,
        phase_noise_var,
        1024
    )  # [B]

    log_q = tf.math.log(tf.cast(q_pair, tf.float32))  # [2]
    logp_all = logp_all + log_q[tf.newaxis, :]  # [B,2]

    log_den = tf.reduce_logsumexp(logp_all, axis=1)  # [B]
    vals = (log_num - log_den) / tf.math.log(2.0)  # [B]
    return tf.reduce_sum(vals)


@tf.function
def cm_batch_contribution_pair(y, x_sent, noise_var, idx_pair, symbols, q_pair):
    y_exp = y  # [B,1]
    sym_pair = tf.gather(symbols, idx_pair)  # [2]
    sym_exp = tf.expand_dims(sym_pair, axis=0)  # [1,2]
    d2_all = tf.abs(y_exp - sym_exp) ** 2  # [B,2]

    exponents_all = -tf.cast(d2_all, tf.float32) / (2.0 * noise_var)
    log_q = tf.math.log(tf.cast(q_pair, tf.float32))  # [2]
    log_terms = exponents_all + log_q[tf.newaxis, :]  # [B,2]

    log_den = tf.reduce_logsumexp(log_terms, axis=1)  # [B]
    d2_sent = tf.squeeze(tf.abs(y - x_sent)) ** 2
    log_num = -tf.cast(d2_sent, tf.float32) / (2.0 * noise_var)  # [B]
    vals = (log_num - log_den) / tf.math.log(2.0)  # [B]

    return tf.reduce_sum(vals)


def estimate_cm_capacity_pair_pn(esno_db_vec, n_samples, batch_size, idx_pair, symbols, pn_var, condition):
    """
    idx_pair: [2] tensor/array containing the two symbol indices to transmit
    """
    ccCap = np.zeros(len(esno_db_vec), dtype=np.float32)
    shannoncap = np.zeros(len(esno_db_vec), dtype=np.float32)
    num_batches = int(np.ceil(n_samples / batch_size))

    idx_pair_tf = tf.constant(idx_pair, dtype=tf.int32)

    p_pair = tf.gather(px, idx_pair)
    q_pair_tf = p_pair / tf.reduce_sum(p_pair)

    for n, esno_db in enumerate(esno_db_vec):
        ebno_db = esno_db - 10 * np.log10(1)  # m=1 for binary
        esno_lin = 10 ** (esno_db / 10.0)
        noise_var = np.float32(1.0 / (2 * esno_lin))
        shannoncap[n] = 0.5 * np.log2(1.0 + 2 * esno_lin)  # Shannon for binary

        total = 0.0
        done = 0

        for _ in range(num_batches):
            B = min(batch_size, n_samples - done)
            if B <= 0:
                break

            bits = tf.random.uniform((B, 1), minval=0, maxval=2, dtype=tf.int32)
            idx, bits_eff = sample_pair_indices(bits, idx_pair_tf, q_pair_tf)
            x_sent = tf.gather(symbols, idx)

            if condition == 'awgn':
                y = add_awgn(x_sent, tf.constant(ebno_db, tf.float32), 1)  # [B]
            else:
                y = add_phase_noise_and_awgn(x_sent, tf.constant(ebno_db, tf.float32), 1, pn_var)  # [B]

            if condition == 'awgn':
                batch_sum = cm_batch_contribution_pair(
                    y,
                    x_sent,
                    tf.constant(noise_var, dtype=tf.float32),
                    idx_pair_tf,
                    symbols,
                    q_pair_tf
                )
            elif condition == 'exact':
                batch_sum = cm_batch_contribution_pair_exact(
                    y,
                    x_sent,
                    tf.constant(noise_var, dtype=tf.float32),
                    tf.constant(pn_var, dtype=tf.float32),
                    idx_pair_tf,
                    symbols,
                    q_pair_tf
                )
            elif condition == 'lowpn':
                batch_sum = cm_batch_contribution_pair_low_pn(
                    y,
                    x_sent,
                    tf.constant(tf.sqrt(noise_var), dtype=tf.float32),
                    tf.constant(tf.sqrt(pn_var), dtype=tf.float32),
                    idx_pair_tf,
                    symbols
                )
            elif condition == 'highsnr':
                batch_sum = cm_batch_contribution_pair_polar(
                    y,
                    x_sent,
                    tf.constant(tf.sqrt(noise_var), dtype=tf.float32),
                    tf.constant(tf.sqrt(pn_var), dtype=tf.float32),
                    idx_pair_tf,
                    symbols
                )

            total += float(batch_sum.numpy())
            done += B

        ccCap[n] = total / n_samples
        print(f"Es/No = {esno_db:.1f} dB, Pair MI = {ccCap[n]:.6f}")

    return ccCap, shannoncap

#############

data = np.load("snr_13_papr_7.2.npz")
symbols_np = data["constellation_points"]
px = np.squeeze(data["constellation_probabilities"])
px = px.astype(np.float64)

# Use GPU if available
print("GPUs:", tf.config.list_physical_devices('GPU'))

# =========================
# Parameters
# =========================
# EsNoVec = np.arange(-6, 35, 1, dtype=np.float32)   # Es/No in dB
EsNoVec = np.arange(-6, 35, 40, dtype=np.float32)   # Es/No in dB
N_bicm = 10_000_0
m = 6
M = 2 ** m
batch_size = 500000  # tune based on GPU memory
EbNoVec = EsNoVec - 10 * np.log10(m)
symbols_np = symbols_np/np.sqrt(np.mean(np.abs(symbols_np)**2, axis=0, keepdims=True))

pn_var = 0.0032
#############

def normalize_symbols_for_px(symbols_np, px):
    avg_e = np.sum(px * (np.abs(symbols_np) ** 2))
    return symbols_np / np.sqrt(avg_e)

px /= px.sum()

symbols_np = normalize_symbols_for_px(symbols_np, px)
symbols = tf.constant(symbols_np, dtype=tf.complex64)

all_pairs = list(itertools.combinations(range(M), 2))

results_exact = {}

for idx_pair in all_pairs:
    idx_pair_tf = tf.constant(list(idx_pair), dtype=tf.int32)
    ccCap_pair, shannon = estimate_cm_capacity_pair_pn(
        EsNoVec, N_bicm, int(batch_size/10), idx_pair_tf, symbols, pn_var,'exact'
    )
    results_exact[idx_pair] = {
        "ccCap": ccCap_pair,
        "shannon": shannon
    }
    print(f"Done pair {idx_pair}")


# Prepare arrays with unique names
save_dict = {}
for idx_pair, data in results_exact.items():
    i, j = idx_pair
    save_dict[f"ccCap_{i}_{j}"] = data["ccCap"]   # convert TF tensors to NumPy
    save_dict[f"shannon_{i}_{j}"] = data["shannon"]

# Save compressed
np.savez_compressed("results_exact.npz", **save_dict)

# data = np.load("results_exact.npz")
#
# # Reconstruct results_exact if needed
# results_exact_loaded = {}
# for i, j in all_pairs:
#     results_exact_loaded[(i, j)] = {
#         "ccCap": data[f"ccCap_{i}_{j}"],
#         "shannon": data[f"shannon_{i}_{j}"],
#     }
#
# # Or access directly for computations
# ccCap_0_1 = data["ccCap_0_1"]
# shannon_0_1 = data["shannon_0_1"]