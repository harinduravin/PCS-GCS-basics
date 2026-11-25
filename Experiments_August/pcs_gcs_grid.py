import importlib
import const_learning_utils

importlib.reload(const_learning_utils)
from const_learning_utils import EndToEndSystem_bitwise
import numpy as np
import tensorflow as tf
import sionna as sn
import gc
import os
import matplotlib.pyplot as plt
import MI_functions

importlib.reload(MI_functions)
from MI_functions import estimate_cm_capacity_pair_pn, binary_switching_algorithm
import itertools


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


## Inner loop
@tf.function
def _inner_loop(lambd_papr, lambd_H, mu, nu, step, num_it, excess_power_target, entropy_target, papr_threshold, No,
                use_papr_constraint):
    # We need to initialize c...
    excess_power = 0.0
    c_excess_power = excess_power
    L = 0.0
    lambda_dist = 10

    k = tf.constant(0, tf.int32)
    while tf.less(k, num_it):
        with tf.GradientTape() as tape:
            loss, loss_ordinary, entropy_value, _, norm_constellation = model(No)

            powers, entropy_value = model.compute_power_to_average_power_samples()
            dist_penalty = model.minimum_distance_penalty_complex(norm_constellation, d_min=0.1)
            excess_H = tf.maximum(tf.abs(entropy_value - entropy_target), lambd_H / nu) * 10

            if use_papr_constraint:
                excess_power = tf.reduce_mean(tf.nn.relu(powers[:, 64:-64] - papr_threshold))
                excess_power = tf.math.log(excess_power + 1e-12) / tf.math.log(10.)
                c_excess_power = excess_power - excess_power_target
                c_excess_power = tf.maximum(c_excess_power, lambd_papr / mu)

            if use_papr_constraint:
                # Augmented Lagrangian
                L = loss - lambd_papr * c_excess_power + 0.5 * mu * (
                    tf.square(c_excess_power)) + lambda_dist * dist_penalty - lambd_H * excess_H + 0.5 * nu * (
                        tf.square(excess_H))
            else:
                L = loss + lambda_dist * dist_penalty - lambd_H * excess_H + 0.5 * nu * (tf.square(excess_H))

        # Computing and applying gradients
        weights = model.trainable_weights
        grads = tape.gradient(L, weights)
        grads, glob_norm = tf.clip_by_global_norm(grads, 0.1)
        # glob_norm = 0.0
        optimizer.apply_gradients(zip(grads, weights))
        # Print progress
        if k % 50 == 0:
            tf.print('Iteration', k, 'L', L / np.log(2), 'loss', loss / np.log(2), 'loss_ordinary',
                     loss_ordinary / np.log(2), 'entropy_value', entropy_value / np.log(2), 'glob_norm', glob_norm)

        k = k + 1
        step = step + 1

    if use_papr_constraint:
        powers, entropy_value = model.compute_power_to_average_power_samples()
        excess_power = tf.reduce_mean(tf.nn.relu(powers[:, 64:-64] - papr_threshold))
        excess_power = tf.math.log(excess_power + 1e-12) / tf.math.log(10.)
        c_excess_power = excess_power - excess_power_target
        c_excess_power = tf.maximum(c_excess_power, lambd_papr / mu)

    return c_excess_power, entropy_value, step


def run_experiment_grid(batch_size, bits_per_symbol, snr_db, papr_threshold_db, use_papr_constraint, entropy_target,
                        outer_iterations, initial_inner_iter, inner_iter, ebno_db_minval, ebno_db_maxval, RESULTS_DIR):
    print(f"\n===== Running SNR={snr_db} dB | PAPR={papr_threshold_db} dB =====\n")

    ebno_db = tf.random.uniform(shape=[batch_size], minval=ebno_db_minval, maxval=ebno_db_maxval)  # 8.0 and 16.0
    No_ = sn.phy.utils.ebnodb2no(ebno_db, num_bits_per_symbol=bits_per_symbol, coderate=1)
    No_ = tf.expand_dims(No_, axis=-1)

    # No_ = tf.convert_to_tensor(1/(10**(snr_db/10)),tf.float32)
    papr_threshold_ = tf.math.pow(10.0, papr_threshold_db / 10.0)
    excess_power_target_ = -6.0  # B
    entropy_target_ = entropy_target * 0.693

    lambd_papr = tf.Variable(tf.constant(0.0, tf.float32), trainable=False, dtype=tf.float32)
    lambd_H = tf.Variable(tf.constant(0.0, tf.float32), trainable=False, dtype=tf.float32)
    mu = tf.Variable(0.01, trainable=False, dtype=tf.float32)
    nu = tf.Variable(0.01, trainable=False, dtype=tf.float32)
    step = tf.constant(0, tf.int32)

    for i in range(outer_iterations):  # 10
        if i == 0:
            num_it = initial_inner_iter  # 1000
        else:
            num_it = inner_iter  # 35
        c_excess_power, entropy_value, step = _inner_loop(lambd_papr, lambd_H, mu, nu, step,
                                                          num_it, excess_power_target_,
                                                          entropy_target_, papr_threshold_, No_, use_papr_constraint)

        excess_H = tf.maximum(tf.abs(entropy_value - entropy_target_), lambd_H / nu) * 10

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

    tf.print(f"Finished {exp_name}")


snr_list = [13]
papr_list = [6.4, 7.2, 8.0, 8.8]

thz_frequency = '220'
condition = 'lowpn'

m = 6
M = 2 ** m
N = M  # Number of classes (symbols)
bits_per_symbol = int(np.log2(N))
shaping_bit_id = 5
use_upsampling_filtering = True  # Remove 'false' this after testing without upsampling and filtering

########
symbol_points_qam64 = np.array([-1.08012344973464 + 1.08012344973464j
                                   , -1.08012344973464 + 0.771516749810460j
                                   , -1.08012344973464 + 0.154303349962092j
                                   , -1.08012344973464 + 0.462910049886276j
                                   , -0.771516749810460 + 1.08012344973464j
                                   , -0.771516749810460 + 0.771516749810460j
                                   , -0.771516749810460 + 0.154303349962092j
                                   , -0.771516749810460 + 0.462910049886276j
                                   , -0.154303349962092 + 1.08012344973464j
                                   , -0.154303349962092 + 0.771516749810460j
                                   , -0.154303349962092 + 0.154303349962092j
                                   , -0.154303349962092 + 0.462910049886276j
                                   , -0.462910049886276 + 1.08012344973464j
                                   , -0.462910049886276 + 0.771516749810460j
                                   , -0.462910049886276 + 0.154303349962092j
                                   , -0.462910049886276 + 0.462910049886276j
                                   , -1.08012344973464 - 1.08012344973464j
                                   , -1.08012344973464 - 0.771516749810460j
                                   , -1.08012344973464 - 0.154303349962092j
                                   , -1.08012344973464 - 0.462910049886276j
                                   , -0.771516749810460 - 1.08012344973464j
                                   , -0.771516749810460 - 0.771516749810460j
                                   , -0.771516749810460 - 0.154303349962092j
                                   , -0.771516749810460 - 0.462910049886276j
                                   , -0.154303349962092 - 1.08012344973464j
                                   , -0.154303349962092 - 0.771516749810460j
                                   , -0.154303349962092 - 0.154303349962092j
                                   , -0.154303349962092 - 0.462910049886276j
                                   , -0.462910049886276 - 1.08012344973464j
                                   , -0.462910049886276 - 0.771516749810460j
                                   , -0.462910049886276 - 0.154303349962092j
                                   , -0.462910049886276 - 0.462910049886276j
                                   , 1.08012344973464 + 1.08012344973464j
                                   , 1.08012344973464 + 0.771516749810460j
                                   , 1.08012344973464 + 0.154303349962092j
                                   , 1.08012344973464 + 0.462910049886276j
                                   , 0.771516749810460 + 1.08012344973464j
                                   , 0.771516749810460 + 0.771516749810460j
                                   , 0.771516749810460 + 0.154303349962092j
                                   , 0.771516749810460 + 0.462910049886276j
                                   , 0.154303349962092 + 1.08012344973464j
                                   , 0.154303349962092 + 0.771516749810460j
                                   , 0.154303349962092 + 0.154303349962092j
                                   , 0.154303349962092 + 0.462910049886276j
                                   , 0.462910049886276 + 1.08012344973464j
                                   , 0.462910049886276 + 0.771516749810460j
                                   , 0.462910049886276 + 0.154303349962092j
                                   , 0.462910049886276 + 0.462910049886276j
                                   , 1.08012344973464 - 1.08012344973464j
                                   , 1.08012344973464 - 0.771516749810460j
                                   , 1.08012344973464 - 0.154303349962092j
                                   , 1.08012344973464 - 0.462910049886276j
                                   , 0.771516749810460 - 1.08012344973464j
                                   , 0.771516749810460 - 0.771516749810460j
                                   , 0.771516749810460 - 0.154303349962092j
                                   , 0.771516749810460 - 0.462910049886276j
                                   , 0.154303349962092 - 1.08012344973464j
                                   , 0.154303349962092 - 0.771516749810460j
                                   , 0.154303349962092 - 0.154303349962092j
                                   , 0.154303349962092 - 0.462910049886276j
                                   , 0.462910049886276 - 1.08012344973464j
                                   , 0.462910049886276 - 0.771516749810460j
                                   , 0.462910049886276 - 0.154303349962092j
                                   , 0.462910049886276 - 0.462910049886276j])

# constellation = symbol_points_qam64
########
r1 = 1.0
r2 = 1.5

angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
radii = np.where(np.arange(N) % 2 == 0, r1, r2)

constellation = radii * np.exp(1j * angles)
constellation = constellation / np.sqrt(np.mean(np.abs(constellation) ** 2))
labels = np.array([f"{i:0{m}b}" for i in range(2 ** m)])
########

tau = 10
batch_size = 10
papr_batch_size = 30
block_size = 4096 - 32 * 4

# No = tf.convert_to_tensor(1/(10**(snr_db/10)),tf.float32)
ebno_db = tf.random.uniform(shape=[batch_size], minval=8.0, maxval=9.0)
No_ = sn.phy.utils.ebnodb2no(ebno_db, num_bits_per_symbol=bits_per_symbol, coderate=1)
No_ = tf.expand_dims(No_, axis=-1)

symbolwise_demapping = False
neural_bitwise_demapping = False
model = EndToEndSystem_bitwise(N, constellation, papr_batch_size,
                               batch_size, block_size,
                               bits_per_symbol, shaping_bit_id,
                               use_upsampling_filtering, thz_frequency, symbolwise_demapping, neural_bitwise_demapping,
                               condition, tau)
_, _, _, _, _ = model(No_)
initial_weights = model.get_weights()

# demodulator_weights = np.load("demodulator_weights.npz")

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

RESULTS_DIR = "results_64_pn_220Hz_lowpn"
os.makedirs(RESULTS_DIR, exist_ok=True)
use_papr_constraint = False
outer_iterations = 2000
inner_initial_iterations = 5000
inner_iterations = 35
entropy_target = 5.6
ebno_db_minval = 6.0
ebno_db_maxval = 16.0

for snr_db in snr_list:
    for papr_db in papr_list:
        print('-------------------------------------------------------')
        print('PAPR :', papr_db, ' dB')
        print('-------------------------------------------------------')
        model.set_weights(initial_weights)
        optimizer = tf.keras.optimizers.Adam(0.001)
        run_experiment_grid(batch_size, bits_per_symbol, snr_db, papr_db, use_papr_constraint, entropy_target,
                            outer_iterations, inner_initial_iterations, inner_iterations, ebno_db_minval,
                            ebno_db_maxval, RESULTS_DIR)
        tf.keras.backend.clear_session()
        gc.collect()

        tf.print("Training constellation finished")

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

print("CCDF Saved")

# # ------------------
# # Save constellation
# # ------------------
# with tf.GradientTape() as tape:
#     _, _, _, shaping_probs,_ = model(No_)

# points = (
#     model.modulator.constellation_real.numpy()
#     + 1j * model.modulator.constellation_imag.numpy()
# )
# probs = shaping_probs.numpy()

# entropy_finalized = entropy(probs)

# normalized_points = normalize_constellation(points, probs)

# exp_name = "64point_const"
# np.save(f"{RESULTS_DIR}/{exp_name}_points.npy", normalized_points)
# np.save(f"{RESULTS_DIR}/{exp_name}_probs.npy", probs)

# plt.figure()
# plot_constellation_wlabels(
#     normalized_points,
#     probs,
#     "PAPR=  dB"
# )
# plt.savefig(f"{RESULTS_DIR}/{exp_name}_constellation_entropy_{entropy_finalized}.png")
# plt.close()

# tf.print("Constellation saved")


# symbols_np = normalized_points
# px = np.squeeze(probs)
# px = px.astype(np.float64)

# # Use GPU if available
# tf.print("GPUs:", tf.config.list_physical_devices('GPU'))

# # =========================
# # Parameters
# # =========================
# # EsNoVec = np.arange(-6, 35, 1, dtype=np.float32)   # Es/No in dB
# EsNoVec = np.arange(-6, 35, 5, dtype=np.float32)   # Es/No in dB
# N_bicm = 10_000_0

# batch_size_mi = 500000  # tune based on GPU memory
# EbNoVec = EsNoVec - 10 * np.log10(m)
# symbols_np = symbols_np/np.sqrt(np.mean(np.abs(symbols_np)**2, axis=0, keepdims=True))

# if thz_frequency == '220':
#     pn_var = 0.0108  # 0.0032 for 120GHz and 0.0108 for 220GHz
# elif thz_frequency == '120':
#     pn_var = 0.0032
# #############

# def normalize_symbols_for_px(symbols_np, px):
#     avg_e = np.sum(px * (np.abs(symbols_np) ** 2))
#     return symbols_np / np.sqrt(avg_e)

# px /= px.sum()

# symbols_np = normalize_symbols_for_px(symbols_np, px)
# symbols = tf.constant(symbols_np, dtype=tf.complex64)

# all_pairs = list(itertools.combinations(range(M), 2))

# results_exact = {}

# for idx_pair in all_pairs:
#     idx_pair_tf = tf.constant(list(idx_pair), dtype=tf.int32)
#     ccCap_pair, shannon = estimate_cm_capacity_pair_pn(
#         px, EsNoVec, N_bicm, int(batch_size_mi/10), idx_pair_tf, symbols, pn_var,'awgn'
#     )
#     results_exact[idx_pair] = {
#         "ccCap": ccCap_pair,
#         "shannon": shannon
#     }
#     tf.print(f"Done pair {idx_pair}")


# # Prepare arrays with unique names
# save_dict = {}
# for idx_pair, data in results_exact.items():
#     i, j = idx_pair
#     save_dict[f"ccCap_{i}_{j}"] = data["ccCap"]   # convert TF tensors to NumPy
#     save_dict[f"shannon_{i}_{j}"] = data["shannon"]

# # Save compressed
# np.savez_compressed(f"{RESULTS_DIR}/{exp_name}_results_exact_{entropy_finalized}.npz", **save_dict)
# tf.print("Pairwise mutual information saved")


# labels_improved, I_sum_current, I_sum_initial = binary_switching_algorithm(labels, px, results_exact, EsNoVec, np.array([0,2,3]), max_iter=100)

# binary = np.array(labels_improved)
# decimal = np.array([int(bits, 2) for bits in binary])
# symbols_np_adjusted = symbols_np[np.argsort(decimal)]

# plt.figure()
# plot_constellation_wlabels(
#     symbols_np_adjusted,
#     probs,
#     "PAPR=  dB"
# )
# plt.savefig(f"{RESULTS_DIR}/adjusted_{exp_name}_constellation_entropy_{entropy_finalized}.png")
# plt.close()
# np.save(f"{RESULTS_DIR}/adjusted_{exp_name}_points.npy", symbols_np_adjusted)
# tf.print("Bit labeling optimization done")

