"""
Shared helpers for parallel PAS BER/BLER sweeps (MPI or multiprocessing).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from misc import sim_ber
from pn_multipath_ber_pcs import PAS_scheme


@dataclass
class PasSimConfig:
    """Simulation parameters (defaults match pn_multipath_ber_pcs.py)."""

    results_dir: str = "results_pn_entropy_120GHz_64apsk1"
    snr_list: Tuple[float, ...] = (13,)
    papr_list: Tuple[float, ...] = (6.4, 7.2, 8.0, 0.0)
    ebno_dbs: np.ndarray = field(
        default_factory=lambda: np.arange(5.0, 13.0, 0.2, dtype=np.float64)
    )
    pas_batch_size: int = 1
    num_bp_iter: int = 50
    max_mc_iter: int = 2
    num_target_block_errors: int = 500
    use_awgn: bool = False
    output_folder: str = "BLER_results_pcs"
    cpu_only: bool = True
    p0: float = 0.8544

    num_bits_per_symbol: int = 6
    use_upsampling_filtering_pn: bool = True
    use_neural_demapper: bool = False
    pn_freq: str = '120'
    condition: str = 'lowpn'

    def channel_tag(self) -> str:
        return "awgn_tap" if self.use_awgn else "multipath"


def configure_tensorflow(cpu_only: bool = True) -> None:
    if cpu_only:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            print("Exception occured when setting visible devices")
            pass


def load_constellation(
    results_dir: str, snr_db: float, papr_db: float
) -> Tuple[np.ndarray, np.ndarray]:
    exp_name = f"snr_{snr_db}_papr_{papr_db}"
    npz_path = os.path.join(results_dir, f"{exp_name}.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(npz_path)
    data = np.load(npz_path)
    points = data["constellation_points"]
    probabilities = data["constellation_probabilities"]
    prob_reshape = np.reshape(probabilities, [4, 16])
    prob_basic_set = prob_reshape[0, :] * 4
    return points, prob_basic_set


def build_pas_model(
    config: PasSimConfig,
    points: np.ndarray,
    prob_basic_set: np.ndarray,
) -> PAS_scheme:
    return PAS_scheme(config.pas_batch_size, config.num_bp_iter, points,
                       prob_basic_set, config.p0, config.num_bits_per_symbol,
                       config.use_upsampling_filtering_pn, config.use_neural_demapper, config.pn_freq, config.condition)


def simulate_papr_at_ebnos(
    config: PasSimConfig,
    papr_db: float,
    ebno_dbs: np.ndarray,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run sim_ber for one PAPR constellation over the given Eb/N0 points."""

    snr_db = config.snr_list[0]
    points, prob_basic = load_constellation(config.results_dir, snr_db, papr_db)

    model = build_pas_model(config, points, prob_basic)
    legend = f"PAS_{config.channel_tag()}_papr_{papr_db}"

    ber, bler = sim_ber(
        model,
        ebno_dbs=np.asarray(ebno_dbs, dtype=np.float32),
        batch_size=config.pas_batch_size,
        max_mc_iter=config.max_mc_iter,
        num_target_block_errors=config.num_target_block_errors,
        soft_estimates=False,
        early_stop=True,
        verbose=verbose,
        forward_keyboard_interrupt=True,
    )

    return {
        "papr_db": float(papr_db),
        "ebno_dbs": np.asarray(ebno_dbs, dtype=np.float64),
        "ber": np.asarray(ber, dtype=np.float64),
        "bler": np.asarray(bler, dtype=np.float64),
        "legend": legend,
    }


def run_worker_slice(
    worker_id: int,
    config: PasSimConfig,
    ebno_slice: np.ndarray,
    papr_list: Optional[Sequence[float]] = None,
    verbose: bool = False,
) -> List[dict[str, Any]]:
    """Simulate all PAPR values for a subset of Eb/N0 points."""

    paprs = list(papr_list if papr_list is not None else config.papr_list)
    results = []
    for papr_db in paprs:
        # try:
        chunk = simulate_papr_at_ebnos(config, papr_db, ebno_slice, verbose=verbose)
        results.append(chunk)
        print(
            f"[worker {worker_id}] PAPR={papr_db} dB, "
            f"{len(ebno_slice)} SNR points done",
            flush=True,
        )
        # except FileNotFoundError as exc:
        #     print(f"[worker] Skipping PAPR={papr_db}: {exc}", flush=True)
    return results


def merge_snr_slices(slices: List[dict[str, Any]]) -> dict[str, Any]:
    """Merge partial Eb/N0 results for the same PAPR (sorted by Eb/N0)."""
    if not slices:
        raise ValueError("No slices to merge")
    papr_db = slices[0]["papr_db"]
    ebno = np.concatenate([s["ebno_dbs"] for s in slices])
    ber = np.concatenate([s["ber"] for s in slices])
    bler = np.concatenate([s["bler"] for s in slices])
    order = np.argsort(ebno)
    return {
        "papr_db": papr_db,
        "ebno_dbs": ebno[order],
        "ber": ber[order],
        "bler": bler[order],
        "legend": slices[0]["legend"],
    }


def merge_all_worker_results(
    worker_outputs: List[List[dict[str, Any]]],
) -> List[dict[str, Any]]:
    """Combine lists from MPI ranks or multiprocessing workers."""
    by_papr: dict[float, List[dict[str, Any]]] = {}
    for local_list in worker_outputs:
        if not local_list:
            continue
        for chunk in local_list:
            by_papr.setdefault(chunk["papr_db"], []).append(chunk)

    merged = [merge_snr_slices(chunks) for chunks in by_papr.values()]
    merged.sort(key=lambda x: x["papr_db"])
    return merged


def save_benchmark_npz(
    output_folder: str,
    merged: List[dict[str, Any]],
    config: PasSimConfig,
    filename: str = "pas_benchmark.npz",
) -> str:
    os.makedirs(output_folder, exist_ok=True)
    papr = np.array([m["papr_db"] for m in merged], dtype=np.float64)
    snr_rows = np.stack([m["ebno_dbs"] for m in merged], axis=0)
    ber_rows = np.stack([m["ber"] for m in merged], axis=0)
    bler_rows = np.stack([m["bler"] for m in merged], axis=0)
    path = os.path.join(output_folder, filename)
    np.savez(
        path,
        papr=papr,
        snr=snr_rows,
        ber=ber_rows,
        bler=bler_rows,
        use_awgn=config.use_awgn,
        ebno_dbs_full=np.asarray(config.ebno_dbs, dtype=np.float64),
    )
    return path


def plot_ber_bler(
    output_folder: str,
    merged: List[dict[str, Any]],
    config: PasSimConfig,
) -> None:
    os.makedirs(output_folder, exist_ok=True)
    tag = config.channel_tag()

    plt.figure()
    for m in merged:
        plt.semilogy(
            m["ebno_dbs"],
            m["ber"],
            "o-",
            label=f"PAS, PAPR={m['papr_db']} dB",
        )
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title(f"BER vs SNR ({tag})")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "BER_plot.png"), dpi=300)
    plt.close()

    plt.figure()
    for m in merged:
        plt.semilogy(
            m["ebno_dbs"],
            m["bler"],
            "o-",
            label=f"PAS, PAPR={m['papr_db']} dB",
        )
    plt.xlabel("SNR (dB)")
    plt.ylabel("BLER")
    plt.title(f"BLER vs SNR ({tag})")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "BLER_plot.png"), dpi=300)
    plt.close()


def finalize_and_save(
    worker_outputs: List[List[dict[str, Any]]],
    config: PasSimConfig,
) -> List[dict[str, Any]]:
    merged = merge_all_worker_results(worker_outputs)
    if not merged:
        raise RuntimeError("No simulation results to save (all PAPR files missing?)")
    save_benchmark_npz(config.output_folder, merged, config)
    plot_ber_bler(config.output_folder, merged, config)
    return merged
