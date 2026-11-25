#!/usr/bin/env python3
"""
Single-node parallel PAS BER sweep with multiprocessing.

Use on SLURM jobs with --ntasks=1 --cpus-per-task=N (no mpi4py required).

Example:
    cd PCS
    python pn_multipath_ber_pcs_mp.py --workers 16
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from pas_parallel_utils import (
    PasSimConfig,
    configure_tensorflow,
    finalize_and_save,
    run_worker_slice,
)


def _worker_entry(
    worker_id: int,
    ebno_slice: np.ndarray,
    config: PasSimConfig,
    verbose: bool,
) -> list:

    configure_tensorflow(cpu_only=config.cpu_only)
    current_pid = os.getpid()
    print(
        f"[pool worker {worker_id}] {len(ebno_slice)} Eb/N0 points (PID: {current_pid})",
        flush=True,
    )
    return run_worker_slice(worker_id, config, ebno_slice, verbose=verbose)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multiprocessing parallel PAS BER simulation"
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of processes (default: SLURM_CPUS_PER_TASK or 4)",
    )
    p.add_argument(
        "--results-dir",
        default="results_64_pn_120Hz_lowpn",
    )
    p.add_argument(
        "--output-folder",
        default="ber_results_pcs_parallel",
    )
    p.add_argument(
        "--papr",
        type=float,
        nargs="+",
        default=[6.5, 7.0, 7.5, 0.0],
    )
    p.add_argument("--ebno-min", type=float, default=5.0)
    p.add_argument("--ebno-max", type=float, default=13.0)
    p.add_argument("--ebno-step", type=float, default=0.2)
    p.add_argument("--max-mc-iter", type=int, default=1000)
    p.add_argument("--num-target-block-errors", type=int, default=500)
    p.add_argument("--use-awgn", action="store_true")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    n_workers = args.workers
    if n_workers is None:
        n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))

    config = PasSimConfig(
        results_dir=args.results_dir,
        papr_list=tuple(args.papr),
        ebno_dbs=np.arange(args.ebno_min, args.ebno_max, args.ebno_step),
        max_mc_iter=args.max_mc_iter,
        num_target_block_errors=args.num_target_block_errors,
        use_awgn=args.use_awgn,
        output_folder=args.output_folder,
        cpu_only=not args.gpu,
    )

    os.makedirs(config.output_folder, exist_ok=True)
    ebno_splits = np.array_split(config.ebno_dbs, n_workers)

    print(
        f"Multiprocessing PAS sweep: {n_workers} workers, "
        f"{len(config.ebno_dbs)} Eb/N0 points, "
        f"{len(config.papr_list)} PAPR values, "
        f"channel={config.channel_tag()}",
        flush=True,
    )

    t0 = time.perf_counter()
    all_outputs: list = [[] for _ in range(n_workers)]

    # Only pass ebno slices that are non-empty
    tasks = [
        (wid, ebno_splits[wid])
        for wid in range(n_workers)
        if len(ebno_splits[wid]) > 0
    ]

    with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {
            pool.submit(
                _worker_entry,
                wid,
                slice_arr,
                config,
                args.verbose and wid == 0,
            ): wid
            for wid, slice_arr in tasks
        }
        for fut in as_completed(futures):
            wid = futures[fut]
            all_outputs[wid] = fut.result()

    merged = finalize_and_save(all_outputs, config)
    elapsed = time.perf_counter() - t0
    print(
        f"Done in {elapsed:.1f} s. "
        f"Saved {len(merged)} PAPR curves to {config.output_folder}/",
        flush=True,
    )


if __name__ == "__main__":
    main()
