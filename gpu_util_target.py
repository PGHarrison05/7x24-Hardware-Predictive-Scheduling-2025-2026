#!/usr/bin/env python3
import argparse
import multiprocessing as mp
import time
import torch


def gpu_worker(gpu_id: int, target_util: float, duration: int, matrix_size: int, cycle_time: float):
    """
    Approximate GPU utilization by busy/sleep duty cycling.
    target_util: 0.0 to 1.0
    """
    if not torch.cuda.is_available():
        print(f"[GPU {gpu_id}] CUDA not available.")
        return

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    # Pre-allocate matrices on the target GPU
    a = torch.randn(matrix_size, matrix_size, device=device)
    b = torch.randn(matrix_size, matrix_size, device=device)

    busy_time = cycle_time * target_util
    sleep_time = max(0.0, cycle_time - busy_time)

    start = time.time()
    cycles = 0

    print(
        f"[GPU {gpu_id}] Starting: target={target_util*100:.1f}% "
        f"duration={duration}s matrix={matrix_size} cycle={cycle_time}s"
    )

    while time.time() - start < duration:
        cycle_start = time.time()

        # Busy phase
        while time.time() - cycle_start < busy_time:
            c = torch.matmul(a, b)
            c = torch.relu(c)
            # force actual completion on GPU
            torch.cuda.synchronize(gpu_id)

        # Sleep phase
        if sleep_time > 0:
            time.sleep(sleep_time)

        cycles += 1

    print(f"[GPU {gpu_id}] Finished after {cycles} cycles.")


def main():
    parser = argparse.ArgumentParser(description="Approximate target utilization on multiple NVIDIA GPUs")
    parser.add_argument("--gpu-count", type=int, default=3, help="Number of GPUs to use")
    parser.add_argument("--util", type=float, default=30.0, help="Target utilization percent, e.g. 30")
    parser.add_argument("--duration", type=int, default=120, help="Duration in seconds")
    parser.add_argument("--matrix-size", type=int, default=4096, help="Matrix size for GPU work")
    parser.add_argument("--cycle-time", type=float, default=1.0, help="Duty cycle period in seconds")
    parser.add_argument("--start-gpu", type=int, default=0, help="Starting GPU index")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Check your NVIDIA driver, CUDA, and PyTorch install.")

    available = torch.cuda.device_count()
    requested_end = args.start_gpu + args.gpu_count

    if requested_end > available:
        raise ValueError(
            f"Requested GPUs {args.start_gpu}..{requested_end-1}, "
            f"but only {available} CUDA GPU(s) are available."
        )

    target_util = args.util / 100.0
    if not (0.0 < target_util <= 1.0):
        raise ValueError("--util must be between 0 and 100")

    processes = []
    for gpu_id in range(args.start_gpu, requested_end):
        p = mp.Process(
            target=gpu_worker,
            args=(gpu_id, target_util, args.duration, args.matrix_size, args.cycle_time)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
