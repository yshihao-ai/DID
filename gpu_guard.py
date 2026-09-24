import argparse
import os
import signal
import subprocess
import time


MATCH_KEYWORD = "xxx"


def get_gpu_uuid(gpu_index: int) -> str:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )

    for line in output.strip().splitlines():
        index, uuid = [x.strip() for x in line.split(",", 1)]
        if int(index) == gpu_index:
            return uuid

    raise RuntimeError(f"GPU {gpu_index} not found")


def get_gpu_processes(gpu_uuid: str) -> set[int]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return set()

    pids = set()

    for line in output.strip().splitlines():
        if not line.strip():
            continue

        parts = [x.strip() for x in line.split(",", 1)]
        if len(parts) != 2:
            continue

        pid_str, process_gpu_uuid = parts

        if process_gpu_uuid == gpu_uuid:
            try:
                pids.add(int(pid_str))
            except ValueError:
                pass

    return pids


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def get_process_cwd(pid: int) -> str:
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except Exception:
        return ""


def get_process_exe(pid: int) -> str:
    try:
        return os.readlink(f"/proc/{pid}/exe")
    except Exception:
        return ""


def get_process_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            data = f.read()

        return data.replace(b"\x00", b" ").decode(
            errors="replace"
        ).strip()
    except Exception:
        return ""


def get_process_info(pid: int) -> dict:
    return {
        "pid": pid,
        "cwd": get_process_cwd(pid),
        "exe": get_process_exe(pid),
        "cmdline": get_process_cmdline(pid),
    }


def process_matches_keyword(pid: int, keyword: str) -> bool:
    info = get_process_info(pid)

    keyword = keyword.lower()

    searchable = " ".join(
        [
            info["cwd"],
            info["exe"],
            info["cmdline"],
        ]
    ).lower()

    return keyword in searchable


def print_process_info(pid: int):
    info = get_process_info(pid)

    print(f"[PROCESS] PID: {pid}")
    print(f"          CWD: {info['cwd']}")
    print(f"          EXE: {info['exe']}")
    print(f"          CMD: {info['cmdline']}")


def get_all_matching_processes(keyword: str) -> set[int]:
    matching_pids = set()
    self_pid = os.getpid()

    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue

        pid = int(entry)

        if pid == self_pid:
            continue

        if not process_exists(pid):
            continue

        if process_matches_keyword(pid, keyword):
            matching_pids.add(pid)

    return matching_pids


def terminate_processes(pids: set[int], timeout: float = 5.0):
    if not pids:
        print("[INFO] No matching processes found.")
        return

    print(f"[ACTION] Sending SIGTERM to {len(pids)} processes.")

    for pid in sorted(pids):
        try:
            print_process_info(pid)
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"[WARNING] Permission denied for PID {pid}")

    deadline = time.time() + timeout

    while time.time() < deadline:
        remaining = {
            pid
            for pid in pids
            if process_exists(pid)
        }

        if not remaining:
            print("[OK] All matching processes exited.")
            return

        time.sleep(0.2)

    remaining = {
        pid
        for pid in pids
        if process_exists(pid)
    }

    if remaining:
        print(
            f"[ACTION] Sending SIGKILL to remaining processes: "
            f"{sorted(remaining)}"
        )

    for pid in sorted(remaining):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"[WARNING] Permission denied for PID {pid}")

    print("[OK] Termination completed.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Monitor a GPU. If a new GPU process appears and its "
            "process path does not contain the configured keyword, "
            "terminate all processes associated with that keyword."
        )
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index to monitor. Default: 0.",
    )

    parser.add_argument(
        "--keyword",
        type=str,
        default=MATCH_KEYWORD,
        help=f"Path keyword to protect. Default: {MATCH_KEYWORD}.",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds. Default: 1.0.",
    )

    args = parser.parse_args()

    gpu_index = args.gpu
    keyword = args.keyword
    interval = args.interval

    gpu_uuid = get_gpu_uuid(gpu_index)

    baseline = get_gpu_processes(gpu_uuid)

    print(f"[INFO] Monitoring GPU {gpu_index}.")
    print(f"[INFO] GPU UUID: {gpu_uuid}")
    print(f"[INFO] Protected keyword: {keyword}")
    print(f"[INFO] Existing GPU PIDs: {sorted(baseline)}")

    for pid in sorted(baseline):
        print_process_info(pid)

    print("[INFO] Waiting for new GPU processes...")

    known_pids = set(baseline)

    while True:
        current_pids = get_gpu_processes(gpu_uuid)

        new_pids = current_pids - known_pids

        for pid in sorted(new_pids):
            print()
            print(f"[INFO] New GPU process detected: PID {pid}")
            print_process_info(pid)

            if process_matches_keyword(pid, keyword):
                print(
                    f"[INFO] PID {pid} contains '{keyword}'. "
                    "Ignoring it."
                )
            else:
                print(
                    f"[WARNING] PID {pid} does not contain "
                    f"'{keyword}'."
                )

                print(
                    f"[WARNING] Terminating all processes "
                    f"containing '{keyword}'."
                )

                matching_pids = get_all_matching_processes(keyword)

                print(
                    f"[INFO] Matching PIDs: "
                    f"{sorted(matching_pids)}"
                )

                terminate_processes(matching_pids)
                return

        known_pids.update(current_pids)

        time.sleep(interval)


if __name__ == "__main__":
    main()
