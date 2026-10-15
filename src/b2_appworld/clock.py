"""Elapsed time that AppWorld's task-clock freezing does not intercept."""
import time


def elapsed_time() -> float:
    # AppWorld runs on Linux/macOS. freezegun patches perf_counter/monotonic but
    # leaves the OS clock_gettime call intact; never alter global freeze rules.
    return time.clock_gettime(time.CLOCK_MONOTONIC)
