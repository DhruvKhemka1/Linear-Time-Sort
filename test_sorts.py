"""
Test cases for distribution_sort and merge_sort.

Run it with:   python3 test_sorts.py

  Part 1: correctness - does each sort give the same answer as Python's sorted()?
  Part 2: runtime on generated data (random values from known distributions)
  Part 3: runtime on real data (facts about the macOS system files on this computer)
"""

import gc
import os
import random
import time

from distribution_sort import distribution_sort, choose_cdf, bucket_index, BIG_BUCKET
from merge_sort import merge_sort


# ---------- Part 1: correctness ----------

def correctness_cases():
    return {
        "empty list":             [],
        "one value":              [5],
        "two values":             [2, 1],
        "all the same":           [7] * 1000,
        "already sorted":         list(range(1000)),
        "reverse sorted":         list(range(1000, 0, -1)),
        "negative numbers":       [random.randint(-1000, 1000) for _ in range(1000)],
        "mixed ints and floats":  [random.choice([random.randint(0, 100), random.random() * 100])
                                   for _ in range(1000)],
        "lots of duplicates":     [random.randint(0, 10) for _ in range(5000)],
        "huge outliers":          [random.random() for _ in range(5000)] + [1e300, -1e300],
        "two far-apart clusters": [random.gauss(0, 1) for _ in range(2000)]
                                  + [random.gauss(1_000_000, 1) for _ in range(2000)],
        "uniform":                [random.uniform(0, 100) for _ in range(5000)],
        "normal":                 [random.gauss(50, 10) for _ in range(5000)],
        "exponential":            [random.expovariate(0.5) for _ in range(5000)],
        "lognormal":              [random.lognormvariate(0, 1) for _ in range(5000)],
    }


def random_lists(how_many):
    """Lists of random length, mixing different kinds of values."""
    lists = []
    for _ in range(how_many):
        length = random.randint(0, 2000)
        lists.append([random.choice([random.gauss(0, 1),
                                     random.randint(-5, 5),
                                     random.lognormvariate(0, 2)])
                      for _ in range(length)])
    return lists


def run_correctness_tests():
    print("PART 1: correctness\n")
    failures = 0

    cases = correctness_cases()
    cases["200 random lists"] = random_lists(200)

    for name, data in cases.items():
        lists = data if name == "200 random lists" else [data]
        passed = all(distribution_sort(values) == sorted(values) and
                     merge_sort(values) == sorted(values)
                     for values in lists)
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        if not passed:
            failures += 1

    print(f"\n  {len(cases) - failures} of {len(cases)} tests passed\n")
    return failures


# ---------- Helpers for timing ----------

def seconds_to_sort(sort_function, data, repeats):
    """Fastest of a few runs. The garbage collector is paused while timing
    (like Python's timeit does), because it slows down code that makes lots of lists."""
    best = float("inf")
    for _ in range(repeats):
        gc.disable()
        start = time.perf_counter()
        sort_function(data)
        best = min(best, time.perf_counter() - start)
        gc.enable()
    return best


def safety_net_share(data, cdf):
    """How much of the data lands in buckets too big for insertion sort."""
    n = len(data)
    counts = [0] * n
    for x in data:
        counts[bucket_index(cdf, x, n)] += 1
    return sum(c for c in counts if c > BIG_BUCKET) / n


def print_header():
    print(f"  {'data':12} {'n':>10}  {'CDF used':11} {'safety net':>10}  "
          f"{'distribution':>12} {'merge sort':>11}  {'speedup':>7}")
    print("  " + "-" * 82)


def print_timing_row(label, data):
    n = len(data)
    repeats = 3 if n <= 100_000 else 1
    cdf_name, cdf = choose_cdf(data)
    share = safety_net_share(data, cdf)
    d = seconds_to_sort(distribution_sort, data, repeats) * 1e9 / n
    m = seconds_to_sort(merge_sort, data, repeats) * 1e9 / n
    print(f"  {label:12} {n:>10,}  {cdf_name:11} {share:>10.0%}  "
          f"{d:>9.0f} ns {m:>8.0f} ns  {m / d:>6.1f}x")


# ---------- Part 2: runtime on generated data ----------

def run_generated_data_timing():
    print("PART 2: runtime on generated data (time per element, lower is better)\n")
    print_header()
    distributions = {
        "uniform":    lambda: random.uniform(0, 100),
        "normal":     lambda: random.gauss(50, 10),
        "lognormal":  lambda: random.lognormvariate(0, 1),
        "duplicates": lambda: random.randint(0, 1000),
    }
    for label, make_value in distributions.items():
        for n in [10_000, 100_000, 1_000_000]:
            print_timing_row(label, [make_value() for _ in range(n)])
        print()


# ---------- Part 3: runtime on real data ----------

def collect_system_file_data():
    """Sizes, modification times and inode IDs of macOS system files.
    Only system folders are read, nothing from the home folder."""
    sizes, mod_times, inodes = [], [], []
    for folder in ["/System/Library", "/usr", "/Library"]:
        for dirpath, _, files in os.walk(folder):
            for name in files:
                try:
                    info = os.lstat(os.path.join(dirpath, name))
                except OSError:
                    continue  # no permission, or the file vanished
                sizes.append(info.st_size)
                mod_times.append(info.st_mtime)
                inodes.append(info.st_ino)
    return sizes, mod_times, inodes


def run_real_data_timing():
    print("PART 3: runtime on real data (macOS system files, in the order found)\n")
    print("  Reading system file info (takes a few seconds)...")
    sizes, mod_times, inodes = collect_system_file_data()
    if len(sizes) < 1000:
        print("  Not enough files found - skipping (this part expects macOS).\n")
        return
    print()

    print_header()
    for label, data in [("file sizes", sizes), ("mod. times", mod_times), ("inode IDs", inodes)]:
        if distribution_sort(data) != sorted(data) or merge_sort(data) != sorted(data):
            print(f"  FAIL  {label} was not sorted correctly")
            continue
        print_timing_row(label, data)
    print()


if __name__ == "__main__":
    random.seed(42)
    failures = run_correctness_tests()
    if failures:
        print("Fix the failing tests before timing anything.")
    else:
        run_generated_data_timing()
        run_real_data_timing()
