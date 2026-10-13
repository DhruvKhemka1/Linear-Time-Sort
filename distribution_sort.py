"""
Distribution sort: a sort that runs in expected linear time.

Key idea: if F is the CDF of the data, then F(x) is spread evenly
between 0 and 1. So F(x) tells us roughly where x belongs in the
sorted output. We use it to drop every value into a bucket, then
tidy up each (small) bucket.

Steps:
  1. Take a small random sample of the data.
  2. Try to fit a known distribution (uniform, normal, exponential).
     If none fits well, build our own CDF from the sample.
  3. Put each value into bucket number  F(x) * number_of_buckets.
  4. Sort each bucket with insertion sort and join them together.
"""

import bisect
import math
import random
from statistics import NormalDist, mean, stdev

from merge_sort import merge_sort

SAMPLE_SIZE = 1000   # how many values we look at to learn the shape
BIG_BUCKET = 64      # buckets bigger than this use merge sort instead


# ---------- Step 2a: known distributions ----------

def fit_uniform(sample):
    lo, hi = min(sample), max(sample)
    if lo == hi:
        return None

    def cdf(x):
        return (x - lo) / (hi - lo)
    return cdf


def fit_normal(sample):
    try:
        sigma = stdev(sample)
    except OverflowError:  # values too huge to square
        return None
    if sigma == 0:
        return None
    return NormalDist(mean(sample), sigma).cdf


def fit_exponential(sample):
    lo = min(sample)
    average_gap = mean(sample) - lo
    if average_gap == 0:
        return None

    def cdf(x):
        if x <= lo:
            return 0.0
        return 1 - math.exp(-(x - lo) / average_gap)
    return cdf


# ---------- Step 2b: our own CDF from the sample ----------

def build_empirical_cdf(sample):
    """Connect the dots between the sorted sample points."""
    points = sorted(sample)
    last = len(points) - 1

    def cdf(x):
        i = bisect.bisect_right(points, x)  # how many sample points are <= x
        if i == 0:
            return 0.0
        if i > last:
            return 1.0
        left, right = points[i - 1], points[i]
        fraction = (x - left) / (right - left)  # how far x is between them
        return (i - 1 + fraction) / last
    return cdf


# ---------- Step 2c: pick the best one ----------

def ks_distance(points, cdf):
    """Kolmogorov-Smirnov distance: the biggest gap between the sample's
    real CDF and the model's CDF. Smaller means a better fit."""
    m = len(points)
    worst = 0.0
    for i, x in enumerate(points):
        p = cdf(x)
        worst = max(worst, abs(p - i / m), abs(p - (i + 1) / m))
    return worst


def choose_cdf(values):
    """Return (name, cdf) for the CDF that best describes the values."""
    sample = random.sample(values, min(len(values), SAMPLE_SIZE))
    # Add the true smallest and largest values, so values outside the
    # sample's range don't all get squashed into the first or last bucket.
    points = sorted(sample + [min(values), max(values)])

    # A gap smaller than this means "the data fits" (5% significance level).
    threshold = 1.36 / math.sqrt(len(points))

    best_name, best_cdf, best_distance = None, None, threshold
    for name, fit in [("uniform", fit_uniform),
                      ("normal", fit_normal),
                      ("exponential", fit_exponential)]:
        cdf = fit(points)
        if cdf is None:
            continue
        distance = ks_distance(points, cdf)
        if distance < best_distance:
            best_name, best_cdf, best_distance = name, cdf, distance

    if best_cdf is None:
        return "empirical", build_empirical_cdf(points)
    return best_name, best_cdf


# ---------- Steps 3 and 4: bucket and sort ----------

def insertion_sort(items):
    """Sort a small list in place."""
    for i in range(1, len(items)):
        value = items[i]
        j = i - 1
        while j >= 0 and items[j] > value:
            items[j + 1] = items[j]
            j -= 1
        items[j + 1] = value


def bucket_index(cdf, x, num_buckets):
    """Which bucket x goes into."""
    index = int(cdf(x) * num_buckets)
    return min(max(index, 0), num_buckets - 1)  # keep it in range


def distribution_sort(values):
    """Return a new sorted list."""
    values = list(values)
    n = len(values)
    if n <= BIG_BUCKET:
        insertion_sort(values)
        return values

    _, cdf = choose_cdf(values)

    # Step 3: drop each value into its bucket (about 1 value per bucket).
    buckets = [[] for _ in range(n)]
    for x in values:
        buckets[bucket_index(cdf, x, n)].append(x)

    # Step 4: sort each bucket and join them.
    result = []
    for bucket in buckets:
        if len(bucket) > BIG_BUCKET:
            bucket = merge_sort(bucket)  # safety net for when the CDF guessed badly
        else:
            insertion_sort(bucket)
        result.extend(bucket)
    return result
