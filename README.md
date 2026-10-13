# Linear-Time Sort

A sorting algorithm that runs in **expected O(n) time** by predicting where
each value belongs, using the shape of the data's distribution.

Ordinary comparison sorts (merge sort, quicksort) can't beat O(n log n).
This one doesn't compare elements to find their place. It computes the place
directly from an estimate of the data's CDF (cumulative distribution function).

## How it works

If F is the CDF of the data, then F(x) is roughly the fraction of values smaller
than x, so x belongs near position n · F(x) in the sorted list.

1. **Sample** 1,000 random values, plus the true minimum and maximum.
2. **Fit a CDF.** Try a uniform, normal, and exponential distribution, and keep
   the best one that passes a Kolmogorov–Smirnov test. If none fits, build an
   **empirical CDF** from the sample by connecting the sample points with
   straight lines. This works for any shape.
3. **Bucket.** Put each value x into bucket ⌊n · F(x)⌋. There are n buckets,
   so about one value per bucket.
4. **Finish.** Insertion-sort each small bucket. Any bucket with more than 64
   values is merge-sorted instead (the *safety net*), which guarantees
   O(n log n) in the worst case.

## Why it's linear

If the model matches the data, F(x) is uniformly spread over [0, 1]
(the probability integral transform), so every bucket gets about one value
and the total work is O(n). [`linearity_proof.pdf`](linearity_proof.pdf)
proves this formally:

| Situation | Result |
|---|---|
| Any input | Always sorted correctly |
| Any input | O(n log n) worst case |
| Random continuous data | Expected (34 + 5χ²)·n + O(√n) steps, where χ² measures how far the model is from the true distribution |
| A value that repeats a fixed fraction of the time | Ω(n log n) for any CDF model, since equal values always share a bucket |

## Results

Speedup over a plain Python merge sort (both in pure Python), measured with
`test_sorts.py` on an Apple Silicon Mac with Python 3.9:

| Data | 10K values | 100K values | 1M values |
|---|---|---|---|
| Uniform | 2.2× | 3.6× | 3.6× |
| Normal | 2.0× | 3.2× | 3.6× |
| Lognormal (empirical CDF) | 1.6× | 2.5× | 2.8× |
| Integers 0–1000 (many repeats) | 2.4× | 1.7× | 2.0× |

On real data (405K macOS system files):

| Data | Speedup |
|---|---|
| Inode IDs (two clusters ~10¹⁸ apart) | 2.7× |
| File sizes (heavy tail, many repeats) | 1.9× |
| Modification times (73% identical) | 0.9× (slower) |

The real data never matched a textbook distribution, so the empirical CDF did
all the work there. The weak spot is repeated values: the timestamp result is
exactly the Ω(n log n) case from the proof.

Python's built-in `sorted()` is still faster than both, because it is written
in C. The comparison above is Python against Python.

## Files

| File | What it is |
|---|---|
| `distribution_sort.py` | The algorithm |
| `merge_sort.py` | Plain merge sort, used as the safety net and as the baseline |
| `test_sorts.py` | Correctness tests and benchmarks |
| `linearity_proof.pdf` / `.tex` | The formal runtime proof |

## Running it

Requires Python 3.8 or newer, with no other packages.

```bash
python3 test_sorts.py
```

This runs 16 correctness tests against Python's `sorted()`, then the
benchmarks (about a minute). The real-data benchmark reads file sizes and
timestamps from macOS system folders (`/System/Library`, `/usr`, `/Library`)
and is skipped on other systems.

To use the sort directly:

```python
from distribution_sort import distribution_sort

print(distribution_sort([5.2, 1.7, 9.3, 3.1]))   # [1.7, 3.1, 5.2, 9.3]
```

## Next steps

- Handle repeated values: skip buckets whose values are all equal, and count
  repeated values instead of sorting every copy.
- Two-pass bucketing to reduce cache misses on large inputs.
