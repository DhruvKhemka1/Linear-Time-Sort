"""
Merge sort: the classic O(n log n) sort.

  1. Split the list in half.
  2. Sort each half (by calling merge sort again).
  3. Merge the two sorted halves into one sorted list.
"""


def merge(left, right):
    """Combine two sorted lists into one sorted list."""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])   # whatever is left over is already sorted
    result.extend(right[j:])
    return result


def merge_sort(values):
    """Return a new sorted list."""
    values = list(values)
    if len(values) <= 1:
        return values
    mid = len(values) // 2
    return merge(merge_sort(values[:mid]), merge_sort(values[mid:]))
