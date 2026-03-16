import time
import numpy as np

# Simulate 400k candles
n = 400000
p_entry = np.random.rand(n)
tau = 0.999 # rare entry

# Method A: Python loop
def python_loop(start_i):
    i = start_i
    while i < n:
        pe = p_entry[i]
        if pe >= tau:
            return i
        i += 1
    return None

# Method B: Numpy vectorized
def numpy_argmax(start_i):
    arr = p_entry[start_i:]
    mask = arr >= tau
    if not np.any(mask):
        return None
    return start_i + np.argmax(mask)

# Benchmark
t0 = time.time()
for _ in range(100):
    python_loop(0)
t1 = time.time()
print(f"Python loop: {t1-t0:.4f}s")

t0 = time.time()
for _ in range(100):
    numpy_argmax(0)
t1 = time.time()
print(f"NumPy argmax: {t1-t0:.4f}s")
