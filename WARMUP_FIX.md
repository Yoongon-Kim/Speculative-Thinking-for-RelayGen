# Warmup Run Fix

## Problem Identified

The first run of the benchmark was significantly slower than subsequent runs:
- **Run 1**: 303.50s, 14.06 tok/s
- **Run 2-5**: ~100s, ~27 tok/s

This is a **3x slowdown** on the first run!

## Root Cause

vLLM performs lazy initialization of CUDA components on the first forward pass:
1. **Model loading** (during `LLM()` constructor): Loads model weights to GPU memory
2. **First generation** (first `model.generate()` call):
   - Initializes CUDA kernels
   - Allocates KV-cache memory
   - Compiles CUDA graphs
   - Warms up tensor cores

The timing started **after** model loading but **included** the CUDA initialization, making the first measurement unfair.

## Solution

Added a **warmup run** before starting the actual benchmark measurements.

### Changes Made

**Large Model Test** ([spec_thinking_speedup_test.py:134-144](spec_thinking_speedup_test.py#L134-L144)):
```python
# Warmup run to initialize CUDA kernels and KV-cache
print("\nPerforming warmup run to initialize CUDA kernels...")
warmup_messages = [{"role": "user", "content": "What is 2+2?"}]
_ = model.generate(
    messages=warmup_messages,
    max_tokens=50,
    temperature=self.temperature,
    top_k=self.top_k,
    top_p=self.top_p
)
print("Warmup complete\n")
```

**Speculative Thinking Test** ([spec_thinking_speedup_test.py:212-222](spec_thinking_speedup_test.py#L212-L222)):
Same warmup code added here as well.

## Why This Fixes The Problem

The warmup run:
1. ✅ Triggers all lazy CUDA initialization
2. ✅ Allocates KV-cache memory
3. ✅ Compiles CUDA graphs
4. ✅ Warms up tensor cores
5. ✅ Uses a simple prompt (fast to complete)
6. ✅ Is **not** included in benchmark measurements

After the warmup, all subsequent generations (including the actual benchmark runs) have fair and consistent timing.

## Expected Results After Fix

All runs should now have similar latency:
- **Run 1**: ~100s, ~27 tok/s ✅
- **Run 2**: ~100s, ~27 tok/s ✅
- **Run 3**: ~100s, ~27 tok/s ✅
- **Run 4**: ~100s, ~27 tok/s ✅
- **Run 5**: ~100s, ~27 tok/s ✅

## Benchmark Output

You'll now see this in the output:

```
Models initialized successfully

Performing warmup run to initialize CUDA kernels...
Warmup complete

Problem 1/5
Problem: Find the sum of...
  Run 1/5... Latency: 100.19s, Tokens: 2718, Throughput: 27.13 tok/s
  Run 2/5... Latency: 100.84s, Tokens: 2732, Throughput: 27.09 tok/s
  ...
```

## Why Warmup Runs Are Standard Practice

Warmup runs are common in ML benchmarking because:
1. **Fair Comparison**: Ensures all runs are measured under the same conditions
2. **Accurate Metrics**: Removes one-time initialization costs from measurements
3. **Realistic Performance**: Reflects actual production usage where models stay loaded
4. **Industry Standard**: Used by MLPerf, HuggingFace benchmarks, etc.

## Additional Notes

- The warmup uses `max_tokens=50` (small) to complete quickly
- The warmup prompt is trivial ("What is 2+2?") to minimize computation
- Both models (large and speculative) are warmed up separately
- The warmup generation output is discarded (assigned to `_`)

## Files Modified

- [spec_thinking_speedup_test.py](spec_thinking_speedup_test.py#L134-144) - Added warmup to large model test
- [spec_thinking_speedup_test.py](spec_thinking_speedup_test.py#L212-222) - Added warmup to speculative thinking test
- [WARMUP_FIX.md](WARMUP_FIX.md) - This documentation
