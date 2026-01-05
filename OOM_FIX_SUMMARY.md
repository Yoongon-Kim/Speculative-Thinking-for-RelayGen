# OOM Error Fix and Test Mode Feature

## Problem
When running both the large model test and speculative thinking test sequentially, the benchmark would encounter Out of Memory (OOM) errors because:
1. Both model configurations need to be loaded
2. Ray workers from the first test weren't fully cleaned up before the second test
3. GPU memory wasn't being released between tests

## Solution

### 1. Added `--test-mode` Parameter

**New command-line argument:**
```bash
--test-mode {large,spec,both}
```

**Options:**
- `large`: Run only the large model (Qwen3-32B) test
- `spec`: Run only the speculative thinking (Qwen3-1.7B + Qwen3-32B) test
- `both`: Run both tests sequentially (default)

**Code locations:**
- [spec_thinking_speedup_test.py:44](spec_thinking_speedup_test.py#L44) - Constructor parameter
- [spec_thinking_speedup_test.py:554-560](spec_thinking_speedup_test.py#L554-L560) - Argument parser
- [spec_thinking_speedup_test.py:461-518](spec_thinking_speedup_test.py#L461-L518) - Test mode logic in `run()` method

### 2. Aggressive GPU Cleanup Between Tests

When running `--test-mode both`, the script now performs aggressive cleanup between tests:

**Cleanup steps** ([spec_thinking_speedup_test.py:422-435](spec_thinking_speedup_test.py#L422-L435)):
1. Shutdown Ray workers
2. Run Python garbage collection (`gc.collect()`)
3. Clear CUDA cache (`torch.cuda.empty_cache()`)
4. Synchronize CUDA operations (`torch.cuda.synchronize()`)
5. Wait 5 seconds for resources to be fully released

### 3. Single Model Result Handling

Added methods to handle results when running only one test:

**New methods:**
- `print_single_model_stats()` ([spec_thinking_speedup_test.py:401-427](spec_thinking_speedup_test.py#L401-L427)) - Print stats for one model
- `save_single_model_results()` ([spec_thinking_speedup_test.py:429-459](spec_thinking_speedup_test.py#L429-L459)) - Save results to JSON

**Output files:**
- `benchmark_results/large_model_benchmark_TIMESTAMP.json` (for `--test-mode large`)
- `benchmark_results/speculative_thinking_benchmark_TIMESTAMP.json` (for `--test-mode spec`)
- `benchmark_results/speedup_benchmark_TIMESTAMP.json` (for `--test-mode both`)

## Usage Examples

### Recommended: Run Tests Separately

This is the **best approach to avoid OOM errors**:

```bash
# Step 1: Run large model test
python spec_thinking_speedup_test.py --test-mode large

# Step 2: Run speculative thinking test (in a separate process)
python spec_thinking_speedup_test.py --test-mode spec
```

**Benefits:**
- Completely separate processes
- No shared GPU memory
- No Ray worker conflicts
- Each test gets full GPU resources

### Run Both Tests (with automatic cleanup)

```bash
python spec_thinking_speedup_test.py --test-mode both
```

**Note:** This now includes aggressive cleanup between tests, but may still encounter OOM on systems with limited GPU memory.

### Other Combinations

```bash
# Only test large model with custom parameters
python spec_thinking_speedup_test.py \
    --test-mode large \
    --num-problems 10 \
    --max-tokens 16384

# Only test speculative thinking
python spec_thinking_speedup_test.py \
    --test-mode spec \
    --temperature 0.7 \
    --top-k 30
```

## Output Comparison

### `--test-mode large`
```
================================================================================
LARGE MODEL ONLY RESULTS
================================================================================

Large Model (Qwen3-32B) Statistics:
--------------------------------------------------------------------------------

1. LATENCY (seconds)
  Mean:        45.32s
  Median:      44.87s
  ...

Results saved to: benchmark_results/large_model_benchmark_20260105_143022.json
```

### `--test-mode spec`
```
================================================================================
SPECULATIVE THINKING RESULTS
================================================================================

Speculative Thinking (Qwen3-1.7B + Qwen3-32B) Statistics:
--------------------------------------------------------------------------------

1. LATENCY (seconds)
  Mean:        28.15s
  Median:      27.92s
  ...

4. LARGE MODEL USAGE RATIO
  Mean ratio:      45.2%
  Median ratio:    44.8%

Results saved to: benchmark_results/speculative_thinking_benchmark_20260105_143022.json
```

### `--test-mode both`
```
================================================================================
TESTING LARGE MODEL ONLY (Qwen3-32B)
================================================================================
...

================================================================================
CLEANING UP RESOURCES BEFORE SPECULATIVE THINKING TEST
================================================================================
GPU memory cleared

================================================================================
TESTING SPECULATIVE THINKING (Qwen3-1.7B + Qwen3-32B)
================================================================================
...

================================================================================
BENCHMARK RESULTS SUMMARY
================================================================================
OVERALL SPEEDUP: 1.61x
...

Results saved to: benchmark_results/speedup_benchmark_20260105_143022.json
```

## Manual Comparison

If you run tests separately, you can manually compare results by loading the JSON files:

```python
import json

# Load results
with open('benchmark_results/large_model_benchmark_20260105_143022.json') as f:
    large_results = json.load(f)

with open('benchmark_results/speculative_thinking_benchmark_20260105_143022.json') as f:
    spec_results = json.load(f)

# Calculate speedup
large_latency = large_results['statistics']['latency']['mean']
spec_latency = spec_results['statistics']['latency']['mean']
speedup = large_latency / spec_latency

print(f"Speedup: {speedup:.2f}x")
```

## Additional OOM Prevention Tips

1. **Reduce memory usage:**
   ```bash
   python spec_thinking_speedup_test.py \
       --test-mode spec \
       --num-problems 3 \
       --max-tokens 16384
   ```

2. **Check GPU memory before running:**
   ```bash
   nvidia-smi
   ```

3. **Kill existing Ray processes:**
   ```bash
   ray stop
   ```

4. **Adjust GPU memory utilization** in [speculative/speculative_vllm.py](speculative/speculative_vllm.py):
   - Small model: Line 22 (`gpu_memory_utilization=0.15`)
   - Large model: Line 31 (`gpu_memory_utilization=0.8`)

## Files Modified

1. [spec_thinking_speedup_test.py](spec_thinking_speedup_test.py) - Main script
2. [SPEEDUP_BENCHMARK_README.md](SPEEDUP_BENCHMARK_README.md) - Documentation
3. [OOM_FIX_SUMMARY.md](OOM_FIX_SUMMARY.md) - This file
