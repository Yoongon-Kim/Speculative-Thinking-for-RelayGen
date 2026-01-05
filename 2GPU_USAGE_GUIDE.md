# 2-GPU Setup Usage Guide

## Problem Solved

The original implementation uses separate Ray actors for the target and speculative models, requiring:
- **3 GPUs minimum**: 1 for spec model + 2 for target model
- **Ray resource allocation conflicts** when only 2 GPUs are available

## Solution

Created `speculative_vllm_2gpu.py` that:
- Loads both models in the **same process**
- Uses **same 2 GPUs** with tensor parallelism
- Reduces memory utilization to **0.4 for each model** (fits both simultaneously)
- **No Ray actor conflicts**

## Usage

### Option 1: Use the `--use-2gpu` Flag (Recommended)

```bash
python spec_thinking_speedup_test.py \
    --test-mode spec \
    --use-2gpu \
    --max-tokens 200
```

This automatically uses the 2-GPU optimized version.

### Option 2: Update Your Config File

If you want the 2-GPU version to be the default, edit your config to ensure both models use 2 GPUs:

**`speculative/config/qwen3_1.7b_32b_2gpu.yml`:**
```yaml
mode: "vllm"
target_model_name: "Qwen/Qwen3-32B"
speculative_model_name: "Qwen/Qwen3-1.7B"
target_model_gpu: 2
speculative_model_gpu: 2  # Both use same 2 GPUs
```

Then run with the --use-2gpu flag:
```bash
python spec_thinking_speedup_test.py \
    --config speculative/config/qwen3_1.7b_32b_2gpu.yml \
    --test-mode spec \
    --use-2gpu \
    --max-tokens 200
```

## Complete Examples

### Test Speculative Thinking Only (2-GPU Mode)
```bash
python spec_thinking_speedup_test.py \
    --test-mode spec \
    --use-2gpu \
    --num-problems 5 \
    --num-runs 5 \
    --max-tokens 32768
```

### Test Large Model Only (Standard Mode)
```bash
python spec_thinking_speedup_test.py \
    --test-mode large \
    --num-problems 5 \
    --max-tokens 32768
```

### Run Both Tests Separately (2-GPU Compatible)

**Step 1: Test large model**
```bash
python spec_thinking_speedup_test.py \
    --test-mode large \
    --max-tokens 32768
```

**Step 2: Test speculative thinking (2-GPU mode)**
```bash
python spec_thinking_speedup_test.py \
    --test-mode spec \
    --use-2gpu \
    --max-tokens 32768
```

## Technical Details

### Memory Utilization

**2-GPU Mode (`speculative_vllm_2gpu.py`):**
- Target model (Qwen3-32B): `gpu_memory_utilization=0.4` (40%)
- Speculative model (Qwen3-1.7B): `gpu_memory_utilization=0.4` (40%)
- Total: ~80% GPU memory used
- Both models share the same 2 GPUs

**Standard Mode (`speculative_vllm.py`):**
- Target model: `gpu_memory_utilization=0.8` (80%) on 2 GPUs
- Speculative model: `gpu_memory_utilization=0.15` (15%) on 1 GPU
- Total: Requires 3 GPUs (Ray actor isolation)

### Key Differences

| Feature | Standard Mode | 2-GPU Mode |
|---------|--------------|------------|
| GPU Requirement | 3 GPUs minimum | 2 GPUs |
| Ray Actors | Separate actors | Single process |
| Memory per Model | Target: 80%, Spec: 15% | Both: 40% |
| Model Loading | Sequential via Ray | Direct in same process |
| Resource Conflicts | Yes (if <3 GPUs) | No |

## Troubleshooting

### Still Getting OOM Errors?

If you still run out of memory, reduce the memory utilization or max tokens:

1. **Edit `speculative/speculative_vllm_2gpu.py`** and reduce memory:
   ```python
   # Line ~66 and ~78
   gpu_memory_utilization=0.3  # Reduced from 0.4
   ```

2. **Reduce max_tokens**:
   ```bash
   python spec_thinking_speedup_test.py \
       --test-mode spec \
       --use-2gpu \
       --max-tokens 8192  # Reduced from 32768
   ```

3. **Use smaller models** or test fewer problems:
   ```bash
   python spec_thinking_speedup_test.py \
       --test-mode spec \
       --use-2gpu \
       --num-problems 1 \
       --num-runs 1
   ```

### Ray Warnings?

When using `--use-2gpu`, you might still see Ray warnings. These can be ignored because the 2-GPU mode doesn't use Ray actors for model loading.

To suppress Ray warnings:
```bash
export RAY_DEDUP_LOGS=0
python spec_thinking_speedup_test.py --test-mode spec --use-2gpu
```

## Performance Considerations

### Expected Performance

The 2-GPU mode may be slightly slower than the standard 3-GPU mode because:
1. Both models share the same GPU memory bandwidth
2. Lower memory utilization means potentially more swapping
3. No parallel loading via Ray actors

However, it enables testing that wouldn't be possible otherwise with only 2 GPUs.

### Optimization Tips

1. **Use smaller test sets first**:
   ```bash
   --num-problems 1 --num-runs 1
   ```

2. **Monitor GPU usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **If one model is much larger**, consider adjusting memory ratios in `speculative_vllm_2gpu.py`:
   ```python
   # Target model (32B - needs more memory)
   gpu_memory_utilization=0.5

   # Speculative model (1.7B - needs less)
   gpu_memory_utilization=0.3
   ```

## Files Created

1. **`speculative/speculative_vllm_2gpu.py`** - 2-GPU optimized implementation
2. **`speculative/config/qwen3_1.7b_32b_2gpu.yml`** - Config for 2-GPU setup
3. **`2GPU_USAGE_GUIDE.md`** - This file
4. **`GPU_SHARING_FIX.md`** - Technical explanation

## Summary

To run speculative thinking benchmark on 2 GPUs:

```bash
python spec_thinking_speedup_test.py \
    --test-mode spec \
    --use-2gpu \
    --max-tokens 200
```

This will successfully run both models on your 2 available GPUs!
