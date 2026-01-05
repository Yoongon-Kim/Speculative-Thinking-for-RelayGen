# GPU Sharing Fix for 2-GPU Setup

## Problem

The speculative thinking implementation creates **two separate Ray actors**:
1. Speculative model actor requesting GPUs
2. Target model actor requesting GPUs

Ray sees these as separate resource requests and wants **exclusive GPU allocation** for each actor, leading to:
- Config: `target_model_gpu: 2`, `speculative_model_gpu: 1` → Needs 3 GPUs
- Your setup: Only 2 GPUs available → **Resource deadlock**

## Solutions

### Option 1: Use CUDA_VISIBLE_DEVICES (Recommended for 2 GPUs)

Set the same GPUs for both models using environment variables:

```bash
CUDA_VISIBLE_DEVICES=0,1 python spec_thinking_speedup_test.py \
    --test-mode spec \
    --max-tokens 200
```

This forces both models to see only GPUs 0 and 1, and vLLM will manage memory within those GPUs.

### Option 2: Run Tests Separately (Easiest)

Run only one model at a time:

**For baseline (large model only):**
```bash
python spec_thinking_speedup_test.py \
    --test-mode large \
    --max-tokens 200
```

**For speculative thinking, you can't run both models simultaneously with 2 GPUs using the current Ray actor setup.**

### Option 3: Modify speculative_vllm.py to Use Fractional GPUs

Edit `speculative/speculative_vllm.py` to allow GPU sharing:

**Current (line 21):**
```python
@ray.remote(num_gpus=target_model_gpu)
```

**Change to:**
```python
@ray.remote(num_gpus=target_model_gpu, num_cpus=1, resources={"shared_gpu": target_model_gpu})
```

But this requires more complex Ray configuration and may not work well with vLLM.

### Option 4: Don't Use Ray Actors (Best for 2 GPUs)

The fundamental issue is that the current implementation assumes you have enough GPUs for separate actors. For a 2-GPU setup, you need to load both models without Ray actors.

Create a new config that doesn't use separate Ray workers:

**Create `speculative/config/qwen3_1.7b_32b_sequential.yml`:**
```yaml
mode: "hf"  # Use HuggingFace mode instead of vLLM
target_model_name: "Qwen/Qwen3-32B"
speculative_model_name: "Qwen/Qwen3-1.7B"
# ... rest of config
```

## Recommended Approach for Your Use Case

Given you only have 2 GPUs and want to run speculative thinking:

### Option A: Test Only Large Model
```bash
python spec_thinking_speedup_test.py \
    --test-mode large \
    --max-tokens 200
```

This works fine with 2 GPUs.

### Option B: Sequential Testing (Get Both Results)

**Step 1: Test large model**
```bash
python spec_thinking_speedup_test.py \
    --test-mode large \
    --max-tokens 200 \
    --num-problems 5 \
    --num-runs 5
```
Output: `benchmark_results/large_model_benchmark_TIMESTAMP.json`

**Step 2: Kill Ray and test with different GPU strategy**

Unfortunately, the current speculative thinking implementation with vLLM Ray actors **requires 3 GPUs minimum** (1 for spec model + 2 for target model).

## Why This Is Difficult

The issue is architectural:

1. **vLLM with Tensor Parallelism**: When you set `tensor_parallel_size=2`, vLLM expects to have 2 GPUs exclusively for that model instance.

2. **Ray Actors**: Each Ray actor requests exclusive GPU allocation via `num_gpus=N`.

3. **Two Separate Actors**: The spec thinking implementation creates TWO actors:
   ```python
   self.target_model = create_ray_model(..., target_model_gpu=2)  # Requests 2 GPUs
   self.speculative_model = create_ray_model(..., speculative_model_gpu=1)  # Requests 1 GPU
   ```

4. **Ray's Resource Manager**: Ray sees these as separate requests and wants 3 GPUs total.

## Actual Solution: Modify the Code

To truly support 2-GPU speculative thinking, you need to modify `speculative/speculative_vllm.py` to:

1. Load both models sequentially (not as separate Ray actors)
2. Use the same 2 GPUs with lower memory utilization for each
3. Or implement a custom scheduling mechanism

Would you like me to create a modified version of `speculative_vllm.py` that works with 2 GPUs?

## Temporary Workaround

For now, to test speculative thinking with your 2 GPUs:

1. Edit `speculative/config/qwen3_1.7b_32b.yml`:
   ```yaml
   target_model_gpu: 1  # Change from 2 to 1
   speculative_model_gpu: 1  # Keep at 1
   ```

2. Run:
   ```bash
   python spec_thinking_speedup_test.py \
       --test-mode spec \
       --max-tokens 200
   ```

This will use 2 GPUs total (1+1) but each model will only use 1 GPU (no tensor parallelism). The 32B model will be slow and may run out of memory on a single GPU.
