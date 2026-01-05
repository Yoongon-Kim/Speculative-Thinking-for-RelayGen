# Benchmark Script Changes Summary

## Changes Made

### 1. YAML Template Loading (✓ Completed)

**What changed:**
- Now loads prompt template from `skythought_evals/tasks/aime/aime25.yaml`
- Uses the `templating_parameters.template` field from the YAML config

**Code location:**
- [spec_thinking_speedup_test.py:76-79](spec_thinking_speedup_test.py#L76-L79) - `load_aime_config()` method
- [spec_thinking_speedup_test.py:89](spec_thinking_speedup_test.py#L89) - Template usage

**Before:**
```python
prompt = (
    "Solve the following math problem efficiently and clearly. "
    "Please reason step by step, separate logical reasoning steps with "
    "two newline characters (\\n\\n), and put your final answer within \\boxed{{}}.\n"
    f"Problem: {problem['problem']}"
)
```

**After:**
```python
# Loads template from YAML
self.prompt_template = self.aime_config['templating_parameters']['template']
prompt = self.prompt_template.format(prompt=problem['problem'])
```

### 2. Command-Line Arguments (✓ Completed)

**What changed:**
- Added full argparse support for all parameters
- Can now customize everything via command line

**Arguments added:**
```bash
--config PATH               # Speculative thinking config file
--aime-config PATH          # AIME25 task config file
--num-problems N            # Number of problems to test
--num-runs N                # Runs per problem
--temperature FLOAT         # Sampling temperature
--top-p FLOAT              # Nucleus sampling
--top-k INT                # Top-k sampling
```

**Code location:**
- [spec_thinking_speedup_test.py:421-473](spec_thinking_speedup_test.py#L421-L473) - Argument parser setup

### 3. Updated Default Sampling Parameters (✓ Completed)

**What changed:**
- Default top_k changed from 50 to 20
- All other defaults remain: temperature=0.6, top_p=0.95

**Code locations:**
- [spec_thinking_speedup_test.py:41-43](spec_thinking_speedup_test.py#L41-L43) - Constructor signature
- [spec_thinking_speedup_test.py:147](spec_thinking_speedup_test.py#L147) - Large model generation
- [spec_thinking_speedup_test.py:220](spec_thinking_speedup_test.py#L220) - Speculative thinking generation

### 4. Added Max Tokens Parameter (✓ Completed)

**What changed:**
- Added `--max-tokens` command-line argument
- Default value: 32768 tokens
- Now configurable instead of hardcoded

**Code locations:**
- [spec_thinking_speedup_test.py:43](spec_thinking_speedup_test.py#L43) - Constructor parameter
- [spec_thinking_speedup_test.py:147](spec_thinking_speedup_test.py#L147) - Large model generation
- [spec_thinking_speedup_test.py:220](spec_thinking_speedup_test.py#L220) - Speculative thinking generation
- [spec_thinking_speedup_test.py:475-480](spec_thinking_speedup_test.py#L475-L480) - Argument parser

**Before:**
```python
generated_text, num_tokens, _, _ = model.generate(
    messages=messages,
    max_tokens=8192,  # Hardcoded
    ...
)
```

**After:**
```python
generated_text, num_tokens, _, _ = model.generate(
    messages=messages,
    max_tokens=self.max_tokens,  # Configurable via --max-tokens
    ...
)
```

## Usage Examples

### Basic usage (uses all defaults)
```bash
python spec_thinking_speedup_test.py
```

### Custom sampling parameters
```bash
python spec_thinking_speedup_test.py --temperature 0.6 --top-p 0.95 --top-k 20 --max-tokens 32768
```

### Custom AIME config (with your modified template)
```bash
python spec_thinking_speedup_test.py --aime-config path/to/custom_aime.yaml
```

### Full customization
```bash
python spec_thinking_speedup_test.py \
    --config speculative/config/qwen3_1.7b_32b.yml \
    --aime-config skythought_evals/tasks/aime/aime25.yaml \
    --num-problems 5 \
    --num-runs 5 \
    --temperature 0.6 \
    --top-p 0.95 \
    --top-k 20
```

## How to Customize the Prompt Template

Edit `skythought_evals/tasks/aime/aime25.yaml`:

```yaml
templating_parameters:
  template: "Your custom prompt here. Problem: {prompt}"
```

The `{prompt}` placeholder will be replaced with the actual problem text.

## GPU Utilization Rates

**Question:** Where are GPU utilization rates set?

**Answer:** In [speculative/speculative_vllm.py:22 and :30](speculative/speculative_vllm.py#L22):

```python
# Small model (speculative)
gpu_memory_utilization=0.2  # 20%

# Large model (target)
gpu_memory_utilization=0.9  # 90%
```

These are **hardcoded** in the `create_ray_model()` function and controlled by the `is_spec_model` parameter.

To change these values, you would need to modify `speculative/speculative_vllm.py` directly.

## Files Modified

1. **[spec_thinking_speedup_test.py](spec_thinking_speedup_test.py)** - Main benchmark script
   - Added argparse support
   - Added YAML template loading
   - Updated sampling parameter defaults
   - Made everything configurable

2. **[SPEEDUP_BENCHMARK_README.md](SPEEDUP_BENCHMARK_README.md)** - Documentation
   - Updated usage instructions
   - Added command-line argument documentation
   - Added examples

3. **[BENCHMARK_CHANGES_SUMMARY.md](BENCHMARK_CHANGES_SUMMARY.md)** - This file
   - Summary of all changes
   - Quick reference guide
