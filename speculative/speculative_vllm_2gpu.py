"""
Modified speculative thinking implementation for 2-GPU setups.

This version loads both models in the same process on the same 2 GPUs,
avoiding the Ray actor resource allocation issue.

Key differences from speculative_vllm.py:
1. No separate Ray actors for each model
2. Both models use the same 2 GPUs with tensor parallelism
3. Lower memory utilization to fit both models simultaneously
"""

import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM
import time

# Add parent directory to path if not already there
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from speculative.spe_utils import *
from utils.data_utils import read_yml
from utils.utils import *
from utils.qwen_math_parser import *
from vllm import LLM, SamplingParams


class spe_thinking_vllm_2gpu:
    """Speculative thinking implementation optimized for 2-GPU setup."""

    def __init__(self, **config):
        """
        Initialize models for 2-GPU setup.

        Both models will be loaded on the same 2 GPUs with reduced memory utilization.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(config['target_model_name'])
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Configuration
        self.config = config
        self.help_think_word_ids = None if config['help_think_word'] is None else \
            self.tokenizer([config['help_think_word']], return_tensors="np", add_special_tokens=False)["input_ids"][0].tolist()
        self.help_recap_words_ids = self.tokenizer([config['help_recap_words']], return_tensors="np",
                                                    add_special_tokens=False)["input_ids"][0].tolist()
        self.TRIGGER_TOKENS = config['TRIGGER_TOKENS']
        self.TARGET_VALIDATION_KEYWORDS = config['TARGET_VALIDATION_KEYWORDS']
        self.choose_large = config['choose_large']
        self.not_reasoning = config.get('not_reasoning', False)

        # Load models on the same 2 GPUs
        print("Loading models on 2 GPUs...")
        print(f"  Target model: {config['target_model_name']}")
        print(f"  Speculative model: {config.get('speculative_model_name', 'None')}")

        # Target model with reduced memory utilization to fit both models
        self.target_model = LLM(
            model=config['target_model_name'],
            tensor_parallel_size=2,  # Use both GPUs
            dtype='bfloat16',
            gpu_memory_utilization=0.8,  # Reduced from 0.8 to fit both models
            max_model_len=32768,
            trust_remote_code=True,
            enforce_eager=True,
        )
        print("  Target model loaded successfully")

        # Speculative model (if specified)
        self.speculative_model = None
        if config.get('speculative_model_name') is not None:
            self.speculative_model = LLM(
                model=config['speculative_model_name'],
                tensor_parallel_size=2,  # Use same 2 GPUs
                dtype='bfloat16',
                gpu_memory_utilization=0.15,  # Reduced to fit alongside target model
                trust_remote_code=True,
                max_model_len=32768,
            )
            print("  Speculative model loaded successfully")

        print("Both models ready on 2 GPUs\n")

    def generate(self, messages=None, max_tokens=1024, temperature=0.6, top_k=50, top_p=0.95):
        """Generate text using speculative thinking or normal mode."""
        if self.speculative_model is None:
            return self.normal_generate(messages, max_tokens, temperature, top_k, top_p)
        else:
            return self.speculative_generate(messages, max_tokens, temperature, top_k, top_p)

    def get_prompt_len(self, messages):
        """Get the length of the prompt in tokens."""
        generated_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="np", add_generation_prompt=True, enable_thinking=True
        ).tolist()[0]
        return len(generated_ids)

    def normal_generate(self, messages=None, max_tokens=1024, temperature=0.6, top_k=50, top_p=0.95):
        """Generate using only the target model."""
        generated_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="np", add_generation_prompt=True, enable_thinking=True
        ).tolist()[0]
        prompt_len = len(generated_ids)

        # Ensure we don't exceed the model's max context length
        max_model_len = 32768
        adjusted_max_tokens = min(max_tokens, max_model_len - prompt_len - 1024)

        sampling_params = SamplingParams(
            max_tokens=adjusted_max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            skip_special_tokens=False
        )

        outputs = self.target_model.generate(
            prompt_token_ids=[generated_ids],
            sampling_params=sampling_params,
            use_tqdm=False
        )

        output_ids = list(outputs[0].outputs[0].token_ids)
        generated_ids.extend(output_ids)
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        num_tokens = len(output_ids)
        correct_tokens = []
        try_correct_num = 0

        return generated_text, num_tokens, correct_tokens, try_correct_num

    def speculative_generate(self, messages=None, max_tokens=1024, temperature=0.6, top_k=50, top_p=0.95):
        """Generate using speculative thinking with both models."""
        start_time = time.time()
        stops = self.TRIGGER_TOKENS

        sampling_params_one = SamplingParams(
            max_tokens=1024,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            skip_special_tokens=False,
            stop=stops
        )

        tgt_sampling_params_cache = SamplingParams(
            max_tokens=self.config['max_target_tokens'],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            skip_special_tokens=False
        )

        token_num, change_tokens, change_flag, begin = 0, 0, False, self.config['begin']
        negative_sent_num, recap_token_num = 0, self.config['original_recap_token_num']

        generated_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="np", add_generation_prompt=True, enable_thinking=True
        ).tolist()[0]
        prompt_len = len(generated_ids)
        correct_tokens, try_correct_num = [], 0
        recap_after_negtive_num = self.config['recap_after_negative_num']

        # Ensure we don't exceed the model's max context length
        max_model_len = 32768
        max_generation_tokens = min(max_tokens, max_model_len - prompt_len - 1024)

        while token_num < max_generation_tokens:
            if self.config['time_out'] is not None and self.config['time_out'] > 0:
                use_time = time.time() - start_time
                if use_time > self.config['time_out']:
                    return None

            if not begin:
                # Generate with speculative model
                outputs = self.speculative_model.generate(
                    prompt_token_ids=[generated_ids],
                    sampling_params=sampling_params_one,
                    use_tqdm=False
                )
                one_token_id = list(outputs[0].outputs[0].token_ids)
                generated_ids.extend(one_token_id)

                if one_token_id[-1] == self.tokenizer.eos_token_id:
                    break

                one_token = self.tokenizer.decode(one_token_id[-5:], skip_special_tokens=True)

            if begin or any(trigger in one_token for trigger in self.TRIGGER_TOKENS):
                if begin:
                    change_tokens = self.config['begin_token_num']
                    begin = False
                    change_flag = True
                    tgt_kv_candidate = None
                    spe_decoded_text = ''
                elif negative_sent_num >= recap_after_negtive_num:
                    generated_ids.extend(self.help_recap_words_ids)
                    change_tokens = recap_token_num
                    change_flag = True
                    negative_sent_num = 0
                    recap_token_num = min(
                        recap_token_num + self.config['add_each_recap'],
                        self.config['max_recap_token_num']
                    )
                    recap_after_negtive_num = min(
                        recap_after_negtive_num + self.config['add_each_neg'],
                        self.config['max_negative_num']
                    )
                else:
                    if self.help_think_word_ids is not None:
                        generated_ids.extend(self.help_think_word_ids)

                    # Generate with speculative model
                    spe_outputs = self.speculative_model.generate(
                        prompt_token_ids=[generated_ids],
                        sampling_params=tgt_sampling_params_cache,
                        use_tqdm=False
                    )
                    spe_ids = list(spe_outputs[0].outputs[0].token_ids)
                    spe_token = self.tokenizer.decode(spe_ids, skip_special_tokens=True)
                    spe_sent = sentiment_analysis(
                        spe_token,
                        self.TARGET_VALIDATION_KEYWORDS['positive'],
                        self.TARGET_VALIDATION_KEYWORDS['negative'] + self.TARGET_VALIDATION_KEYWORDS['verify']
                    )

                    if self.not_reasoning or spe_sent != 0:
                        try_correct_num = try_correct_num + 1

                        # Generate with target model
                        tgt_outputs = self.target_model.generate(
                            prompt_token_ids=[generated_ids],
                            sampling_params=tgt_sampling_params_cache,
                            use_tqdm=False
                        )
                        tgt_ids = list(tgt_outputs[0].outputs[0].token_ids)
                        tgt_token = self.tokenizer.decode(tgt_ids, skip_special_tokens=True)
                        tgt_sent = sentiment_analysis(
                            tgt_token,
                            self.TARGET_VALIDATION_KEYWORDS['positive'],
                            self.TARGET_VALIDATION_KEYWORDS['negative'] + self.TARGET_VALIDATION_KEYWORDS['verify']
                        )

                        if self.choose_large or (spe_sent < 0 and tgt_sent >= 0) or (spe_sent > 0 and tgt_sent < 0):
                            decode_text = tgt_token
                            correct_tokens.append({
                                'pos': len(generated_ids) - prompt_len,
                                'token_num': self.config['max_target_tokens'],
                                'traget': tgt_token,
                                'speculative': spe_token
                            })
                            generated_ids.extend(tgt_ids)
                            final_sent = tgt_sent
                        else:
                            generated_ids.extend(spe_ids)
                            decode_text = spe_token
                            final_sent = spe_sent

                        if final_sent < 0:
                            negative_sent_num = negative_sent_num + 1
                        if contains_keywords(decode_text, self.TARGET_VALIDATION_KEYWORDS['verify']):
                            change_tokens = self.config['original_recap_token_num']
                            change_flag = True

                if change_flag:
                    try_correct_num = try_correct_num + 1
                    tgt_sampling_params_ = SamplingParams(
                        max_tokens=change_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        skip_special_tokens=False
                    )
                    tgt_outputs = self.target_model.generate(
                        prompt_token_ids=[generated_ids],
                        sampling_params=tgt_sampling_params_,
                        use_tqdm=False
                    )
                    tgt_ids = list(tgt_outputs[0].outputs[0].token_ids)
                    tgt_token = self.tokenizer.decode(tgt_ids, skip_special_tokens=True)
                    correct_tokens.append({
                        'pos': len(generated_ids) - prompt_len,
                        'token_num': change_tokens,
                        'traget': tgt_token
                    })
                    generated_ids.extend(tgt_ids)
                    change_flag = False

            token_num = len(generated_ids)
            if self.tokenizer.eos_token_id in generated_ids[-self.config['max_target_tokens']:]:
                break

        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text, len(generated_ids) - prompt_len, correct_tokens, try_correct_num


if __name__ == "__main__":
    yml_path = 'speculative/config/qwen3_1.7b_32b_2gpu.yml'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = read_yml(yml_path)
    model = spe_thinking_vllm_2gpu(**config)

    messages = []
    messages.append({
        "role": "user",
        "content": "Please reason step by step, and put your final answer within \\boxed{}. " +
                  'What is 2+2?' + ' <think>\n'
    })

    result = model.generate(messages, 1024)
    print(result)
