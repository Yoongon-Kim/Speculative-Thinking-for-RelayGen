#!/usr/bin/env python3
"""
Calculate token statistics from speculative decoding inference results.

This script analyzes the token_usages field in the JSON output from inference_and_check.py
to compute:
- Total tokens generated
- Tokens from small (speculative) model
- Tokens from large (target) model
- Acceptance rate and other metrics
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def calculate_tokens_from_corrections(completion_tokens: int, correct_tokens: List[Dict]) -> Tuple[int, int]:
    """
    Calculate how many tokens came from small vs large model.

    Args:
        completion_tokens: Total number of completion tokens
        correct_tokens: List of correction events from the large model

    Returns:
        Tuple of (small_model_tokens, large_model_tokens)
    """
    # Tokens from large model are all the corrections
    large_model_tokens = sum(correction.get('token_num', 0) for correction in correct_tokens)

    # Tokens from small model are the rest
    small_model_tokens = completion_tokens - large_model_tokens

    return small_model_tokens, large_model_tokens


def analyze_result_file(file_path: str) -> Dict:
    """
    Analyze a single result JSON file and compute token statistics.

    Args:
        file_path: Path to the JSON result file

    Returns:
        Dictionary containing analysis results
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    stats = {
        'total_completion_tokens': 0,
        'total_prompt_tokens': 0,
        'small_model_tokens': 0,
        'large_model_tokens': 0,
        'total_corrections': 0,
        'total_try_correct': 0,
        'total_samples': 0,
        'problems': 0,
        'correct_samples': 0,
        'total_generation_length': 0,
        'per_temperature': defaultdict(lambda: {
            'completion_tokens': 0,
            'prompt_tokens': 0,
            'small_model_tokens': 0,
            'large_model_tokens': 0,
            'corrections': 0,
            'try_correct': 0,
            'samples': 0,
            'correct_samples': 0
        })
    }

    for problem_key, problem_data in data.items():
        stats['problems'] += 1

        # Get responses to check correctness
        responses = problem_data.get('responses', {})

        # Get token usages
        token_usages = problem_data.get('token_usages', {})

        for temp, samples in token_usages.items():
            temp_stats = stats['per_temperature'][temp]

            # Get corresponding responses for this temperature
            temp_responses = responses.get(temp, [])

            for idx, sample in enumerate(samples):
                stats['total_samples'] += 1
                temp_stats['samples'] += 1

                completion_tokens = sample.get('completion_tokens', 0)
                prompt_tokens = sample.get('prompt_tokens', 0)
                correct_tokens = sample.get('correct_tokens', [])
                try_correct = sample.get('try_correct', 0)

                stats['total_completion_tokens'] += completion_tokens
                stats['total_prompt_tokens'] += prompt_tokens
                stats['total_try_correct'] += try_correct

                temp_stats['completion_tokens'] += completion_tokens
                temp_stats['prompt_tokens'] += prompt_tokens
                temp_stats['try_correct'] += try_correct

                # Track generation length (completion tokens per problem, not per sample)
                # We add to total_generation_length for averaging per problem later

                # Check correctness from responses
                if idx < len(temp_responses):
                    is_correct = temp_responses[idx].get('correctness', False)
                    if is_correct:
                        stats['correct_samples'] += 1
                        temp_stats['correct_samples'] += 1

                # Calculate small vs large model tokens
                small_tokens, large_tokens = calculate_tokens_from_corrections(
                    completion_tokens, correct_tokens
                )

                stats['small_model_tokens'] += small_tokens
                stats['large_model_tokens'] += large_tokens
                stats['total_corrections'] += len(correct_tokens)

                temp_stats['small_model_tokens'] += small_tokens
                temp_stats['large_model_tokens'] += large_tokens
                temp_stats['corrections'] += len(correct_tokens)

    # Calculate percentages and averages
    if stats['total_completion_tokens'] > 0:
        stats['small_model_percentage'] = (stats['small_model_tokens'] / stats['total_completion_tokens']) * 100
        stats['large_model_percentage'] = (stats['large_model_tokens'] / stats['total_completion_tokens']) * 100
    else:
        stats['small_model_percentage'] = 0
        stats['large_model_percentage'] = 0

    if stats['total_samples'] > 0:
        stats['avg_completion_tokens'] = stats['total_completion_tokens'] / stats['total_samples']
        stats['avg_prompt_tokens'] = stats['total_prompt_tokens'] / stats['total_samples']
        stats['avg_small_tokens'] = stats['small_model_tokens'] / stats['total_samples']
        stats['avg_large_tokens'] = stats['large_model_tokens'] / stats['total_samples']
        stats['avg_corrections_per_sample'] = stats['total_corrections'] / stats['total_samples']
        stats['avg_try_correct_per_sample'] = stats['total_try_correct'] / stats['total_samples']
        stats['accuracy'] = (stats['correct_samples'] / stats['total_samples']) * 100

    if stats['problems'] > 0:
        stats['avg_completion_tokens_per_problem'] = stats['total_completion_tokens'] / stats['problems']

    # Calculate per-temperature percentages
    for temp, temp_stats in stats['per_temperature'].items():
        if temp_stats['completion_tokens'] > 0:
            temp_stats['small_model_percentage'] = (temp_stats['small_model_tokens'] / temp_stats['completion_tokens']) * 100
            temp_stats['large_model_percentage'] = (temp_stats['large_model_tokens'] / temp_stats['completion_tokens']) * 100
        if temp_stats['samples'] > 0:
            temp_stats['avg_completion_tokens'] = temp_stats['completion_tokens'] / temp_stats['samples']
            temp_stats['avg_small_tokens'] = temp_stats['small_model_tokens'] / temp_stats['samples']
            temp_stats['avg_large_tokens'] = temp_stats['large_model_tokens'] / temp_stats['samples']
            temp_stats['accuracy'] = (temp_stats['correct_samples'] / temp_stats['samples']) * 100

    return stats


def print_statistics(stats: Dict, file_path: str):
    """
    Pretty print the statistics.

    Args:
        stats: Statistics dictionary from analyze_result_file
        file_path: Path to the analyzed file
    """
    print("=" * 80)
    print(f"Token Statistics for: {Path(file_path).name}")
    print("=" * 80)
    print(f"\nTotal Problems: {stats['problems']}")
    print(f"Total Samples: {stats['total_samples']}")
    print(f"Correct Samples: {stats['correct_samples']}")
    print(f"Accuracy: {stats.get('accuracy', 0):.2f}%")

    print("\n" + "-" * 80)
    print("OVERALL STATISTICS")
    print("-" * 80)

    print(f"\nPrompt Tokens:")
    print(f"  Total:   {stats['total_prompt_tokens']:,}")
    print(f"  Average: {stats.get('avg_prompt_tokens', 0):.2f}")

    print(f"\nCompletion Tokens:")
    print(f"  Total:                    {stats['total_completion_tokens']:,}")
    print(f"  Average per Sample:       {stats.get('avg_completion_tokens', 0):.2f}")
    print(f"  Average per Problem:      {stats.get('avg_completion_tokens_per_problem', 0):.2f}")

    print(f"\nSmall Model (Speculative) Tokens:")
    print(f"  Total:      {stats['small_model_tokens']:,}")
    print(f"  Percentage: {stats['small_model_percentage']:.2f}%")
    print(f"  Average:    {stats.get('avg_small_tokens', 0):.2f} per sample")

    print(f"\nLarge Model (Target) Tokens:")
    print(f"  Total:      {stats['large_model_tokens']:,}")
    print(f"  Percentage: {stats['large_model_percentage']:.2f}%")
    print(f"  Average:    {stats.get('avg_large_tokens', 0):.2f} per sample")

    print(f"\nCorrections:")
    print(f"  Total Correction Events: {stats['total_corrections']}")
    print(f"  Average per Sample:      {stats.get('avg_corrections_per_sample', 0):.2f}")
    print(f"  Total Try Correct:       {stats['total_try_correct']}")
    print(f"  Average per Sample:      {stats.get('avg_try_correct_per_sample', 0):.2f}")

    # Per-temperature statistics
    if stats['per_temperature']:
        print("\n" + "-" * 80)
        print("PER-TEMPERATURE STATISTICS")
        print("-" * 80)

        for temp in sorted(stats['per_temperature'].keys(), key=lambda x: float(x)):
            temp_stats = stats['per_temperature'][temp]
            print(f"\nTemperature {temp}:")
            print(f"  Samples:              {temp_stats['samples']}")
            print(f"  Correct Samples:      {temp_stats['correct_samples']}")
            print(f"  Accuracy:             {temp_stats.get('accuracy', 0):.2f}%")
            print(f"  Total Completion:     {temp_stats['completion_tokens']:,}")
            print(f"  Small Model Tokens:   {temp_stats['small_model_tokens']:,} ({temp_stats.get('small_model_percentage', 0):.2f}%)")
            print(f"  Large Model Tokens:   {temp_stats['large_model_tokens']:,} ({temp_stats.get('large_model_percentage', 0):.2f}%)")
            print(f"  Avg Small per Sample: {temp_stats.get('avg_small_tokens', 0):.2f}")
            print(f"  Avg Large per Sample: {temp_stats.get('avg_large_tokens', 0):.2f}")
            print(f"  Total Corrections:    {temp_stats['corrections']}")

    print("\n" + "=" * 80)


def save_statistics(stats: Dict, output_path: str):
    """
    Save statistics to a JSON file.

    Args:
        stats: Statistics dictionary
        output_path: Path to save the JSON file
    """
    # Convert defaultdict to regular dict for JSON serialization
    stats_copy = dict(stats)
    stats_copy['per_temperature'] = dict(stats_copy['per_temperature'])

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats_copy, f, indent=2, ensure_ascii=False)

    print(f"\nStatistics saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate token statistics from speculative decoding results'
    )
    parser.add_argument(
        'result_file',
        type=str,
        help='Path to the inference result JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save statistics JSON (optional)'
    )

    args = parser.parse_args()

    # Check if file exists
    if not Path(args.result_file).exists():
        print(f"Error: File not found: {args.result_file}")
        return 1

    # Analyze the file
    print(f"Analyzing: {args.result_file}")
    print("Please wait...\n")

    stats = analyze_result_file(args.result_file)

    # Print statistics
    print_statistics(stats, args.result_file)

    # Save to file if requested
    if args.output:
        save_statistics(stats, args.output)

    return 0


if __name__ == '__main__':
    exit(main())
