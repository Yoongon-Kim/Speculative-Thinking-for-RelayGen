import json
import os

import pandas as pd

from .livecodebench_handler import LiveCodeBenchTaskHandler


class LiveCodeBenchV5TaskHandler(LiveCodeBenchTaskHandler):

    def generate_prompt(self, problem):
        # JSONL prompt is already fully formatted — return as-is
        return problem["prompt"]

    def load_and_filter_dataset(
        self, start, end, split=None, subset=None, difficulty=None, args=None
    ):
        jsonl_path = self.task_config.preprocess_config["jsonl_path"]
        tests_dir = self.task_config.preprocess_config["tests_dir"]

        # Load JSONL
        items = []
        with open(jsonl_path) as f:
            for line in f:
                items.append(json.loads(line))

        # Build rows in the format the existing handler expects
        rows = []
        for item in items:
            # Load test cases from downloaded test files
            test_path = os.path.join(tests_dir, item["tests"]["fname"])
            with open(test_path) as tf:
                test_data = json.load(tf)

            # Parse input_output (it's a JSON string inside the JSON)
            io_data = json.loads(test_data["input_output"])
            inputs = io_data["inputs"]
            outputs = io_data["outputs"]
            fn_name = io_data.get("fn_name")

            # Convert to existing handler's test format
            testtype = "functional" if fn_name else "stdin"
            test_cases = [
                {"input": inp, "output": out, "testtype": testtype}
                for inp, out in zip(inputs, outputs)
            ]

            rows.append({
                "task_id": item["question_id"],
                "prompt": item["prompt"],
                "test": test_cases,
                "is_stdin": fn_name is None,
                "difficulty": item.get("difficulty", "unknown"),
                "entry_point": "",
                "public_test_cases": "[]",
                "canonical_solution": "",
            })

        df = pd.DataFrame(rows)

        # Filter by difficulty if requested
        if difficulty or "difficulty" in self.task_config.preprocess_config:
            difficulty = (
                difficulty
                if difficulty
                else self.task_config.preprocess_config["difficulty"]
            )
            df = df[df["difficulty"] == difficulty]

        return df.iloc[start:end] if end > 0 else df.iloc[start:]
