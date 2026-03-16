"""
Download LiveCodeBench v5 test cases from HuggingFace.

Copied from QwQ/eval/data/process_data.py and adapted to save test files
into skythought_evals/tasks/livecodebench/livecodebench_v5_tests/.

Usage:
    python -m skythought_evals.tasks.livecodebench.process_data
"""

import json
import zlib
import pickle
import base64
import hashlib
from enum import Enum
from datetime import datetime
from dataclasses import dataclass

from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)


@dataclass
class CodeGenerationProblem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[Test]
    private_test_cases: list[Test]
    metadata: dict

    def __post_init__(self):
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        self.contest_date = datetime.fromisoformat(self.contest_date)

        self.public_test_cases = json.loads(self.public_test_cases)  # type: ignore
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]

        try:
            self.private_test_cases = json.loads(self.private_test_cases)  # type: ignore
        except Exception:
            self.private_test_cases = json.loads(pickle.loads(zlib.decompress(base64.b64decode(self.private_test_cases.encode("utf-8"))  # type: ignore
                                                                             )))  # type: ignore
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]

        self.metadata = json.loads(self.metadata)  # type: ignore

    def get_evaluation_sample(self):
        return {
            "input_output": json.dumps({
                "inputs": [t.input for t in self.public_test_cases + self.private_test_cases],
                "outputs": [t.output for t in self.public_test_cases + self.private_test_cases],
                "fn_name": self.metadata.get("func_name", None),
            }),
        }


def load_code_generation_dataset(release_version="release_v5") -> list[CodeGenerationProblem]:
    dataset = load_dataset("livecodebench/code_generation_lite", split="test", version_tag=release_version, trust_remote_code=True)
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    print(f"Loaded {len(dataset)} problems")
    return dataset


def calculate_string_md5(input_string: str):
    md5 = hashlib.md5()
    md5.update(input_string.encode('utf-8'))
    return md5.hexdigest()


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    output_tests_dir = script_dir / "livecodebench_v5_tests"
    output_tests_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_code_generation_dataset(release_version="release_v5")

    for global_id, sample in enumerate(tqdm(dataset)):
        inputs_outputs = sample.get_evaluation_sample()

        # save test cases
        with open(output_tests_dir / f"{global_id}.json", "w") as f:
            json.dump(inputs_outputs, f)

    print(f"Saved {len(dataset)} test files to {output_tests_dir}")
