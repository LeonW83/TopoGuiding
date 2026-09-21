"""Average the test results of each model across all seeds"""

import json
from pathlib import Path
import pandas as pd


def process_results(
    results_path: str, filename: str = "test_results.json"
) -> None:
    """

    :param results_path: Path to the result folder.
    :param filename: Name of the files to average.
    :return: Nothing.
    """
    base_path = Path(results_path)

    seed_files = list(base_path.glob(f"**/seed_*/{filename}"))

    groups = {}
    for path in seed_files:
        exp_dir = path.parent.parent
        groups.setdefault(exp_dir, []).append(path)

    for exp_dir, files in groups.items():
        data_list = []
        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                data_list.append(json.load(f))

        df = pd.DataFrame(data_list)

        output = {
            "files": [str(p) for p in files],
            "mean": df.mean(numeric_only=True).to_dict(),
            "std": df.std(numeric_only=True).to_dict(),
        }

        out_path = exp_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4)


if __name__ == "__main__":
    process_results("./results")