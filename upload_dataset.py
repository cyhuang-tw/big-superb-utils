import argparse
from pathlib import Path

from datasets import load_dataset, Audio, Features, Value


def main(data_dir: Path, remote_path: str) -> None:
    # The order in schema matters.
    schema = Features(
        {
            "file": Value("string"),
            "audio": Audio(),
            "instruction": Value("string"),
            "label": Value("string"),
        }
    )
    data = load_dataset(str(data_dir), features=schema)

    # You must login first, otherwise you are not able to upload datasets.
    # To login, run "huggingface-cli login" in your terminal.
    data.push_to_hub(remote_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("remote_path", type=str)
    main(**vars(parser.parse_args()))
