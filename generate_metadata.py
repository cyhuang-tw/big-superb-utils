import argparse
import random
from pathlib import Path

from tqdm import tqdm


class InstructionGenerator:
    # Instructions used in train & validation (and probably test).
    tr_instrs = [
        "Detect whether the provided audio has been produced using a speech enhancement model. The answer could be yes or no.",
        "Establish whether the given audio is the result produced by a speech enhancement model. The answer could be yes or no.",
        "Identify whether the given audio has undergone generation or processing via a speech enhancement model. The answer could be yes or no.",
        "Determine if the provided audio has been synthetically generated using a speech enhancement model. The answer could be yes or no.",
        "Find out if the given audio is a product of generation by a speech enhancement model. The answer could be yes or no.",
        "Establish whether the provided audio is the outcome of artificial generation using a speech enhancement model. The answer could be yes or no.",
        "Detect if the given audio is created or manipulated by a speech enhancement model. The answer could be yes or no.",
        "Identify whether the given audio has been artificially produced or processed via a speech enhancement model. The answer could be yes or no.",
    ]
    # Instructions ONLY used in test.
    tt_instrs = [
        "Find out if the given audio is created or modified through a speech enhancement model. The answer could be yes or no.",
        "Detect if the given audio is artificially generated or manipulated by a speech enhancement model. The answer could be yes or no.",
        "Find out if the given audio is created synthetically or altered through a speech enhancement model. The answer could be yes or no.",
        "Detect if the given audio is produced by a speech enhancement model. The answer could be yes or no.",
    ]
    # The probability to sample an instruction from training instructions.
    # This variable is only used in 'get_instrs'.
    tr_prob = 0.7

    @classmethod
    def get_tr_instrs(cls) -> str:
        """Sample an instruction from tr_instrs."""
        return random.choice(cls.tr_instrs)

    @classmethod
    def get_tt_instrs(cls) -> str:
        """Sample an instruction from tt_instrs."""
        return random.choice(cls.tt_instrs)

    @classmethod
    def get_instrs(cls) -> str:
        if random.random() <= cls.tr_prob:
            return cls.get_tr_instrs()
        else:
            return cls.get_tt_instrs()


def main(data_dir: Path, seed: int, mode: str) -> None:
    random.seed(seed)
    if not (data_dir / "data").exists():
        raise FileNotFoundError
    rows = []
    # Add more attributes/columns here if needed.
    field_list = ["file_name", "file", "instruction", "label"]
    fields = ",".join(field_list)
    rows.append(fields)

    split_list = ["train", "validation"] if mode == "train" else ["test"]
    split_list = [data_dir / "data" / split for split in split_list]

    for split in split_list:
        for class_dir in split.iterdir():
            for file in tqdm(class_dir.iterdir()):
                # If you added more atrributes/columns, you should modify the below line.
                # Each element in dpr_list corresponds to the field defined in field_list.
                # E.g., file_name -> file.relative_to(data_dir), file -> file.name, instruction -> instr, label -> class_dir.name.
                if split.name == "train" or split.name == "validation":
                    instr = InstructionGenerator.get_tr_instrs()
                else:
                    instr = InstructionGenerator.get_instrs()
                dpr_list = [
                    str(file.relative_to(data_dir)),
                    file.name,
                    instr,
                    class_dir.name,
                ]
                dpr = ",".join(dpr_list)
                rows.append(dpr)
    with (data_dir / "metadata.csv").open(mode="w") as f:
        for row in rows:
            f.write(f"{row}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type=str, choices=["train", "test"], required=True)
    main(**vars(parser.parse_args()))
