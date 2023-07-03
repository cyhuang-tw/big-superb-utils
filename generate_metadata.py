import argparse
import random
from pathlib import Path

from tqdm import tqdm


class InstructionGenerator:
    # Instructions used in train & validation (and probably test).
    tr_instrs = ["This is train instruction 0.", "This is train instruction 1."]
    # Instructions ONLY used in test.
    tt_instrs = ["This is test instruction 0.", "This is test instruction 1."]
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


def main(data_dir: Path, seed: int) -> None:
    random.seed(seed)
    if not (data_dir / "data").exists():
        raise FileNotFoundError
    rows = []
    # Add more attributes/columns here if needed.
    field_list = ["file_name", "file", "instruction", "label"]
    fields = ",".join(field_list)
    rows.append(fields)

    for split in (data_dir / "data").iterdir():
        for class_dir in split.iterdir():
            for file in tqdm(class_dir.iterdir()):
                # If you added more atrributes/columns, you should modify the below line.
                # Each element in dpr_list corresponds to the field defined in field_list.
                # E.g., file_name -> file.relative_to(data_dir), file -> file.name, instruction -> instr, label -> class_dir.name.
                if split == "train" or split == "validation":
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
    main(**vars(parser.parse_args()))
