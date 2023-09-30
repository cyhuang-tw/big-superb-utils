import argparse
import random
from pathlib import Path

from tqdm import tqdm


class InstructionGenerator:
    # The list of all instructions.
    instrs = ["instruction 1", "instruction 2", "instruction 3"]

    @classmethod
    def get_instrs(cls) -> str:
        return random.choice(cls.instrs)


def main(data_dir: Path, seed: int) -> None:
    random.seed(seed)
    if not (data_dir / "data").exists():
        raise FileNotFoundError
    rows = []
    # Add more attributes/columns here if needed.
    field_list = ["file_name", "file", "instruction", "label"]
    fields = ",".join(field_list)
    rows.append(fields)

    tt_dir = data_dir / "data" / "test"

    for class_dir in tt_dir.iterdir():
        for file in tqdm(class_dir.iterdir()):
            # If you added more atrributes/columns, you should modify the below line.
            # Each element in dpr_list corresponds to the field defined in field_list.
            # E.g., file_name -> file.relative_to(data_dir), file -> file.name, instruction -> instr, label -> class_dir.name.
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
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    main(**vars(parser.parse_args()))
