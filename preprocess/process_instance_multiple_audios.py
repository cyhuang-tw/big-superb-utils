import argparse
import json
import soundfile as sf
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

SEED = 42


def main(json_path: Path, save_dir: Path, split: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    info = json.load(json_path.open(mode="r"))
    dataset = load_dataset(info["path"], revision=info["version"], split=split)
    dataset = dataset.shuffle(SEED)
    meta_data = defaultdict(dict)
    for example in tqdm(dataset):
        file_path = Path(example["file"])

        audio = example["audio"]["array"]
        sr = example["audio"]["sampling_rate"]
        task_prefix = save_dir.parent.name
        save_path = save_dir / f"{task_prefix}_{split}_{file_path.name}"
        sf.write(save_path, audio, sr)

        audio2 = example["audio2"]["array"]
        sr2 = example["audio2"]["sampling_rate"]
        file_name2 = f"{file_path.stem}_pair{file_path.suffix}"
        save_path2 = save_dir / f"{task_prefix}_{split}_{file_name2}"
        sf.write(save_path2, audio2, sr2)

        for key in example.keys():
            if key == "audio":
                continue
            if key == "audio2":
                continue
            meta_data[save_path.name][key] = example[key]

    with (save_dir / "metadata.json").open(mode="w") as f:
        json.dump(meta_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("save_dir", type=Path)
    parser.add_argument("split", type=Path)
    main(**vars(parser.parse_args()))
