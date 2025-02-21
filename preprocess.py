import os
from datasets import Dataset
def load_data(sanskrit_path, english_path):
    print(f"Loading Sanskrit file from: {sanskrit_path}")
    print(f"Loading English file from: {english_path}")
    try:
        with open(sanskrit_path, "r", encoding="utf-8") as sn_file, \
                open(english_path, "r", encoding="utf-8") as en_file:
            sanskrit_lines = sn_file.readlines()
            english_lines = en_file.readlines()

        data = [{"input_text": sn.strip(), "target_text": en.strip()} for sn, en in zip(sanskrit_lines, english_lines)]
        return Dataset.from_list(data)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

for split in ["train", "dev", "test"]:
    sanskrit_path = f"ByT5-Sanskrit/dataset/{split}-sn.json"
    english_path = f"ByT5-Sanskrit/dataset/{split}-en.json"
    dataset = load_data(sanskrit_path, english_path)

    if dataset is not None:
        print(f"Saving {split} dataset.")
        dataset.save_to_disk(f"ByT5-Sanskrit/dataset/{split}_hf")
    else:
        print(f"Failed to load {split} dataset.")