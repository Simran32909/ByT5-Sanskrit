from datasets import load_from_disk

train_dataset = load_from_disk(r"ByT5-Sanskrit\dataset\train_hf")
dev_dataset = load_from_disk(r"ByT5-Sanskrit\dataset\dev_hf")
test_dataset = load_from_disk(r"ByT5-Sanskrit\dataset\test_hf")

print("Train Dataset:", train_dataset)
print("Dev Dataset:", dev_dataset)
print("Test Dataset:", test_dataset)

print(train_dataset[...])