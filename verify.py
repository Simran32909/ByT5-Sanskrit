from datasets import load_from_disk

train_dataset = load_from_disk(r"C:\Users\simra\PycharmProjects\TextModels\dataset\train_hf")
dev_dataset = load_from_disk(r"C:\Users\simra\PycharmProjects\TextModels\dataset\dev_hf")
test_dataset = load_from_disk(r"C:\Users\simra\PycharmProjects\TextModels\dataset\test_hf")

print("Train Dataset:", train_dataset)
print("Dev Dataset:", dev_dataset)
print("Test Dataset:", test_dataset)

print(train_dataset[23451])