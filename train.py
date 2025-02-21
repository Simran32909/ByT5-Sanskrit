from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments

train_dataset = load_from_disk(r'/dataset/train_hf')
dev_dataset = load_from_disk(r'/dataset/test_hf')
test_dataset = load_from_disk(r'/dataset/dev_hf')

print("Train dataset columns:", train_dataset.column_names)
print("Dev dataset columns:", dev_dataset.column_names)

model_name = "google/byt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def preprocess_function(examples):
    model_inputs = tokenizer(examples['input_text'], truncation=True, padding="max_length", max_length=512)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['target_text'], truncation=True, padding="max_length", max_length=512)

    model_inputs['labels'] = labels['input_ids']

    return model_inputs


train_dataset = train_dataset.map(preprocess_function, batched=True)
dev_dataset = dev_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=r"ByT5-Sanskrit\models",
    evaluation_strategy="epochs",
    save_strategy="epochs",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    max_steps=2000,
    fp16=True,
    save_total_limit=2,
    learning_rate=2e-5,
    dataloader_num_workers=4,
    logging_steps=100,
    save_steps=500,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    max_grad_norm=1.0,
    greater_is_better=True,
    seed=42,
    gradient_checkpointing=True,
)


if __name__ == "__main__":
    from transformers import Seq2SeqTrainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer
    )
    trainer.train()