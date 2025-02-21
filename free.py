import os
from datasets import load_from_disk
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer,
                          EarlyStoppingCallback, TrainerCallback)
import torch

def load_datasets(train_path, dev_path, test_path):
    train_dataset = load_from_disk(train_path)
    dev_dataset = load_from_disk(dev_path)
    test_dataset = load_from_disk(test_path)
    return train_dataset, dev_dataset, test_dataset

def preprocess_function(examples, tokenizer, max_length=512):
    model_inputs = tokenizer(examples['input_text'], truncation=True, padding="max_length", max_length=max_length)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['target_text'], truncation=True, padding="max_length", max_length=max_length)

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def setup_model(model_name="google/byt5-small"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

def get_training_arguments(output_dir, max_steps=2000, eval_metric="bleu"):
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        max_steps=max_steps,
        fp16=True,
        save_total_limit=2,
        learning_rate=2e-5,
        dataloader_num_workers=8,
        logging_steps=100,
        save_steps=500,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model=eval_metric,
        max_grad_norm=1.0,
        greater_is_better=True,
        seed=42,
        gradient_checkpointing=True,
    )
    return training_args

def create_trainer(model, tokenizer, train_dataset, dev_dataset, training_args, eval_metric="bleu"):
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2, metric=eval_metric, greater_is_better=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        callbacks=[early_stopping_callback]
    )
    return trainer

def train_model(train_path, dev_path, test_path, output_dir="models"):
    train_dataset, dev_dataset, test_dataset = load_datasets(train_path, dev_path, test_path)

    tokenizer, model = setup_model()

    train_dataset = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    dev_dataset = dev_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    training_args = get_training_arguments(output_dir)

    trainer = create_trainer(model, tokenizer, train_dataset, dev_dataset, training_args)

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    evaluation_results = trainer.evaluate(test_dataset)
    print(f"Evaluation Results: {evaluation_results}")

def setup_tensorboard_logging():
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir="runs/byt5_training")
    return writer

if __name__ == "__main__":
    train_path = '/content/train_hf'
    dev_path = '/content/dev_hf'
    test_path = '/content/test_hf'

    writer = setup_tensorboard_logging()

    train_model(train_path, dev_path, test_path)