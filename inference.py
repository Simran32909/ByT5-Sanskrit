from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def load_model(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def test_model(text, model, tokenizer):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    return outputs

# Usage
model_path = r"ByT5-Sanskrit\models"
model, tokenizer = load_model(model_path)

text = "एतदिच्छाम्यहं श्रोतुं परं कौतूहलं हि मे। महर्षे त्वं समर्थोऽसि ज्ञातुमेवंविधं नरम्॥"
outputs = test_model(text, model, tokenizer)

print(outputs)