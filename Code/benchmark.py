from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer
import torch

model_name = "microsoft/deberta-v3-large"
tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

context = "Today was a great day at work!"
target = "sure, today couldn't possibly get any better!"

input_text = f"Context: {context} [SEP] Target: {target}"
inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=-1)

label = torch.argmax(predictions).item()
confidence = torch.max(predictions).item()

print(f"Label: {label}, Confidence: {confidence:.3f}")