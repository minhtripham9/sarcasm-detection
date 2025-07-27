from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import csv
import os

# Example: Change this to any supported model from Hugging Face
model_name = "distilbert-base-uncased"  

# Load tokenizer and model, change tokenizer depending on the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Example input (with or without context)
context = "politics: Female Voters Flock from Hillary Clinton to Bernie Sanders"
target = "That means she'll lose men too.... Because the guys are only supporting Clinton to meet girls."

# You can skip the context if you want a context-free test
input_text = f"Context: {context} [SEP] Target: {target}"

inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    label = torch.argmax(probs).item()
    confidence = torch.max(probs).item()

# CSV filename
csv_filename = "sarcasm_predictions.csv"

# Check if file exists to write header only once
write_header = not os.path.isfile(csv_filename)

with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    if write_header:
        writer.writerow(["context", "text", "label_sarcastic", "confidence"])
    writer.writerow([context, target, label, confidence])

print(f"Label: {label}, Confidence: {confidence:.3f} (saved to {csv_filename})")