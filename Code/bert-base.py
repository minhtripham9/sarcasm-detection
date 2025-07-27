# Use a pipeline as a high-level helper
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification


pipe = pipeline("text-classification", model="helinivan/english-sarcasm-detector")



tokenizer = AutoTokenizer.from_pretrained("helinivan/english-sarcasm-detector")
model = AutoModelForSequenceClassification.from_pretrained("helinivan/english-sarcasm-detector")
context = "a kid is playing with a toy"
target = "wow, youre so smart"

input_text = f"Context: {context} [SEP] Target: {target}"

tokenized_text = tokenizer(input_text, padding=True, truncation=True, max_length=256, return_tensors="pt")
output = model(**tokenized_text)
probs = output.logits.softmax(dim=-1).tolist()[0]
confidence = max(probs)
prediction = probs.index(confidence)
results = {"is_sarcastic": prediction, "confidence": confidence}

print(results)


