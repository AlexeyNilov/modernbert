import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch._dynamo

torch._dynamo.config.suppress_errors = True

model_id = "clapAI/modernBERT-base-multilingual-sentiment"
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

device = torch.device("cpu")
model.to(device)
model.eval()


# Retrieve labels from the model's configuration
id2label = model.config.id2label

texts = [
    # English
    {"text": "I absolutely love the new design of this app!", "label": "positive"},
    {"text": "The customer service was disappointing.", "label": "negative"},
]

for item in texts:
    text = item["text"]
    label = item["label"]

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)
    print(f"Text: {text} | Label: {label} | Prediction: {id2label[predictions.item()]}")
