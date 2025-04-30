# import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load model and tokenizer
model_name = "answerdotai/ModernBERT-Large-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = 'cpu'
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.to(device)


def predict(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model(**inputs)
    mask_idx = (inputs.input_ids == tokenizer.mask_token_id).nonzero()[0, 1]
    pred_id = outputs.logits[0, mask_idx].argmax()
    return tokenizer.decode(pred_id)


text = """You will be given a question and options. Select the right answer.
QUESTION: Wolf sees a car. What will the wolf do?
CHOICES:
- 1: attack
- 2: run away
- 3: eat
- 4: none of these
ANSWER: [unused0] [MASK]"""
answer = predict(text)
print(f"Predicted answer: {answer}")
