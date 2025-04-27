# import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load model and tokenizer
model_name = "answerdotai/ModernBERT-Large-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = 'cpu'
model = AutoModelForMaskedLM.from_pretrained(model_name)

model.to(device)

text = """You will be given a question and options. Select the right answer.
QUESTION: Wolf sees a pig. What will the wolf do?
CHOICES:
- A: Attack
- B: Run
- C: Eat
- D: None of these
ANSWER: [unused0] [MASK]"""

# Get prediction
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model(**inputs)
mask_idx = (inputs.input_ids == tokenizer.mask_token_id).nonzero()[0, 1]
pred_id = outputs.logits[0, mask_idx].argmax()
answer = tokenizer.decode(pred_id)
print(f"Predicted answer: {answer}")
