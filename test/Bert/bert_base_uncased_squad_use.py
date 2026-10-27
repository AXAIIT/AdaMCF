import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

MODEL_DIR = "/workspace/OTOv2_v2/test/Bert/models/bert-base-uncased-squad-v1"  # 改成你的本地路径
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_DIR, local_files_only=True)
model.to(device).eval()

question = "What is HESSO?"
context = "HESSO is a framework that supports automatic efficient and user-friendly neural network training and pruning."

inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=384)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

start = int(outputs.start_logits.argmax(dim=-1)[0])
end = int(outputs.end_logits.argmax(dim=-1)[0])
if end < start:
    end = start

ans_ids = inputs["input_ids"][0][start:end+1]
answer = tokenizer.decode(ans_ids, skip_special_tokens=True)

print("Answer:", answer)
