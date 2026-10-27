import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_dir = "/workspace/OTOv2_v2/test/Bert/models/bert-base-uncased"  # 改成你的实际路径

tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForQuestionAnswering.from_pretrained(model_dir, local_files_only=True)
model.eval()

question = "Which framework is used to train and prune networks automatically?"
context = "HESSO is a framework that supports automatic efficient and user-friendly neural network training and pruning."

inputs = tokenizer(
    question,
    context,
    return_tensors="pt",
    truncation=True,
    max_length=384
)

with torch.no_grad():
    outputs = model(**inputs)

start_logits = outputs.start_logits[0]
end_logits = outputs.end_logits[0]

start_idx = int(torch.argmax(start_logits))
end_idx = int(torch.argmax(end_logits))

# 处理 end < start 的极端情况
if end_idx < start_idx:
    end_idx = start_idx

answer_ids = inputs["input_ids"][0][start_idx:end_idx + 1]
answer = tokenizer.decode(answer_ids, skip_special_tokens=True)

start_prob = torch.softmax(start_logits, dim=-1)[start_idx].item()
end_prob = torch.softmax(end_logits, dim=-1)[end_idx].item()

print("Pred answer:", answer)
print("Start idx / prob:", start_idx, start_prob)
print("End idx   / prob:", end_idx, end_prob)
