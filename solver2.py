import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json
import torch
import random
import ast
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
import numpy as np
import re

# 加载数据
with open("formatted_questions.json", "r", encoding="utf-8") as f:
    formatted_dataset = json.load(f)

# 降低数据集规模到10%
random.seed(42)
sample_ratio = 1  # 从1%提升到10%
formatted_dataset = random.sample(formatted_dataset, int(len(formatted_dataset) * sample_ratio))
print(f"Reduced dataset size: {len(formatted_dataset)}")

# 数据集拆分（训练集70%，验证集15%，测试集15%）
train_data, test_valid_data = train_test_split(formatted_dataset, test_size=0.3, random_state=42)

# 移除重叠的测试集样本
train_questions = set(item['question'] for item in train_data)
test_valid_data = [item for item in test_valid_data if item['question'] not in train_questions]
print(f"After removing overlaps, test_valid_data size: {len(test_valid_data)}")

# 重新拆分验证集和测试集
valid_data, test_data = train_test_split(test_valid_data, test_size=0.5, random_state=42)

print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(valid_data)}")
print(f"Test set size: {len(test_data)}")

# 确认没有重叠
train_questions = set(item['question'] for item in train_data)
test_questions = set(item['question'] for item in test_data)
overlap = train_questions.intersection(test_questions)
print(f"Number of overlapping questions between train and test sets: {len(overlap)}")

# 转换为 HuggingFace Dataset 格式
train_dataset = Dataset.from_list(train_data)
valid_dataset = Dataset.from_list(valid_data)
test_dataset = Dataset.from_list(test_data)

# 加载模型和分词器
model_name = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# 添加 pad_token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 答案标准化函数
def normalize_answer(answer):
    """标准化答案，通过移除标点符号、冠词和多余的空白"""
    answer = answer.lower()
    answer = re.sub(r'\b(a|an|the)\b', ' ', answer)  # 移除冠词
    answer = re.sub(r'[^\w\s]', '', answer)  # 移除标点符号
    answer = re.sub(r'\s+', ' ', answer).strip()  # 移除多余空白
    return answer

# 数据预处理函数
def preprocess_function(examples, tokenizer):
    questions = examples["question"]
    passages = examples["passage"]
    answers_list = examples["answers_spans"]

    inputs = []
    for question, passage, answers_str in zip(questions, passages, answers_list):
        if not question.strip() or not passage.strip():
            continue
        try:
            # 清理 answers_spans 字符串
            cleaned_answers_str = answers_str.replace("array([", "[").replace("], dtype=object)", "]")
            # 解析清理后的字符串为字典
            answers = ast.literal_eval(cleaned_answers_str)
        except (ValueError, SyntaxError):
            answers = {}
        
        # 获取第一个答案span，如果不存在则设为 "No answer"
        answer = answers["spans"][0] if answers and "spans" in answers and answers["spans"] else "No answer"

        # 构建提示
        prompt = (
            f"You are an expert in extracting information. Based on the provided context, "
            f"answer the question as accurately and concisely as possible.\n\n"
            f"Context:\n{passage}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer:"
        )
        full_text = prompt + " " + answer + tokenizer.eos_token
        inputs.append(full_text)

    # 分词
    model_inputs = tokenizer(
        inputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt"
    )

    # 创建标签，仅监督答案部分
    labels = model_inputs["input_ids"].clone()
    answer_token_ids = tokenizer.encode("Answer:", add_special_tokens=False)
    answer_token_length = len(answer_token_ids)

    for i in range(len(labels)):
        input_ids = model_inputs["input_ids"][i]
        answer_start = None
        for idx in range(len(input_ids) - answer_token_length + 1):
            if torch.equal(input_ids[idx:idx+answer_token_length], torch.tensor(answer_token_ids)):
                answer_start = idx + answer_token_length
                break
        if answer_start is None:
            labels[i] = -100  # 忽略整个序列
        else:
            labels[i][:answer_start] = -100  # 仅监督答案部分

    model_inputs["labels"] = labels
    return model_inputs

# 分词训练和验证数据集
tokenized_train_dataset = train_dataset.map(
    lambda examples: preprocess_function(examples, tokenizer),
    batched=True,
    remove_columns=train_dataset.column_names
)

tokenized_valid_dataset = valid_dataset.map(
    lambda examples: preprocess_function(examples, tokenizer),
    batched=True,
    remove_columns=valid_dataset.column_names
)

# 数据整理器
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,  # 调整学习率
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,  # 减少训练轮数以防止过拟合
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=50,
    fp16=torch.cuda.is_available(),
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 开始训练
trainer.train()

# 定义 Solver 类用于测试
class Solver:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def solve(self, question, passage):
        if not question.strip() or not passage.strip():
            return "Invalid input"

        # 构建提示
        prompt = (
            f"You are an expert in extracting information. Based on the provided context, "
            f"answer the question as accurately and concisely as possible.\n\n"
            f"Context:\n{passage}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer:"
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=150,  # 限制生成的token数量
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                early_stopping=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                num_beams=5  # 使用beam search
            )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 提取 "Answer:" 后的内容
        answer_start = generated_text.find("Answer:")
        if answer_start != -1:
            answer = generated_text[answer_start + len("Answer:"):].strip()
        else:
            # 如果未找到 "Answer:"，则从提示后提取
            answer = generated_text[len(prompt):].strip()
        return answer if answer else "No answer"

# 初始化 Solver
solver = Solver(model, tokenizer)

# 生成测试集的答案
def generate_answers(test_dataset, solver):
    solved_dataset = []
    for example in test_dataset:
        question = example["question"]
        passage = example["passage"]
        solved_answer = solver.solve(question, passage)
        answers_spans = example["answers_spans"]
        solved_dataset.append({
            "question": question,
            "passage": passage,
            "solved_answer": solved_answer,
            "answers_spans": answers_spans  # 保留 answers_spans 以便评估
        })
    return solved_dataset

solved_test_dataset = generate_answers(test_dataset, solver)

# 保存测试集结果
output_file = "test_solved_answers.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(solved_test_dataset, f, ensure_ascii=False, indent=4)

print(f"All test set answers have been saved to {output_file}.")

# 评估结果
predictions = [normalize_answer(item["solved_answer"]) for item in solved_test_dataset]
references = []
for item in test_data:
    answers_spans_str = item.get("answers_spans", "{}")
    try:
        # 清理 answers_spans 字符串
        cleaned_answers_str = answers_spans_str.replace("array([", "[").replace("], dtype=object)", "]")
        answers_spans = ast.literal_eval(cleaned_answers_str)
    except (ValueError, SyntaxError):
        answers_spans = {}
    answer = answers_spans["spans"][0] if "spans" in answers_spans and answers_spans["spans"] else "No answer"
    references.append(normalize_answer(answer))

# 统计参考答案和预测答案
no_answer_count = references.count("no answer")
actual_answer_count = len(references) - no_answer_count

pred_no_answer_count = predictions.count("no answer")
pred_actual_answer_count = len(predictions) - pred_no_answer_count

print(f"\nNumber of 'no answer' in references: {no_answer_count}")
print(f"Number of actual answers in references: {actual_answer_count}")
print(f"Number of 'no answer' in predictions: {pred_no_answer_count}")
print(f"Number of actual answers in predictions: {pred_actual_answer_count}")

# 打印有实际答案的预测和参考答案
print("\nSample Predictions vs References (Only for samples with actual answers):")
for i, item in enumerate(test_data):
    reference = references[i]
    prediction = predictions[i]
    if reference != "no answer":
        print(f"Test Sample {i}:")
        print(f"  Question: {item['question']}")
        print(f"  Reference Answer: {reference}")
        print(f"  Predicted Answer: {prediction}")
        print("-" * 50)

# 计算分数
def compute_exact_match(predictions, references):
    return np.mean([1 if pred.strip() == ref.strip() else 0 for pred, ref in zip(predictions, references)])

def compute_f1(predictions, references):
    f1_scores = []
    for pred, ref in zip(predictions, references):
        if ref == "no answer" and pred == "no answer":
            f1_scores.append(1.0)
            continue
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        common = set(pred_tokens) & set(ref_tokens)
        if not common:
            f1_scores.append(0)
            continue
        precision = len(common) / len(pred_tokens) if len(pred_tokens) > 0 else 0
        recall = len(common) / len(ref_tokens) if len(ref_tokens) > 0 else 0
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1)
    return np.mean(f1_scores)

em_score = compute_exact_match(predictions, references)
f1_score_value = compute_f1(predictions, references)

print("\nEvaluation Results:")
print(f"Exact Match (EM): {em_score:.4f}")
print(f"F1 Score: {f1_score_value:.4f}")
