import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json
import torch
import random
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

# Load formatted_questions.json data
with open("formatted_questions.json", "r") as f:
    formatted_dataset = json.load(f)

# Reduce dataset size to 10%
random.seed(42)
formatted_dataset = random.sample(formatted_dataset, int(len(formatted_dataset) * 1))
print(f"Reduced dataset size: {len(formatted_dataset)}")

# Split data
train_data, test_valid_data = train_test_split(formatted_dataset, test_size=0.3, random_state=42)
valid_data, test_data = train_test_split(test_valid_data, test_size=1/3, random_state=42)

print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(valid_data)}")
print(f"Test set size: {len(test_data)}")

# Convert to HuggingFace Dataset format
train_dataset = Dataset.from_list(train_data)
valid_dataset = Dataset.from_list(valid_data)
test_dataset = Dataset.from_list(test_data)

# Load model and tokenizer
model_name = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Add pad_token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Normalize answer for consistent comparison
def normalize_answer(answer):
    """Normalize text by removing punctuation, articles, and extra whitespace."""
    answer = answer.lower()
    answer = re.sub(r'\b(a|an|the)\b', ' ', answer)  # Remove articles
    answer = re.sub(r'[^\w\s]', '', answer)  # Remove punctuation
    answer = re.sub(r'\s+', ' ', answer).strip()  # Remove extra spaces
    return answer

# Data preprocessing function
def preprocess_function(examples, tokenizer):
    questions = examples["question"]
    passages = examples["passage"]
    answers_list = examples["answers_spans"]

    inputs = []
    for question, passage, answers in zip(questions, passages, answers_list):
        if not question.strip() or not passage.strip():
            continue
        answer = answers["spans"][0] if answers and "spans" in answers and answers["spans"] else "No answer"
        
        # Improved prompt design
        prompt = (
            f"You are an expert in extracting information. Based on the provided context, "
            f"answer the question as accurately and concisely as possible.\n\n"
            f"Context:\n{passage}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer:"
        )
        full_text = prompt + " " + answer + tokenizer.eos_token
        inputs.append(full_text)

    # Tokenization
    model_inputs = tokenizer(
        inputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt"
    )

    # Create labels, supervising only the answer part
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
            labels[i] = -100
        else:
            labels[i][:answer_start] = -100

    model_inputs["labels"] = labels
    return model_inputs

# Tokenize training and validation datasets
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

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=50,
    fp16=torch.cuda.is_available(),
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Define Solver class for testing
class Solver:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def solve(self, question, passage):
        if not question.strip() or not passage.strip():
            return "Invalid input"

        # Improved prompt
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
                max_new_tokens=150,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                early_stopping=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                num_beams=5  # Improved beam search
            )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text[len(prompt):].strip()
        return answer if answer else "No answer"

# Initialize Solver
solver = Solver(model, tokenizer)

# Generate answers for the test set
def generate_answers(test_dataset, solver):
    solved_dataset = []
    for example in test_dataset:
        question = example["question"]
        passage = example["passage"]
        solved_answer = solver.solve(question, passage)
        solved_dataset.append({
            "question": question,
            "passage": passage,
            "solved_answer": solved_answer
        })
    return solved_dataset

solved_test_dataset = generate_answers(test_dataset, solver)

# Save test set results
output_file = "test_solved_answers.json"
with open(output_file, "w") as f:
    json.dump(solved_test_dataset, f, ensure_ascii=False, indent=4)

print(f"All test set answers have been saved to {output_file}.")

# Evaluate results
predictions = [normalize_answer(item["solved_answer"]) for item in solved_test_dataset]
references = [normalize_answer(item.get("answers_spans", {}).get("spans", ["No answer"])[0]) for item in test_data]

# Compute scores
def compute_exact_match(predictions, references):
    return np.mean([1 if pred.strip() == ref.strip() else 0 for pred, ref in zip(predictions, references)])

def compute_f1(predictions, references):
    f1_scores = []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.split())
        ref_tokens = set(ref.split())
        common = pred_tokens & ref_tokens
        if not common:
            f1_scores.append(0)
        else:
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(ref_tokens)
            f1 = 2 * (precision * recall) / (precision + recall)
            f1_scores.append(f1)
    return np.mean(f1_scores)

em_score = compute_exact_match(predictions, references)
f1_score_value = compute_f1(predictions, references)

print("Evaluation Results:")
print(f"Exact Match (EM): {em_score:.4f}")
print(f"F1 Score: {f1_score_value:.4f}")
