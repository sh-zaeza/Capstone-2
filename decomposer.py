import spacy
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import os
import json
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

dataset = load_dataset('drop', split='train[:100%]')

nlp = spacy.load("en_core_web_sm")


class QuestionDecomposer:
    def __init__(self):
        pass  

    def decompose(self, question):
        sub_questions = re.split(r'\s+(?:and|but|or)\s+', question, flags=re.IGNORECASE)
        sub_questions = [q.strip() if q.strip().endswith('?') else q.strip() + '?' for q in sub_questions]
        return sub_questions

decomposer = QuestionDecomposer()
example_question = "What are the effects of gravity and how does it influence tides?"
decomposed_questions = decomposer.decompose(example_question)
print("Example Decomposition:", decomposed_questions)

def decompose_questions_in_dataset(dataset, decomposer):
    tokenized_dataset = []
    for example in dataset:
        question = example['question']
        sub_questions = decomposer.decompose(question)
        for sub_q in sub_questions:
            tokenized_dataset.append({
                "question": sub_q,
                "passage": example["passage"],
                "answers_spans": example.get("answers_spans", []),
            })
    return tokenized_dataset

decomposed_dataset = decompose_questions_in_dataset(dataset, decomposer)

model_name = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

def tokenize_function(example):
    inputs = tokenizer(
        example["question"], padding="max_length", truncation=True, max_length=128
    )
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

tokenized_dataset = [tokenize_function(example) for example in decomposed_dataset]

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=1000,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()

def answer_question(model, tokenizer, decomposer, question):
    sub_questions = decomposer.decompose(question)
    answers = []

    for sub_q in sub_questions:
        inputs = tokenizer(
            sub_q, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        input_ids = inputs["input_ids"].to(model.device)

        outputs = model.generate(input_ids, max_length=50)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answers.append(answer)

    return " ".join(answers)

model.to("cuda" if torch.cuda.is_available() else "cpu")
question = "What are the effects of gravity and how does it influence tides?"
print("Answer to question:", answer_question(model, tokenizer, decomposer, question))

# 假设 decomposed_dataset 已经准备好并包含需要保存的问题
output_file = "formatted_questions.json"

# 定义存储的结构列表
formatted_results = []

# 遍历数据集并提取所需字段
for idx, example in enumerate(decomposed_dataset):
    question = example['question']
    passage = example['passage']
    answers_spans = example.get('answers_spans', [])
    
    # 添加到结果列表中
    formatted_results.append({
        "question": question,
        "passage": passage,
        "answers_spans": answers_spans
    })
    
    # 输出当前进度
    print(f"Processed {idx + 1}/{len(decomposed_dataset)} questions")

    # 每处理 10 个问题保存一次
    if (idx + 1) % 10 == 0:
        with open(output_file, "w") as f:
            json.dump(formatted_results, f, ensure_ascii=False, indent=4)
        print(f"Progress saved to {output_file} after processing {idx + 1} questions")

# 保存最终结果到文件
with open(output_file, "w") as f:
    json.dump(formatted_results, f, ensure_ascii=False, indent=4)

print(f"所有数据已保存到 {output_file} 文件中。")