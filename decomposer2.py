import spacy
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import os
import json
import re

# 配置 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 加载本地 CSV 数据集
dataset = load_dataset('csv', data_files='/data2/student/temp_1732438271819.425073248.csv', split='train[:100%]')

# 加载 SpaCy
nlp = spacy.load("en_core_web_sm")

# 定义问题分解类
class QuestionDecomposer:
    def __init__(self):
        pass  

    def decompose(self, question):
        # 使用正则表达式分解问题
        sub_questions = re.split(r'\s+(?:and|but|or)\s+', question, flags=re.IGNORECASE)
        sub_questions = [q.strip() if q.strip().endswith('?') else q.strip() + '?' for q in sub_questions]
        return sub_questions

# 初始化分解器
decomposer = QuestionDecomposer()
example_question = "What are the effects of gravity and how does it influence tides?"
decomposed_questions = decomposer.decompose(example_question)
print("Example Decomposition:", decomposed_questions)

# 将数据集中问题分解
def decompose_questions_in_dataset(dataset, decomposer):
    tokenized_dataset = []
    for example in dataset:
        question = example['question']  # 替换为 CSV 中的问题字段
        passage = example['context'] if 'context' in example else example['passage']  # 替换为 CSV 中的上下文字段
        answers_spans = example.get('answers_spans', [])  # 替换为 CSV 中的答案字段（可选）

        sub_questions = decomposer.decompose(question)
        for sub_q in sub_questions:
            tokenized_dataset.append({
                "question": sub_q,
                "passage": passage,
                "answers_spans": answers_spans,
            })
    return tokenized_dataset

decomposed_dataset = decompose_questions_in_dataset(dataset, decomposer)

# 加载模型和分词器
model_name = "facebook/opt-350m"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
data_collator = DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

# 数据集分词函数
def tokenize_function(example):
    inputs = tokenizer(
        example["question"], padding="max_length", truncation=True, max_length=128
    )
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

# 对数据集进行分词
tokenized_dataset = [tokenize_function(example) for example in decomposed_dataset]

# 设置训练参数
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

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# 开始训练
trainer.train()

# 定义问答函数
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

# 生成答案
model.to("cuda" if torch.cuda.is_available() else "cpu")
question = "What are the effects of gravity and how does it influence tides?"
print("Answer to question:", answer_question(model, tokenizer, decomposer, question))

# 将处理后的数据集保存为 JSON
output_file = "formatted_questions.json"

formatted_results = []
for idx, example in enumerate(decomposed_dataset):
    question = example['question']
    passage = example['passage']
    answers_spans = example.get('answers_spans', [])

    formatted_results.append({
        "question": question,
        "passage": passage,
        "answers_spans": answers_spans
    })
    
    # 输出进度
    print(f"Processed {idx + 1}/{len(decomposed_dataset)} questions")

    # 每处理 10 个问题保存一次
    if (idx + 1) % 10 == 0:
        with open(output_file, "w") as f:
            json.dump(formatted_results, f, ensure_ascii=False, indent=4)
        print(f"Progress saved to {output_file} after processing {idx + 1} questions")

# 保存最终结果
with open(output_file, "w") as f:
    json.dump(formatted_results, f, ensure_ascii=False, indent=4)

print(f"所有数据已保存到 {output_file} 文件中。")
