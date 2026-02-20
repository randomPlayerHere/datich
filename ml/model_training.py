# %%
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# %%
df = pd.read_csv('./data/final/data_labelled.csv', engine='python')

# %%
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.1)

# %%
model_id = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["text"])

# %%
import torch
from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType

model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=6,
    problem_type="regression",
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model.config.pad_token_id = tokenizer.pad_token_id

# %%
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    modules_to_save=["score"]
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# %%
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import root_mean_squared_error

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.clip(predictions, 0.0, 1.0)
    rmse = root_mean_squared_error(labels, predictions)
    return {"rmse": rmse}

training_args = TrainingArguments(
    output_dir="./qwen_mental_health_scorer",
    eval_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    remove_unused_columns=False,
    bf16=True
)

# %%
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer, # <-- Update this line
    compute_metrics=compute_metrics
)

# %%
trainer.train()
model.save_pretrained("./final_qwen_lora")
tokenizer.save_pretrained("./final_qwen_lora")

# %%
import shutil
from google.colab import files

print("Zipping the model files...")
shutil.make_archive("final_qwen_lora", 'zip', "./final_qwen_lora")

print("Triggering download...")
files.download("final_qwen_lora.zip")


