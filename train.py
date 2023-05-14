# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import MT5Tokenizer
import numpy as np
import string

# Read the data from CSV file
from datasets import load_dataset, load_metric

data = load_dataset("wisewizer/nepali-news", split="train")

data = data.train_test_split(test_size=0.1)

def sentence_tokenize(text):
    """This function tokenize the sentences
    
    Arguments:
        text {string} -- Sentences you want to tokenize
    
    Returns:
        sentence {list} -- tokenized sentence in list
    """
    sentences = text.strip().split(u"।")
    sentences = [sentence.translate(str.maketrans('', '', string.punctuation)) for sentence in sentences]
    return sentences

checkpoint = "google/mt5-small"
tokenizer = MT5Tokenizer.from_pretrained(checkpoint)

prefix = "summarize: "

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_data = data.map(preprocess_function, batched=True)

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

# import evaluate

# rouge = evaluate.load("rouge")

# import numpy as np

metric = load_metric("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(sentence_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sentence_tokenize(label.strip())) for label in decoded_labels]
    
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

training_args = Seq2SeqTrainingArguments(
    output_dir="nepali_sum_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5, #5e-5
    save_steps=500,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=20,
    predict_with_generate=True,
    push_to_hub=False,
    # load_best_model_at_end=True,
    # metric_for_best_model="rouge2",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()