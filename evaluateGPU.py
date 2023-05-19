# -*- coding: utf-8 -*-
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
from rouge import Rouge 
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
import matplotlib.pyplot as plt
# from wordcloud import WordCloud
from stemmer import Stem

# Check if CUDA is available and set device to GPU if it is, else use CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Handler function
WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

# Load the model
model_name = "text_summarizer/nepali_sum_model/finetuned-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Move the model to GPU
model = model.to(device)

class SummaryDataset(Dataset):
    def __init__(self, articles, tokenizer, max_len=512):
        self.articles = articles
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        article = WHITESPACE_HANDLER(self.articles[idx])
        inputs = self.tokenizer(article, return_tensors='pt', max_length=self.max_len, padding='max_length', truncation=True)
        
        # Move inputs to GPU
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        return inputs

# Load the test data
test_data = pd.read_csv('testdata.csv', encoding='utf-8')
dataset = SummaryDataset(test_data['text'].tolist(), tokenizer)
data_loader = DataLoader(dataset, batch_size=12)  # Adjust batch_size as needed

# Generate summaries and calculate Rouge and BLEU scores
rouge = Rouge()
scores = {'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu': []}
smoothing = SmoothingFunction().method1

generated_summaries = []

for batch_idx, batch in enumerate(data_loader):
    input_ids = batch['input_ids'].squeeze()
    print(input_ids.shape)
    output_ids = model.generate(
        input_ids=input_ids,
        max_length=128,
        no_repeat_ngram_size=2,
        num_beams=4
    )

    for idx, output_id in enumerate(output_ids):
        generated_summary = tokenizer.decode(
            output_id,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        generated_summaries.append(generated_summary)

        generated_summary = Stem().rootify(generated_summary)

        # Correct the indexing for the expected summary
        expected_summary = test_data['summary'].iloc[batch_idx * data_loader.batch_size + idx]

        expected_summary = Stem().rootify(expected_summary)

       # Calculate the Rouge and BLEU scores for the summary
        rouge_scores = rouge.get_scores(generated_summary, expected_summary)[0]
        bleu_score = sentence_bleu([expected_summary.split()], generated_summary.split(), smoothing_function=smoothing)

        # Rouge scores
        scores['rouge-1'].append(rouge_scores['rouge-1']['f'])
        scores['rouge-2'].append(rouge_scores['rouge-2']['f'])
        scores['rouge-l'].append(rouge_scores['rouge-l']['f'])
        scores['bleu'].append(bleu_score)

avg_rouge_1 = sum(scores['rouge-1']) / len(scores['rouge-1'])
avg_rouge_2 = sum(scores['rouge-2']) / len(scores['rouge-2'])
avg_rouge_l = sum(scores['rouge-l']) / len(scores['rouge-l'])
avg_bleu = sum(scores['bleu']) / len(scores['bleu'])

print(f"Average ROUGE-1 score: {avg_rouge_1}")
print(f"Average ROUGE-2 score: {avg_rouge_2}")
print(f"Average ROUGE-L score: {avg_rouge_l}")
print(f"Average BLEU score: {avg_bleu}")

# Visualization 1: Histogram of Scores
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.hist(scores['rouge-1'], bins=20, alpha=0.5, label='ROUGE-1', color='r')
plt.hist(scores['rouge-2'], bins=20, alpha=0.5, label='ROUGE-2', color='g')
plt.hist(scores['rouge-l'], bins=20, alpha=0.5, label='ROUGE-L', color='b')
plt.legend(loc='upper right')
plt.title('Histogram of ROUGE Scores')
plt.xlabel('Score')

plt.subplot(1, 2, 2)
plt.hist(scores['bleu'], bins=20, alpha=0.5, color='b')
plt.title('Histogram of BLEU Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('histogram_scores.png')
plt.show()


# Visualization 2: Bar Chart of Average Scores
average_scores = {metric: sum(values)/len(values) for metric, values in scores.items()}
plt.bar(range(len(average_scores)), list(average_scores.values()), align='center', alpha=0.5)
plt.xticks(range(len(average_scores)), list(average_scores.keys()))
plt.ylabel('Average Score')
plt.title('Average ROUGE and BLEU Scores')
plt.savefig('average_scores.png')
plt.show()


# Visualization 3: Scatter Plot of ROUGE-1 vs BLEU Scores
plt.scatter(scores['rouge-1'], scores['bleu'], alpha=0.5)
plt.title('Scatter Plot of ROUGE-1 vs BLEU Scores')
plt.xlabel('ROUGE-1 Score')
plt.ylabel('BLEU Score')
plt.savefig('scatterplot_rouge_bleu.png')
plt.show()

df = pd.DataFrame({'Actual_summary': test_data['summary'],'Generated_summary': generated_summaries})
df.to_csv('generated_summaries.csv')