
from tqdm import tqdm
from dotenv import load_dotenv
import os
import pandas as pd
pd.set_option('display.max_columns', None)
import evaluate
from collections import Counter
import re
import string
import unicodedata
import numpy as np 


# load environ variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# Normalize text for EM/F1
def normalize_text(text):
    def remove_articles(s):
        return re.sub(r'\b(a|an|the)\b', ' ', s)

    def white_space_fix(s):
        return ' '.join(s.split())

    def remove_punc(s):
        return ''.join(ch for ch in s if ch not in string.punctuation)

    def lower(s):
        return s.lower()

    def unicode_to_ascii(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    return white_space_fix(remove_articles(remove_punc(lower(unicode_to_ascii(text)))))

# Exact Match
def exact_match_score(prediction, ground_truth):
    return int(normalize_text(prediction) == normalize_text(ground_truth))

# F1 Score
def f1_score(prediction, ground_truth):
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

# Load Hugging Face metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")


def get_metrics(df):

    answers = df['answer'].fillna("").astype(str).tolist()
    predictions = df['llm_response'].fillna("").astype(str).tolist()

    em_scores = [exact_match_score(a, p) for a, p in zip(answers, predictions)]
    f1_scores = [f1_score(a, p) for a, p in zip(answers, predictions)]

    # BLEU
    bleu_result = bleu.compute(predictions=predictions, references=[[a] for a in answers])
    # ROUGE
    rouge_result = rouge.compute(predictions=predictions, references=answers)

    return em_scores, f1_scores, bleu_result, rouge_result


# Parameters
models = ["google/flan-t5-large", "google/flan-t5-base", "t5-base", "t5-large"]
sample_sizes = [100, 1000, 10000]
embedding_models = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet": "sentence-transformers/all-mpnet-base-v2",
    "E5-small": "intfloat/e5-small-v2",
    "E5-base": "intfloat/e5-base-v2"
}
results = []
# Main loop with batching
for sample_size in tqdm(sample_sizes):
    stratified_sample = pd.read_parquet(f"data/sample/sampled_hotpot_train_{sample_size}.parquet")
    for model in models:
        for label in embedding_models.keys():
            print(f"sample size: {sample_size}, embedding model key: {label}, llm model: {model}")
            save_folder = f"data/outputs/sample={sample_size}/label={label}/model={model.replace('/', '_')}"
            save_path = f"{save_folder}/out.parquet"
            try:
                save_df = pd.read_parquet(save_path)
                df = pd.merge(stratified_sample, save_df, on='id', how='inner')
                em_scores, f1_scores, bleu_result, rouge_result = get_metrics(df)
                result = {
                    "sample_size": sample_size, 
                    "model": model, 
                    "label": label, 
                    "level": "all", 
                    "exact_match": np.mean(em_scores), 
                    "f1": np.mean(f1_scores), 
                    "bleu": bleu_result["bleu"], 
                    "rouge1": rouge_result["rouge1"],
                    "rouge2": rouge_result["rouge2"],
                    "rougeL": rouge_result["rougeL"],
                    "rougeLsum": rouge_result["rougeLsum"]
                }
                results.append(result)

                for level in df["level"].unique():
                    level_df = df[df["level"] == level]
                    em_scores, f1_scores, bleu_result, rouge_result = get_metrics(level_df)
                    result = {
                        "sample_size": sample_size, 
                        "model": model, 
                        "label": label, 
                        "level": level, 
                        "exact_match": np.mean(em_scores), 
                        "f1": np.mean(f1_scores), 
                        "bleu": bleu_result["bleu"], 
                        "rouge1": rouge_result["rouge1"],
                        "rouge2": rouge_result["rouge2"],
                        "rougeL": rouge_result["rougeL"],
                        "rougeLsum": rouge_result["rougeLsum"]
                    }
                    results.append(result)
            except Exception as e:
                pass

results_df = pd.DataFrame(results)
results_df.head()
os.makedirs("data/evaluation_results", exist_ok=True)
results_df.to_excel(f"data/evaluation_results/results.xlsx", index = None)
