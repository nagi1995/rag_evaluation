import os
import pandas as pd
import json

def sample_and_save(json_path, output_dir, sample_sizes=[100, 1000, 10000]):
    # Load the full dataset
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for item in data:
        context = ' '.join([' '.join(p[1]) for p in item['context']])
        samples.append({
            "id": item.get("_id"),
            "question": item.get("question"),
            "answer": item.get("answer"),
            "type": item.get("type"),
            "level": item.get("level"),
            "supporting_facts": json.dumps(item.get("supporting_facts")),
            "context": context
        })
    df = pd.DataFrame(samples)

    # Compute proportions from the actual distribution
    level_proportions = df["level"].value_counts(normalize=True).to_dict()

    for sample_size in sample_sizes:
        # Perform proportional stratified sampling
        stratified_sample = pd.concat([
            df[df["level"] == level].sample(
                n=int(sample_size * prop),
                random_state=42
            )
            for level, prop in level_proportions.items()
        ])
        print(f"sample size: {sample_size}")
        print(stratified_sample["level"].value_counts()*100/stratified_sample.shape[0])
        stratified_sample.to_parquet(f"{output_dir}/sampled_hotpot_train_{sample_size}.parquet")

if __name__ == "__main__":
    # Change this if your dataset path changes
    INPUT_JSON = "data/raw/hotpot_train.json"
    OUTPUT_DIR = "data/sample"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    SAMPLE_SIZES = [100, 1000, 10000]

    sample_and_save(INPUT_JSON, OUTPUT_DIR, SAMPLE_SIZES)
