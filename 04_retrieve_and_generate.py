from transformers import pipeline
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv
import os
import pandas as pd
pd.set_option('display.max_columns', None)
import chromadb
from langchain.prompts import PromptTemplate
from datasets import Dataset


# load environ variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")


prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use the following retrieved context to answer the question.
    If you don't know, say "I don't know." Keep it precise and concise.
    {context}

    Question: {question}
    Answer:"""
)


# Batch generation function
def generate_llm_responses_batch(questions, embedding_model, collection, generator, k=3):
    # Embed all questions at once
    query_embeddings = embedding_model.encode(questions, show_progress_bar=False)

    # Retrieve relevant documents for each question
    contexts = []
    for query_embedding in query_embeddings:
        results = collection.query(query_embeddings=[query_embedding], n_results=k)
        context = " ".join(results['documents'][0]) if results['documents'][0] else ""
        contexts.append(context)

    # Prepare prompts
    prompts = [
        prompt_template.format(context=contexts[i], question=questions[i])
        for i in range(len(questions))
    ]

    # Batch generate responses
    outputs = generator(prompts, max_length=300, truncation=True)
    responses = [output["generated_text"].strip() for output in outputs]
    return responses


# Parameters
models = ["google/flan-t5-base", "google/flan-t5-large", "t5-base", "t5-large"]
sample_sizes = [100, 1000, 10000]
embedding_models = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet": "sentence-transformers/all-mpnet-base-v2",
    "E5-small": "intfloat/e5-small-v2",
    "E5-base": "intfloat/e5-base-v2"
}

# Main loop with batching
for sample_size in tqdm(sample_sizes):
    stratified_sample = pd.read_parquet(f"data/sample/sampled_hotpot_train_{sample_size}.parquet")

    for model in models:
        generator = pipeline("text2text-generation", model=model)

        for label in embedding_models.keys():
            print(f"llm model: {model}, sample size: {sample_size}, embedding model key: {label}")
            save_folder = f"data/outputs/sample={sample_size}/label={label}/model={model.replace('/', '_')}"
            save_path = f"{save_folder}/out.parquet"
            print(f"save path: {save_path}")

            if os.path.exists(save_path):
                print("Output already exists. Skipping...")
                continue

            embeddings_save_path = f"data/embeddings/{sample_size}"
            persist_dir = os.path.join(embeddings_save_path, label)
            print(f"loading embeddings from {persist_dir}")

            chroma_client = chromadb.PersistentClient(path=persist_dir)
            collection = chroma_client.get_or_create_collection(name="hotpotqa_chunks")
            embedding_model = SentenceTransformer(embedding_models[label])

            # Batch processing using datasets
            dataset = Dataset.from_pandas(stratified_sample)
            batched_data = dataset.map(
                lambda batch: {
                    "llm_response": generate_llm_responses_batch(
                        batch["question"], embedding_model, collection, generator, k=3
                    )
                },
                batched=True,
                batch_size=32
            )

            os.makedirs(save_folder, exist_ok=True)
            batched_data.select_columns(["id", "llm_response"]).to_parquet(save_path)


