import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from more_itertools import chunked
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# load environ variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

MAX_CHROMA_BATCH = 5000

# Define embedding models
embedding_models = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet": "sentence-transformers/all-mpnet-base-v2",
    "E5-small": "intfloat/e5-small-v2",
    "E5-base": "intfloat/e5-base-v2"
}

# Overwrite existing embeddings if True
overwrite = False


def _generate_embeddings(stratified_sample, embeddings_save_path):
	text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=50)

	documents = []
	metadatas = []

	for _, row in stratified_sample.iterrows():
	    chunks = text_splitter.split_text(row["context"])
	    for i, chunk in enumerate(chunks):
	        documents.append(chunk)
	        metadatas.append({
	            "id": row["id"],
	            "question": row["question"],
	            "answer": row["answer"],
	            "chunk_index": i
	        })


	for label, model_name in tqdm(embedding_models.items()):
	    persist_dir = os.path.join(embeddings_save_path, label)
	    sqlite_path = os.path.join(persist_dir, "chroma.sqlite3")

	    if os.path.exists(sqlite_path):
	        if overwrite:
	            print(f"🧹 Overwriting {label} — removing old files...")
	            for item in os.listdir(persist_dir):
	                item_path = os.path.join(persist_dir, item)
	                if os.path.isfile(item_path):
	                    os.remove(item_path)
	                elif os.path.isdir(item_path):
	                    shutil.rmtree(item_path)
	        else:
	            print(f"✅ Skipping {label} — already saved.")
	            continue

	    print(f"🔹 Loading model: {model_name}")
	    model = SentenceTransformer(model_name)

	    print(f"🔸 Generating embeddings for: {label}")
	    embeddings = model.encode(documents, show_progress_bar=True, batch_size=512)

	    print(f"💾 Saving to ChromaDB at: {persist_dir}")
	    chroma_client = chromadb.PersistentClient(path=persist_dir)

	    collection = chroma_client.get_or_create_collection(name="hotpotqa_chunks")



	    # Precompute IDs
	    ids = [f"{meta['id']}_chunk{meta['chunk_index']}" for meta in metadatas]

	    # Chunk everything into batches of MAX_CHROMA_BATCH
	    for doc_batch, emb_batch, meta_batch, id_batch in zip(
	        chunked(documents, MAX_CHROMA_BATCH),
	        chunked(embeddings.tolist(), MAX_CHROMA_BATCH),
	        chunked(metadatas, MAX_CHROMA_BATCH),
	        chunked(ids, MAX_CHROMA_BATCH),
	    ):
	        collection.add(
	            documents=doc_batch,
	            embeddings=emb_batch,
	            metadatas=meta_batch,
	            ids=id_batch,
	        )

	    print(f"✅ Saved {label} to ChromaDB\n")


def generate_embeddings(sample_sizes):
	for sample_size in tqdm(sample_sizes):
		path = f"data/sample/sampled_hotpot_train_{sample_size}.parquet"
		embeddings_save_path = f"data/embeddings/{sample_size}"
		os.makedirs(embeddings_save_path, exist_ok=True)
		df = pd.read_parquet(path)
		_generate_embeddings(df, embeddings_save_path)


if __name__ == "__main__":
	SAMPLE_SIZES = [100, 1000, 10000]
	generate_embeddings(SAMPLE_SIZES)
