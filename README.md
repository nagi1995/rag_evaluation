# 🧪 RAG Evaluation Pipeline

A modular pipeline to evaluate **Retrieval-Augmented Generation (RAG)** systems using popular datasets, dense embeddings, and standard NLP evaluation metrics.

---

## 📌 Features

- ✅ Dataset download and preparation  
- ✅ Embedding generation using transformer models  
- ✅ Document retrieval and response generation  
- ✅ Response evaluation with multiple metrics  
- ✅ Modular scripts for each step  

---

## 📁 Project Structure

- `01_download_dataset.py` – Download evaluation datasets  
- `02_prepare_samples.py` – Preprocess and format samples  
- `03_generate_embeddings.py` – Generate vector embeddings  
- `04_retrieve_and_generate.py` – Perform retrieval and generate answers  
- `05_evaluate_metrics.py` – Evaluate generated responses  
- `requirements.txt` – Python dependencies  

---

## ⚙️ Installation

```bash
git clone https://github.com/nagi1995/rag_evaluation.git
cd rag_evaluation
pip install -r requirements.txt
```

---

## 📥 Dataset

This pipeline is currently tested using the [HotpotQA](https://hotpotqa.github.io/) dataset.

Download link: [HotpotQA](http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json)

---

## 🚀 How to Use

Run each step in order to complete the evaluation pipeline:

```bash
python 01_download_dataset.py
python 02_prepare_samples.py
python 03_generate_embeddings.py
python 04_retrieve_and_generate.py
python 05_evaluate_metrics.py
```
Intermediate outputs are saved and used in subsequent steps.

---

## 🧠 Evaluation Metrics Used

- ROUGE 
- BLEU 
- Exact Match (EM)
- F1 score

---

## 🧑‍💻 Author

**Nagesh**  
[GitHub](https://github.com/nagi1995) | [LinkedIn](https://www.linkedin.com/in/bnagesh1/)

