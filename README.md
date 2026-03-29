# Project-LLM-Eval-Pipeline
# LLM Eval Pipeline

A production-style evaluation pipeline for RAG systems that combines classical NLP metrics, semantic scoring, and LLM-as-Judge — using two different AI providers for generation and evaluation.

---

## What it does

1. **Generates answers** via a RAG system (OpenAI GPT-4o-mini + ChromaDB)
2. **Scores semantically** using BERTScore
3. **Evaluates with a Judge** using Anthropic Claude — measuring Faithfulness, Answer Relevancy, Context Recall, and overall judgment
4. **Exports results** as a structured CSV

---

## Architecture

```
data/questions.py        → Test questions
       ↓
rag/generator.py         → RAG chain (OpenAI + ChromaDB) generates answers
       ↓
eval/BERT_score.py       → BERTScore calculated per answer
       ↓
eval/judge.py            → Anthropic Claude evaluates quality
       ↓
pipeline.py              → Orchestrates all steps + exports CSV
```

---

## Metrics

| Metric | Description | Range |
|---|---|---|
| BERTScore | Semantic similarity between answer and context | 0–1 |
| LLM_judgment | Overall quality score by Claude | 0–1 |
| Faithfulness | Is the answer grounded in the context? | 0–1 |
| Answer_Relevancy | Does the answer address the question? | 0–1 |
| Context_Recall | Were the right documents retrieved? | 0–1 |

---

## Setup

**1. Clone the repo and create a virtual environment:**
```bash
git clone <your-repo-url>
cd LLM-Eval-Pipeline
python3 -m venv venv
source venv/bin/activate
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Create a `.env` file in the root:**
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

**4. Add your ChromaDB path in `rag/generator.py`:**
```python
persist_directory='path/to/your/chroma_db'
```

**5. Add your questions in `data/questions.py`:**
```python
questions = [
    {"question": "Your question here?"},
    ...
]
```

---

## Run

Always execute from the project root:
```bash
python pipeline.py
```

Results are saved to `results.csv`.

---

## Tech Stack

- **Generation:** OpenAI GPT-4o-mini via LangChain
- **Vector Store:** ChromaDB with OpenAI Embeddings
- **Semantic Scoring:** BERTScore (roberta-large)
- **Judge:** Anthropic Claude (claude-opus-4-5)
- **Output:** Pandas DataFrame → CSV