# Interactive Movie Recommendation System
## Belonovskaya Kristina

This repository contains the experimental pipeline for a thesis project on **interactive movie recommendation in a natural language interface**.

The project studies the task of generating movie recommendations from a conversational context. Given a dialogue between a user and an assistant, the system must produce a ranked list of movies that match the user's current preferences. The target item is the movie accepted by the user as the final recommendation in the original dialogue.

The experiments compare several groups of methods:

* classical recommender baselines,
* semantic retrieval over a fixed movie catalog,
* LLM-only recommendation,
* a proposed hybrid method,
* a modified hybrid method with explicit dialogue state.

The main dataset is **INSPIRED**, a collection of human-human movie recommendation dialogues.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set:

```bash
OPENROUTER_API_KEY=...
MODEL_NAME=...
```

Example model names used in the experiments:

```bash
MODEL_NAME=minimax/minimax-m2.5
MODEL_NAME=openai/gpt-5.3-chat
```

Raw INSPIRED files should be placed manually into:

```text
data/raw/
```

Processed artifacts such as the movie catalog, title index, IMDb ID mappings, and FAISS index are generated automatically by the preprocessing pipeline.

---


## Run experiments

Run one recommender:

```bash
python scripts/evaluate.py --recommender popularity
python scripts/evaluate.py --recommender svd
python scripts/evaluate.py --recommender semantic_retrieval
python scripts/evaluate.py --recommender llm_query_only
python scripts/evaluate.py --recommender hybrid
python scripts/evaluate.py --recommender hybrid_state
```

Run all recommenders:

```bash
python scripts/evaluate.py --recommender all
```

Predictions are written incrementally to:

```text
results/logs/<recommender>.jsonl
```

Re-running an experiment resumes from the existing log. To start from scratch, pass:

```bash
--no-resume
```

---

## Recommenders

| # | Name                 | What it uses                                                                         |
| - | -------------------- | ------------------------------------------------------------------------------------ |
| 1 | `popularity`         | Global movie popularity from the training set                                        |
| 2 | `svd`                | Latent factors from a binary user-item co-mention matrix                             |
| 3 | `semantic_retrieval` | Dialogue text and user query → `BAAI/bge-base-en-v1.5` embeddings → FAISS search     |
| 4 | `llm_query_only`     | Annotated user query → LLM-generated movie titles → title resolution                 |
| 5 | `hybrid`             | Preference extraction → FAISS candidates ∪ LLM open-world candidates → LLM reranking |
| 6 | `hybrid_state`       | `hybrid` + explicit turn-by-turn dialogue state                                      |

---



## Project layout

```text
recsys/
├── config.py                 # Paths, model names, and global settings
├── data/                     # INSPIRED loaders, catalog, splits, title index, ID mapping
├── llm/                      # OpenRouter client and LLM utilities
├── retrieval/                # FAISS index and semantic search
├── recommenders/             # BaseRecommender and recommender implementations
└── evaluation/               # Metrics, runner, and reporting utilities

scripts/
├── evaluate.py               # Main CLI entry point for full-context evaluation
└── evaluate_turnwise.py      # Evaluation at dialogue-history cutoffs

data/
├── raw/                      # Raw INSPIRED files, downloaded manually
└── processed/                # Generated artifacts:
                             # catalog.json
                             # faiss_index/
                             # movie_id_map.json
                             # movie_title_retrieval_index.json

results/
└── logs/                     # Prediction logs in JSONL format, resume-safe
```

---

## Main results

| Recommender                         | HitRate@10 |        MRR |    NDCG@10 |
| ----------------------------------- | ---------: | ---------: | ---------: |
| `popularity`                        |     10.57% |      4.41% |      5.87% |
| `svd`                               |     11.01% |      5.44% |      6.74% |
| `semantic_retrieval`                |     17.62% |      9.85% |     11.98% |
| `llm_query_only` / `minimax-m2.5`   |     22.42% |     13.36% |     15.47% |
| `llm_query_only` / `gpt-5.3-chat`   |     27.97% |     15.92% |     18.76% |
| `hybrid` / `minimax-m2.5`           |     29.07% |     17.69% |     20.35% |
| `hybrid` / `gpt-5.3-chat`           |     27.75% |     16.39% |     19.00% |
| `hybrid_state` / `gpt-5.3-chat`     |     28.41% |     16.82% |     19.55% |
| **`hybrid_state` / `minimax-m2.5`** | **30.18%** | **18.78%** | **21.49%** |

The results show a consistent improvement when moving from classical recommenders to methods that use natural language. The best overall result is achieved by `hybrid_state` with `minimax-m2.5`.

---

## Cost and runtime

LLM-based methods require API calls during inference. The table below reports the validation cost and runtime from the thesis experiments.

| Method           | Model          | API calls | Cost, USD | Runtime, h |
| ---------------- | -------------- | --------: | --------: | ---------: |
| `llm_query_only` | `minimax-m2.5` |       446 |      0.38 |        1.2 |
| `hybrid`         | `minimax-m2.5` |     1,362 |      1.17 |        1.5 |
| `hybrid_state`   | `minimax-m2.5` |     7,036 |      6.04 |        2.0 |
| `llm_query_only` | `gpt-5.3-chat` |       454 |      1.44 |        0.7 |
| `hybrid`         | `gpt-5.3-chat` |     1,362 |      4.33 |        0.9 |
| `hybrid_state`   | `gpt-5.3-chat` |     6,133 |     19.48 |        1.7 |

`hybrid` gives a quality improvement over `llm_query_only` with a moderate cost increase. `hybrid_state` gives the best quality, but is substantially more expensive because it updates dialogue state after each user turn.
