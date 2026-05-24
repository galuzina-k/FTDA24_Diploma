# Interactive Movie Recommendation System

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set OPENROUTER_API_KEY and MODEL_NAME
```

## Run

```bash
# One recommender
python scripts/evaluate.py --recommender popularity
python scripts/evaluate.py --recommender semantic_retrieval

# All recommenders
python scripts/evaluate.py --recommender all

# Parallel workers (for LLM-using recommenders)
python scripts/evaluate.py --recommender llm_query_only --workers 5
```

Predictions are written incrementally to `results/logs/<recommender>.jsonl`. Re-running resumes from where it left off — pass `--no-resume` to start fresh.

## Recommenders

| # | Name | What it uses |
|---|------|--------------|
| 1 | `popularity` | global popularity, no dialog signal |
| 2 | `svd` | latent factors from co-mention matrix |
| 3 | `semantic_retrieval` | dialog text → MiniLM embedding → FAISS search |
| 4 | `llm_query_only` | annotated user query → LLM → title resolution |
| 5 | `hybrid` | preference extraction → (FAISS pool ∪ LLM open-world recall) → LLM rerank |
| 6 | `hybrid_state` | hybrid + turn-by-turn conversation state (seen/rejected/constraints) |

### How `hybrid` works

Three LLM calls per dialog. The two retrieval methods cover complementary failure modes — FAISS finds plot-similar movies the seeker never named, while LLM open-world recall surfaces movies the seeker described in a way embeddings miss — and the reranker fuses both pools.

1. **Preference extraction** (LLM call #1). Dialog history + annotated query → structured `{genres, moods, likes, dislikes, search_query}` JSON.
2. **FAISS retrieval** (no LLM). The enriched search query goes through the bge-base FAISS index → top-20 candidates, tagged `source="faiss"`.
3. **LLM open-world recall** (LLM call #2). Seeker block + query → 20 movie titles → resolved to IMDb IDs via `TitleIndex` → tagged `source="llm"`.
4. **Merge and dedup** by IMDb ID. Typical merged pool is ~38 candidates.
5. **LLM rerank** (LLM call #3). Numbered list of candidates (with source tag, title, year, genres, overview) + conversation context + preferences → `{"ranking": [<10 indices>]}`. Indices map back to IMDb IDs.

Pool sizes are knobs (`FAISS_POOL`, `LLM_RECALL` constants in [recsys/recommenders/hybrid.py](recsys/recommenders/hybrid.py)) — increase them to raise the retrieval ceiling at the cost of larger rerank prompts.

Open-world recall and rerank token caps are set to **12,288** to avoid `finish_reason=length` failures on reasoning models like `minimax-m2.5`.

### `hybrid_state` — turn-by-turn conversation state

Extends `hybrid` with a `ConversationState` updated incrementally (one LLM call per seeker turn). State holds:
- `rejected_titles` — **hard filter** on the candidate pool (titles the bot pitched that the seeker rejected),
- `seen_titles` — **soft signal** in the rerank prompt ("deprioritize these"),
- `constraints` — **soft signal** in the rerank prompt (situational factors only: "watching with partner", "family-friendly", etc.).

On the full 454-dialog test set with `minimax-m2.5`, `hybrid_state` **outperforms `hybrid` by ~1pp** on all metrics (HR@10 30.18% vs 29.07%, NDCG@10 21.49% vs 20.35%) — a **positive result** that breaks down cleanly by dialog length: state hurts on short dialogs (≤4 seeker turns, −2.6pp HR) where extractor signals are noisy, but delivers a strong +3.6pp HR lift on medium-length dialogs (5–7 turns) and a modest +0.8pp on long dialogs (≥8 turns). Full analysis in [METRICS.md](METRICS.md).

## Turn-level evaluation

`scripts/evaluate_turnwise.py` evaluates each recommender at multiple dialog-history cutoffs (after 2, 4, 6, 8 seeker turns, and the full pre-recommendation context). At intermediate cutoffs the annotated `user_query` is set to `""` to prevent label leakage (it was generated from the full dialog). Logs are written to `results/logs/<recommender>__cutoff_<label>.jsonl`, resume-safe.

```bash
python scripts/evaluate_turnwise.py \
  --recommenders popularity svd semantic_retrieval llm_query_only \
  --workers 8
```

The output is a per-recommender table of HR@10/MRR/NDCG by cutoff — a "how quickly does each system converge on the right answer?" curve.

## Project layout

```
recsys/             # All code lives here
├── config.py       # Single source of truth for paths and settings
├── data/           # INSPIRED loaders, catalog, splits, title index, id mapping
├── llm/            # OpenRouter client
├── retrieval/      # FAISS index + searcher
├── recommenders/   # One BaseRecommender ABC + implementations + REGISTRY
└── evaluation/     # Metrics, runner, reporter

scripts/
└── evaluate.py     # CLI entry point

data/
├── raw/            # INSPIRED files (download manually)
└── processed/      # Auto-generated: catalog.json, faiss_index/, movie_id_map.json, movie_title_retrieval_index.json

results/
└── logs/           # Per-recommender prediction logs (resume-safe)
```

## Metrics

- **HitRate@10** — fraction of dialogs where the target movie appears in top-10
- **MRR** — mean reciprocal rank of the target
- **NDCG@10** — normalized discounted cumulative gain

| Recommender | HitRate@10 | MRR | NDCG@10 |
|---|---|---|---|
| popularity | 10.57% | 4.41% | 5.87% |
| svd | 11.01% | 5.44% | 6.74% |
| history_only | 3.08% | 1.03% | 1.50% |
| semantic_retrieval (NLP based) | 17.62% | 9.85% | 11.98% |
| llm_query_only (minimax) | 22.42% | 13.36% | 15.47% |
| llm_query_only (openai/gpt-5.3-chat) | 27.97% | 15.92% | 18.76% |
| hybrid (minimax) | 29.07% | 17.69% | 20.35% |
| hybrid (openai/gpt-5.3-chat) | 27.75% | 16.39% | 19.00% |
| hybrid_state (openai/gpt-5.3-chat) | 28.41% | 16.82% | 19.55% |
| **hybrid_state (minimax)** | **30.18%** | **18.78%** | **21.49%** |
