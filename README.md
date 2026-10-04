# Krishi Shakti — From-Scratch LLM Prototype

This prototype is a **small decoder-only Transformer trained from random initialization**. It does not use Qwen/Llama/Gemma weights.

It is intentionally small enough to experiment with on a 6 GB GPU. The default configuration is ~20–30M parameters depending on vocabulary size.

## What is included

- `prepare_data.py` — creates a clean bootstrap agriculture corpus and an unlabeled query set.
- `tokenizer_train.py` — trains a SentencePiece tokenizer from scratch.
- `model.py` — decoder-only Transformer with RMSNorm, RoPE, SwiGLU, causal self-attention, and tied embeddings.
- `train.py` — self-supervised next-token pretraining from random weights.
- `generate.py` — interactive CLI generation.
- `semi_supervised.py` — prototype pseudo-labeling loop over unlabeled farmer queries.
- `requirements.txt` — Python dependencies.

## Important

The included corpus is a **small bootstrap corpus written for this prototype**, not a replacement for the larger public datasets we identified. It is intended to prove that the full from-scratch pipeline works today.

For a serious model, replace/extend `data/labeled_corpus.txt` with legally usable public agriculture data such as KCC/KisanVaani and curated government/ICAR material, after checking the source license/terms.

## Run

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python prepare_data.py
python tokenizer_train.py
python train.py --max_steps 1500
python generate.py
```

For a quick smoke test:

```bash
python train.py --max_steps 20 --batch_size 2 --grad_accum 2
```

Then:

```bash
python semi_supervised.py --max_examples 25
```

## Suggested hardware

- Prototype: NVIDIA GPU with 6 GB+ VRAM or CPU for tiny smoke tests.
- 32 GB system RAM is helpful for larger corpora.
- The first version intentionally uses a short context window to keep memory use manageable.
