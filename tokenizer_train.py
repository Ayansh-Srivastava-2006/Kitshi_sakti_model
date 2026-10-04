from pathlib import Path
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CORPUS = DATA / "labeled_corpus.txt"
PREFIX = DATA / "krishi_sp"

spm.SentencePieceTrainer.train(
    input=str(CORPUS),
    model_prefix=str(PREFIX),
    vocab_size=2048,
    model_type="bpe",
    character_coverage=1.0,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    user_defined_symbols=["<|user|>", "<|assistant|>", "<|system|>"],
    hard_vocab_limit=False,
    shuffle_input_sentence=True,
    seed_sentencepiece_size=100000,
)
print(f"Tokenizer written to {PREFIX}.model")
