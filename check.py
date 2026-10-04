from pathlib import Path
import sentencepiece as spm
from model import ModelConfig, KrishiLM, count_parameters

ROOT = Path(__file__).resolve().parent
sp = spm.SentencePieceProcessor(model_file=str(ROOT/'data'/'krishi_sp.model'))
cfg = ModelConfig(vocab_size=sp.vocab_size())
model = KrishiLM(cfg)
print('vocab:', sp.vocab_size())
print('parameters:', count_parameters(model))
print('sample tokens:', sp.encode('गेहूं की फसल में पत्तियां पीली हैं', out_type=int))
