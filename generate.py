from pathlib import Path
import argparse
import sentencepiece as spm
import torch
from model import ModelConfig, KrishiLM

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints" / "krishi_base.pt"

def load():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(CKPT, map_location=device)
    cfg = ModelConfig(**state["config"])
    model = KrishiLM(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    sp = spm.SentencePieceProcessor(model_file=str(DATA / "krishi_sp.model"))
    return model, sp, device

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default=None)
    args = ap.parse_args()
    model, sp, device = load()
    print("Krishi Shakti prototype. Type 'exit' to quit.")
    while True:
        prompt = args.prompt if args.prompt is not None else input("You: ").strip()
        if prompt.lower() == "exit":
            break
        ids = sp.encode(prompt, out_type=int, add_bos=True)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(x, max_new_tokens=100, temperature=0.75, top_k=40)
        text = sp.decode(out[0].tolist())
        print("Krishi Shakti:", text)
        if args.prompt is not None:
            break

if __name__ == "__main__":
    main()
