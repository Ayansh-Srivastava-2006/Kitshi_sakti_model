from pathlib import Path
import argparse, math, time
import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import ModelConfig, KrishiLM, count_parameters

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints"

class TokenBlockDataset(Dataset):
    def __init__(self, ids, block_size):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.block = block_size
    def __len__(self):
        return max(0, len(self.ids) - self.block)
    def __getitem__(self, i):
        x = self.ids[i:i+self.block]
        y = self.ids[i+1:i+self.block+1]
        return x, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_steps", type=int, default=1500)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    if device == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    sp = spm.SentencePieceProcessor(model_file=str(DATA / "krishi_sp.model"))
    text = (DATA / "labeled_corpus.txt").read_text(encoding="utf-8")
    ids = sp.encode(text, out_type=int, add_bos=True, add_eos=True)
    print("tokens:", len(ids))

    ds = TokenBlockDataset(ids, args.block_size)
    if len(ds) < 16:
        raise RuntimeError("Corpus is too small after tokenization")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    it = iter(loader)

    config = ModelConfig(
        vocab_size=sp.vocab_size(),
        block_size=args.block_size,
        n_layer=8,
        n_head=8,
        n_embd=384,
        dropout=0.0,
    )
    model = KrishiLM(config).to(device)
    print(f"parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    model.train()
    best = float("inf")
    t0 = time.time()
    for step in tqdm(range(1, args.max_steps + 1)):
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for _ in range(args.grad_accum):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader)
                x, y = next(it)
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                _, loss = model(x, y)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            running += loss.item()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % 25 == 0 or step == 1:
            elapsed = time.time() - t0
            ppl = math.exp(min(12.0, running * args.grad_accum))
            print(f"step={step} loss={running*args.grad_accum:.4f} ppl~={ppl:.1f} elapsed={elapsed/60:.1f}m")
        if step % 250 == 0 or step == args.max_steps:
            state = {
                "model": model.state_dict(),
                "config": config.__dict__,
                "step": step,
                "sp_model": str(DATA / "krishi_sp.model"),
            }
            CKPT.mkdir(exist_ok=True)
            path = CKPT / "krishi_base.pt"
            torch.save(state, path)
            print("saved", path)
        if running < best:
            best = running

if __name__ == "__main__":
    main()
