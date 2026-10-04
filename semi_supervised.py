from pathlib import Path
import argparse, math
import sentencepiece as spm
import torch
import torch.nn.functional as F
from model import ModelConfig, KrishiLM

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints" / "krishi_base.pt"
OUT = DATA / "pseudo_labeled.txt"

def load():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(CKPT, map_location=device)
    cfg = ModelConfig(**state["config"])
    model = KrishiLM(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    sp = spm.SentencePieceProcessor(model_file=str(DATA / "krishi_sp.model"))
    return model, sp, device

@torch.no_grad()
def generate_with_score(model, sp, device, prompt, max_new_tokens=70):
    prompt_ids = sp.encode(prompt, out_type=int, add_bos=True)
    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = x.clone()
    logps = []
    for _ in range(max_new_tokens):
        logits, _ = model(generated[:, -model.config.block_size:])
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        token = torch.argmax(probs, dim=-1, keepdim=True)
        logps.append(torch.log(probs.gather(-1, token).clamp_min(1e-9)))
        generated = torch.cat([generated, token], dim=1)
        if token.item() == sp.eos_id():
            break
    answer = sp.decode(generated[0].tolist())
    score = torch.cat(logps).mean().item() if logps else -99.0
    return answer, score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_examples", type=int, default=25)
    ap.add_argument("--min_logprob", type=float, default=-5.0)
    args = ap.parse_args()

    model, sp, device = load()
    queries = [x.strip() for x in (DATA / "unlabeled_queries.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    accepted = []
    for q in queries[:args.max_examples]:
        answer, score = generate_with_score(model, sp, device, q)
        good_len = 8 <= len(answer.split()) <= 100
        accepted_flag = score >= args.min_logprob and good_len
        print(f"score={score:.3f} accepted={accepted_flag} | {q}")
        if accepted_flag:
            accepted.append(f"### pseudo_label\nQuestion: {q}\nAnswer: {answer}\n")

    OUT.write_text("\n".join(accepted), encoding="utf-8")
    print(f"Accepted {len(accepted)} pseudo-labeled examples -> {OUT}")
    print("NOTE: This is a prototype filter, not a production confidence estimator. For deployment, add source grounding and human/expert verification.")

if __name__ == "__main__":
    main()
