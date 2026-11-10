"""用训练好的 checkpoint 写五言绝句。

用法：
    .venv/bin/python inference.py                                   # 随机写 5 首
    .venv/bin/python inference.py -n 3 -t 0.8                       # 温度 0.8
    .venv/bin/python inference.py -p "床前明月光，"                   # 续写第一句
    .venv/bin/python inference.py --free                             # 不加格式约束

默认开启“格式约束解码”：模型只能在“五个汉字 / 逗号 / 句号”里选下一个 token，
所以输出一定是四句五言。--free 关掉约束，让模型自由发挥。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from model import BPETokenizer, Config, GPT

HAN = re.compile(r"^[\u4e00-\u9fff]+$")
PUNCT = "，。"


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    tok = BPETokenizer(ckpt["merges"])
    cfg = Config(**ckpt["config"])
    assert cfg.vocab_size == tok.vocab_size + 1, "checkpoint 与分词器词表不一致"

    model = GPT(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(device)
    return model, tok


class StrictFormat:
    """给“五言绝句”这一固定格式生成候选 token 的缓存。"""

    def __init__(self, token_text: list[str]):
        self.token_text = token_text
        self.cache: dict[tuple[int, int], torch.Tensor] = {}

    def allowed(self, n_punct: int, n_han: int) -> torch.Tensor:
        key = (n_punct, n_han)
        if key not in self.cache:
            if n_han >= 5:
                # 该打标点了：第 1、3 句后是逗号，第 2、4 句后是句号
                want = PUNCT[n_punct % 2]
                ids = [i for i, s in enumerate(self.token_text) if s == want]
            else:
                # 还在句内，还能再写 5 - n_han 个汉字
                room = 5 - n_han
                ids = [i for i, s in enumerate(self.token_text)
                       if s and len(s) <= room and HAN.match(s)]
            self.cache[key] = torch.tensor(ids, dtype=torch.long)
        return self.cache[key]


def penalize_repeats(logits: torch.Tensor, ids: list[int],
                     penalty: float, window: int = 12) -> torch.Tensor:
    """轻微惩罚最近出现过的 token，避免“草草草草”这种死循环。"""
    if penalty <= 1.0:
        return logits
    recent = torch.tensor(sorted(set(ids[-window:])), device=logits.device)
    vals = logits[recent]
    logits[recent] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def poem_state(text: str) -> tuple[int, int]:
    """返回（已经写出的标点数，当前句已有几个汉字）。"""
    n_punct = sum(text.count(c) for c in PUNCT)
    cut = max(text.rfind(c) for c in PUNCT)
    return n_punct, len(text) - cut - 1


@torch.no_grad()
def generate(model: GPT, tok: BPETokenizer, prompt: str = "", *,
             max_new_tokens: int = 64, temperature: float = 0.9,
             top_k: int = 50, repeat_penalty: float = 1.1,
             strict: bool = True, device: torch.device) -> str:
    cfg = model.cfg
    eos = tok.vocab_size
    ids = tok.encode(prompt) if prompt else [eos]  # 空提示时用 EOS 当作“开新诗”
    text = prompt
    strict_format = StrictFormat(tok.token_text)

    if not strict:
        # 自由模式：只允许能完整解码的 token，避免输出半个汉字变成乱码
        keep = [i for i, s in enumerate(tok.token_text) if "\ufffd" not in s]
        whole_text = torch.zeros(tok.vocab_size + 1, dtype=torch.bool, device=device)
        whole_text[keep] = True
        whole_text[eos] = True

    for _ in range(max_new_tokens):
        n_punct, n_han = poem_state(text)
        if strict and n_punct == 4:  # 四句已经写完
            break

        idx = torch.tensor([ids[-cfg.block_size:]], device=device)
        logits, _ = model(idx)
        logits = penalize_repeats(logits[0, -1] / temperature, ids,
                                  repeat_penalty)

        if strict:
            allowed = strict_format.allowed(n_punct, n_han).to(device)
            mask = torch.full_like(logits, float("-inf"))
            mask[allowed] = logits[allowed]
            logits = mask
        else:
            logits = logits.masked_fill(~whole_text, float("-inf"))

        if top_k:
            k = min(top_k, int(torch.isfinite(logits).sum()))
            if 0 < k < logits.numel():
                kth = torch.topk(logits, k).values[-1]
                logits = logits.masked_fill(logits < kth, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        ids.append(nxt)
        if not strict and nxt == eos:  # 模型自己写了首新的诗
            break
        text += tok.token_text[nxt]

    return text


def format_poem(text: str) -> str:
    """在句号后换行，让两联显示成两行。"""
    return text.replace("。", "。\n").strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="生成五言绝句",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-c", "--ckpt", type=Path, default=Path(__file__).parent / "ckpt.pt")
    p.add_argument("-p", "--prompt", default="", help="前半句提示，例如 '床前明月光，'")
    p.add_argument("-n", "--num", type=int, default=5, help="生成几首")
    p.add_argument("-t", "--temperature", type=float, default=0.9)
    p.add_argument("-k", "--top-k", type=int, default=50)
    p.add_argument("--repeat-penalty", type=float, default=1.1,
                   help="最近 12 个 token 的重复惩罚，1.0 表示关闭")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--free", action="store_true", help="关闭五言格式约束")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    if not args.ckpt.exists():
        raise SystemExit(f"找不到 {args.ckpt}，请先运行 train.py")

    model, tok = load_checkpoint(args.ckpt, device)
    print(f"模型 {model.num_params():,} 参数，device={device.type}，"
          f"温度 {args.temperature}，top-k {args.top_k}，"
          f"{'自由生成' if args.free else '五言格式约束'}\n")

    for i in range(args.num):
        poem = generate(
            model, tok, args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            repeat_penalty=args.repeat_penalty,
            strict=not args.free,
            device=device,
        )
        print(f"[{i + 1}]")
        print(format_poem(poem))
        print()


if __name__ == "__main__":
    main()
