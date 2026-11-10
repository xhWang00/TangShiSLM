"""从头训练一个约 1M 参数的五言绝句生成模型。

用法：
    .venv/bin/python train.py                    # 默认 150 个 epoch
    .venv/bin/python train.py --epochs 300 --dropout 0.35

产物：
    ckpt.pt          验证集 loss 最低的模型（含 config 和 BPE merges）
    loss_curve.png   train / val loss 曲线
    tokenizer.json   BPE 合并规则缓存（删掉就会重新训练分词器）
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import torch

from model import BPETokenizer, Config, GPT

ROOT = Path(__file__).resolve().parent
HAN_RUN = re.compile(r"[\u4e00-\u9fff]+")


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------


def load_poems(path: Path) -> list[str]:
    """把 json 规整成 "五言，五言。五言，五言。" 的形式。

    原数据里混了少量残句、六句诗、标点缺失等情况，这里只保留恰好四句五言的。
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    poems = []
    for item in raw:
        runs = HAN_RUN.findall("".join(item["paragraphs"]))
        lines = [r for r in runs if len(r) == 5]
        if len(lines) == 4:
            poems.append(f"{lines[0]}，{lines[1]}。{lines[2]}，{lines[3]}。")
    print(f"数据：{len(raw)} 首 -> 保留 {len(poems)} 首标准五言绝句")
    return poems


def get_tokenizer(texts: list[str], vocab_size: int, path: Path,
                  retrain: bool) -> BPETokenizer:
    """读取缓存的 BPE；没有或词表大小不符时重新训练。"""
    if path.exists() and not retrain:
        tok = BPETokenizer.load(path)
        if tok.vocab_size == vocab_size:
            print(f"分词器：复用 {path.name}（词表 {tok.vocab_size}）")
            return tok

    print(f"分词器：在训练集上学习 BPE，目标词表 {vocab_size} ...")
    t0 = time.time()
    tok = BPETokenizer.train(texts, vocab_size)
    tok.save(path)
    print(f"分词器：完成，词表 {tok.vocab_size}，耗时 {time.time() - t0:.1f}s")
    return tok


def make_blocks(texts: list[str], tok: BPETokenizer, block_size: int,
                eos: int) -> torch.Tensor:
    """把所有诗拼成一条长流，再切成 (block_size + 1) 的定长块。"""
    ids: list[int] = []
    for t in texts:
        ids.extend(tok.encode(t))
        ids.append(eos)  # EOS 既是句号后的分隔符，也是“新诗开始”的标记

    n = len(ids) // (block_size + 1)
    data = torch.tensor(ids[: n * (block_size + 1)], dtype=torch.long)
    return data.view(n, block_size + 1)


def batches(data: torch.Tensor, batch_size: int, device: torch.device):
    """一个 epoch 的 mini-batch（打乱顺序，丢掉最后一个不满的 batch）。"""
    perm = torch.randperm(len(data))
    for i in range(0, len(data) - batch_size + 1, batch_size):
        block = data[perm[i: i + batch_size]].to(device)
        yield block[:, :-1], block[:, 1:]


# ---------------------------------------------------------------------------
# 训练工具
# ---------------------------------------------------------------------------


def lr_at(step: int, total: int, base_lr: float, warmup: int,
          min_ratio: float = 0.1) -> float:
    """线性 warmup + cosine 衰减。"""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def evaluate(model: GPT, data: torch.Tensor, batch_size: int,
             device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(data), batch_size):
        block = data[i: i + batch_size].to(device)
        _, loss = model(block[:, :-1], block[:, 1:])
        total += loss.item() * len(block)
        count += len(block)
    model.train()
    return total / count


def rolling_mean(values: list[float], window: int = 50) -> list[float]:
    out, acc = [], 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


def plot_loss(steps: list[int], train_loss: list[float],
              val_steps: list[int], val_loss: list[float], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 让标题里的汉字能正常显示（macOS 自带字体）
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Songti SC",
                                       "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps, train_loss, color="C0", alpha=0.2, lw=0.8, label="train (raw)")
    ax.plot(steps, rolling_mean(train_loss), color="C0", lw=1.8,
            label="train (mean 50)")
    ax.plot(val_steps, val_loss, "o-", color="C1", lw=1.4, ms=3, label="val")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("NoPE 1M 五言绝句模型训练曲线")
    ax.grid(alpha=0.3)

    # 标出验证集最好的那一步（保存的 checkpoint 就来自这里）
    i = min(range(len(val_loss)), key=val_loss.__getitem__)
    ax.scatter([val_steps[i]], [val_loss[i]], color="C1", zorder=5)
    ax.annotate(f"best {val_loss[i]:.4f} @ step {val_steps[i]}",
                (val_steps[i], val_loss[i]), textcoords="offset points",
                xytext=(10, 12), fontsize=9, color="C1")

    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"loss 曲线已保存到 {path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data/poet.tang.wuyan.zh_cn.json")
    p.add_argument("--out", type=Path, default=ROOT / "ckpt.pt")
    p.add_argument("--plot", type=Path, default=ROOT / "loss_curve.png")
    p.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizer.json")
    p.add_argument("--retrain-tokenizer", action="store_true")

    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--vocab-size", type=int, default=1600)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--d-model", type=int, default=128)

    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # ---- 数据 -------------------------------------------------------------
    poems = load_poems(args.data)
    g = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(poems), generator=g).tolist()
    n_val = max(1, int(len(poems) * args.val_frac))
    val_texts = [poems[i] for i in order[:n_val]]
    train_texts = [poems[i] for i in order[n_val:]]

    tok = get_tokenizer(train_texts, args.vocab_size, args.tokenizer,
                        args.retrain_tokenizer)
    eos = tok.vocab_size

    cfg = Config(
        vocab_size=tok.vocab_size + 1,  # +1 给 EOS
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        dropout=args.dropout,
    )

    train_data = make_blocks(train_texts, tok, cfg.block_size, eos)
    val_data = make_blocks(val_texts, tok, cfg.block_size, eos)
    tokens_per_epoch = len(train_data) * cfg.block_size
    print(f"数据：train {len(train_data)} 块 / val {len(val_data)} 块，"
          f"每块 {cfg.block_size + 1} token，约 {tokens_per_epoch} token/epoch")

    # ---- 模型与优化器 -----------------------------------------------------
    model = GPT(cfg).to(device)
    print(f"模型：{model.num_params():,} 参数 "
          f"({model.num_params() / 1e6:.2f}M)，device={device.type}")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    steps_per_epoch = max(1, len(train_data) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    print(f"训练：{args.epochs} epoch × {steps_per_epoch} step = {total_steps} step，"
          f"batch {args.batch_size}，lr {args.lr}，dropout {cfg.dropout}\n")

    best_val = evaluate(model, val_data, args.batch_size, device)
    print(f"step 0/{total_steps} | 初始 val loss {best_val:.4f}\n", flush=True)

    # ---- 训练循环 ---------------------------------------------------------
    train_steps: list[int] = []
    train_losses: list[float] = []
    val_steps: list[int] = [0]
    val_losses: list[float] = [best_val]
    step = 0
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_loss, epoch_steps = 0.0, 0

        for x, y in batches(train_data, args.batch_size, device):
            step += 1
            lr = lr_at(step, total_steps, args.lr, args.warmup)
            for group in optim.param_groups:
                group["lr"] = lr

            _, loss = model(x, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()

            train_steps.append(step)
            train_losses.append(loss.item())
            epoch_loss += loss.item()
            epoch_steps += 1

            elapsed = time.time() - t_start
            tok_s = step * args.batch_size * cfg.block_size / max(elapsed, 1e-9)
            print(f"step {step}/{total_steps} | loss {loss.item():.4f} "
                  f"| lr {lr:.2e} | {tok_s / 1000:.1f}k tok/s", flush=True)

        val = evaluate(model, val_data, args.batch_size, device)
        val_steps.append(step)
        val_losses.append(val)

        improved = val < best_val
        if improved:
            best_val = val
            torch.save({
                "config": cfg.to_dict(),
                "state_dict": model.state_dict(),
                "merges": tok.merges,
                "best_val": best_val,
                "step": step,
            }, args.out)

        print(f"epoch {epoch}/{args.epochs} | train loss {epoch_loss / max(epoch_steps, 1):.4f} "
              f"| val loss {val:.4f} | best {best_val:.4f}"
              f"{' (saved)' if improved else ''}", flush=True)

    # ---- 收尾 -------------------------------------------------------------
    plot_loss(train_steps, train_losses, val_steps, val_losses, args.plot)
    print(f"训练结束，最好 val loss {best_val:.4f}，模型保存在 {args.out}，"
          f"总耗时 {(time.time() - t_start) / 60:.1f} 分钟")
    print(f"试一下：.venv/bin/python inference.py --prompt '床前明月光，'")


if __name__ == "__main__":
    main()
