"""五言绝句生成模型：byte-level BPE 分词器 + 无位置编码（NoPE）的 decoder-only Transformer。

设计要点
--------
* NoPE：没有 positional embedding，token 的先后顺序只能通过 causal mask 体现
  （参见 "NoPE: Positional Information is Redundant in Decoder-only Transformers"）。
* 激活函数用 ReLU（不是 GELU/SwiGLU），FFN 宽度 4x。
* 训练数据只有几千首诗，因此 dropout 用得比较重（默认 0.3）。
* 词嵌入与输出层共享权重（weight tying），省下一半 embedding 参数。
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = 256  # BPE 的起始词表：所有 UTF-8 字节


# ---------------------------------------------------------------------------
# BPE 分词器
# ---------------------------------------------------------------------------


def _apply_merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """把 ids 里所有相邻出现的 pair 替换成 new_id。"""
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """极简 byte-level BPE：词表 = 256 个字节 + 若干合并出来的 token。

    文本先编码成 UTF-8 字节，再按学到的 merge 规则贪心地合并。
    这样任何汉字（包括训练集里没出现过的）都能被表示。
    """

    def __init__(self, merges: list[tuple[int, int]] | None = None):
        self.merges = [tuple(m) for m in (merges or [])]
        self.rank = {pair: i for i, pair in enumerate(self.merges)}

        # token id -> 字节串（同时也是 id -> 原文）
        self.vocab = [bytes([i]) for i in range(BASE)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        # 约束解码时用来检查某个 token 展开成了什么文字
        self.token_text = [v.decode("utf-8", "replace") for v in self.vocab]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ---- 训练 -------------------------------------------------------------

    @classmethod
    def train(cls, texts, vocab_size: int, log_every: int = 200) -> "BPETokenizer":
        """在 texts 上学习 BPE 合并规则，直到词表达到 vocab_size。"""
        corpus = [list(t.encode("utf-8")) for t in texts]
        merges: list[tuple[int, int]] = []

        while BASE + len(merges) < vocab_size:
            counts: Counter = Counter()
            for ids in corpus:
                counts.update(zip(ids, ids[1:]))
            if not counts:
                break
            pair, freq = counts.most_common(1)[0]
            if freq < 2:  # 只出现一次的相邻组合没有推广价值
                break

            new_id = BASE + len(merges)
            merges.append(pair)
            corpus = [_apply_merge(ids, pair, new_id) for ids in corpus]

            if len(merges) % log_every == 0:
                print(f"  BPE {len(merges)}/{vocab_size - BASE} merges, "
                      f"最高频 pair={pair} × {freq}", flush=True)

        return cls(merges)

    # ---- 编解码 -----------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) > 1:
            # 找出当前所有相邻对中 merge 优先级最高（rank 最小）的一个
            best_rank, best_pos = None, -1
            for i in range(len(ids) - 1):
                r = self.rank.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pos = r, i
            if best_rank is None:
                break
            ids[best_pos:best_pos + 2] = [BASE + best_rank]
        return ids

    def decode(self, ids) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", "replace")

    # ---- 存取 -------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"merges": self.merges}), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["merges"])


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------


@dataclass
class Config:
    vocab_size: int = 1600   # + 1 个 EOS，由 train.py 覆盖
    block_size: int = 128    # 上下文长度
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 128
    dropout: float = 0.3

    def to_dict(self) -> dict:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    """多头因果自注意力：位置信息只来自因果掩码（NoPE）。"""

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q, k, v = (
            t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            for t in (q, k, v)
        )
        # is_causal=True 就是唯一的“位置编码”
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    """ReLU 前馈网络，宽度 4x。"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.fc2 = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(F.relu(self.fc1(x)))))


class Block(nn.Module):
    """pre-LN 残差块。"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)  # 唯一的 embedding，没有 position
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)
        # 残差投影层缩小初始化，训练更稳
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        # 共享权重只算一次
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.drop(self.tok_emb(idx))          # (B, T, C)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))          # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss
