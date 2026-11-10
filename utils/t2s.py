"""使用 opencc 把全唐诗五言绝句数据转换为简体中文。

输入：data/poet.tang.wuyan.json
输出：data/poet.tang.wuyan.zh_cn.json

用法：
    python t2s.py
"""

import json
from pathlib import Path

import opencc

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "poet.tang.wuyan.json"
DST = ROOT / "data" / "poet.tang.wuyan.zh_cn.json"


def main() -> None:
    converter = opencc.OpenCC("t2s")

    with SRC.open(encoding="utf-8") as f:
        poems = json.load(f)

    simplified = [
        {
            "author": converter.convert(poem["author"]),
            "paragraphs": [converter.convert(line) for line in poem["paragraphs"]],
            "title": converter.convert(poem["title"]),
        }
        for poem in poems
    ]

    with DST.open("w", encoding="utf-8") as f:
        json.dump(simplified, f, ensure_ascii=False, indent=4)
        f.write("\n")

    print(f"{len(simplified)} 首 -> {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
