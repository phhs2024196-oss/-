#!/usr/bin/env python3
"""원본 PDF 에서 '확인 필요' 부분만 잘라 이미지로 만든다(전체 쪽 대신 몇 줄만 보게 해 이미지 토큰 절약).

사용:
  python3 tools/crop.py --pdf G24_am.pdf --find "닫힌구간에서의최댓값" [--lines 2] [--dpi 130] --out /tmp/c1.png
  --find 는 한글만 이어 쓴 조각(source_check 출력의 한글 문맥을 그대로 복사). 공백·수식은 무시하고 찾는다.
  찾은 줄 위아래로 --lines 줄씩 더해 자른다(분수·첨자가 줄 사이에 걸리므로 기본 2).
출력: 저장한 이미지 경로와 쪽 번호. 못 찾으면 종료코드 1(그 경우 쪽 전체를 봐야 함).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ET


def hangul(s: str) -> str:
    return "".join(c for c in s if "가" <= c <= "힣")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--find", required=True)
    ap.add_argument("--lines", type=int, default=2)
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    key = hangul(a.find)
    xml = subprocess.run(["pdftotext", "-bbox-layout", a.pdf, "-"], capture_output=True, text=True, check=True).stdout
    xml = re.sub(r"<!DOCTYPE[^>]*>", "", xml).replace('xmlns="http://www.w3.org/1999/xhtml"', "")
    root = ET.fromstring(xml)
    for pno, page in enumerate(root.iter("page"), 1):
        lines = [ln for ln in page.iter("line")]
        hs = [hangul("".join(w.text or "" for w in ln.iter("word"))) for ln in lines]
        joined, owner = "", []
        for i, h in enumerate(hs):
            joined += h
            owner += [i] * len(h)
        k = joined.find(key)
        if k < 0:
            continue
        i0, i1 = owner[k], owner[k + len(key) - 1]
        i0, i1 = max(0, i0 - a.lines), min(len(lines) - 1, i1 + a.lines)
        y0 = min(float(lines[i]. get("yMin")) for i in range(i0, i1 + 1)) - 4
        y1 = max(float(lines[i].get("yMax")) for i in range(i0, i1 + 1)) + 4
        W = float(page.get("width"))
        sc = a.dpi / 72
        out = a.out[:-4] if a.out.endswith(".png") else a.out
        subprocess.run(["pdftoppm", "-f", str(pno), "-l", str(pno), "-r", str(a.dpi), "-png", "-singlefile",
                        "-x", "0", "-y", str(int(max(0, y0) * sc)), "-W", str(int(W * sc)),
                        "-H", str(int((y1 - y0) * sc)), a.pdf, out], check=True)
        print(f"{out}.png  (p.{pno}, {i1 - i0 + 1}줄)")
        return 0
    print("못 찾음 — 쪽 전체를 확인", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
