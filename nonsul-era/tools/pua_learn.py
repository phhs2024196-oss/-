#!/usr/bin/env python3
"""PDF 수식 글꼴의 PUA 글자 → 실제 글자 표(db/pua_map.yaml)를 '검수된' 문항과 원본으로 학습·누적.

사용: python3 tools/pua_learn.py --pair G22A.pdf 'items/AJOU/PAST/AJOU-G22A-0*.yaml' --pair … [--write]
학습(신뢰성 우선):
  1) 한글이 일치하는 이웃 두 글자 사이 수식 조각에서 원본·DB 기호 수가 같은 것만 쓴다.
  2) 반복: 조각의 비-PUA 기호와 이미 아는 PUA 가 자리마다 전부 맞을 때만 나머지 PUA 에 투표
     (분수·첨자로 순서가 섞인 조각은 여기서 걸러진다). 첫 회는 기호 3개 이하 조각만.
  3) 채택: 2표 이상 + 95% 이상이 한 글자. 미달 글자는 표에 없음 → source_check 에서 그 필드는 FLAG.
"""
from __future__ import annotations

import argparse
import collections
import difflib
import glob
import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from source_check import Source, read_src  # noqa: E402
from textnorm import POINTS, PUA_PATH, is_pua, math_seq, plain  # noqa: E402


def segments(src: Source, text: str):
    p = POINTS.sub(" ", plain(text, "latex"))
    ph = [(i, c) for i, c in enumerate(p) if "가" <= c <= "힣"]
    fh = "".join(c for _, c in ph)
    for b in difflib.SequenceMatcher(None, src.h, fh, autojunk=False).get_matching_blocks():
        for k in range(max(0, b.size - 1)):
            si, ei = src.pos[b.a + k], src.pos[b.a + k + 1]
            di, de = ph[b.b + k][0], ph[b.b + k + 1][0]
            st, dt = math_seq(src.text[si + 1:ei]), math_seq(p[di + 1:de])
            if st and any(is_pua(c) for c in st):
                yield st, dt


def learn(segs_all, known: dict) -> tuple[dict, dict]:
    """known 값 '' = 구조 글리프(분수선·괄호 조각 등, 대응 기호 없음)."""
    known = dict(known)
    for it in range(8):
        segs = []
        dele = collections.defaultdict(collections.Counter)
        for st0, dt in segs_all:
            st = [c for c in st0 if known.get(c, None) != ""]
            if len(st) == len(dt):
                segs.append((st, dt))
            elif len(st) > len(dt):
                # 모르는 PUA 한 종류를 모두 빼면 길이가 맞고 나머지가 전부 일치 → 그 글자는 구조 글리프 후보
                for c in {x for x in st if is_pua(x) and x not in known}:
                    rest = [x for x in st if x != c]
                    if len(rest) == len(dt) and all((known.get(x) == y) if is_pua(x) else x == y
                                                    for x, y in zip(rest, dt)):
                        dele[c]["∅"] += 1
        votes = collections.defaultdict(collections.Counter)
        for st, dt in segs:
            if it == 0 and not known and len(st) > 3:
                continue
            ok = all((a == c) if not is_pua(a) else (known[a] == c) if a in known else True
                     for a, c in zip(st, dt))
            if not ok:
                continue
            for a, c in zip(st, dt):
                if is_pua(a) and a not in known:
                    votes[a][c] += 1
        new = {a: c.most_common(1)[0][0] for a, c in votes.items()
               if c.most_common(1)[0][1] >= 2 and c.most_common(1)[0][1] / sum(c.values()) >= 0.95}
        # 구조 글리프(분수선·큰 괄호 조각 등): '빼면 나머지가 전부 맞는' 표가 5표 이상이고
        # 실제 글자 표(오정렬 잡음)의 3배 이상일 때만
        for a, c in dele.items():
            top = max(votes[a].values()) if a in votes else 0
            if c["∅"] >= 5 and c["∅"] >= 3 * top and a not in new:
                new[a] = ""
        if not new:
            break
        known.update(new)
    return known, votes


BLOCKS = ["ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz", "1234567890", "αβγδεζηθικλμνξοπ"]


def interpolate(known: dict) -> dict:
    """글꼴이 글자를 블록(대문자·소문자·숫자·그리스)으로 연속 배치한 경우, 같은 블록에서 확인된
    글자가 3개 이상이고 모두 같은 간격(offset)이면 확인된 최소~최대 사이만 채운다(바깥 추정 금지)."""
    inferred = {}
    for blk in BLOCKS:
        pts = [(ord(pu) - blk.index(ch), blk.index(ch)) for pu, ch in known.items() if ch in blk]
        offs = {o for o, _ in pts}
        if len(pts) < 3 or len(offs) != 1:
            continue
        off = offs.pop()
        lo, hi = min(i for _, i in pts), max(i for _, i in pts)
        for i in range(lo, hi + 1):
            pu = chr(off + i)
            if pu not in known:
                inferred[pu] = blk[i]
    return inferred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, action="append", required=True, metavar=("SRC", "ITEMS_GLOB"))
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--seed", action="append", default=[], metavar="HEX=글자",
                    help="문맥으로 확인한 글자를 직접 지정(빈 값=구조 글리프). 근거는 --note 에. 예: E06D= E05B=∫")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    data = (yaml.safe_load(PUA_PATH.read_text(encoding="utf-8")) if PUA_PATH.exists() else None) or {}
    known = {chr(int(k, 16)): v["char"] for k, v in (data.get("map") or {}).items() if not v.get("inferred")}
    seeds = {k: v for k, v in (data.get("map") or {}).items() if v.get("seed")}
    for sd in a.seed:
        k, _, v = sd.partition("=")
        known[chr(int(k, 16))] = v
        seeds[k.upper()] = {"char": v, "seed": True, "note": a.note}
    segs = []
    for srcp, pat in a.pair:
        src = Source(read_src(srcp), "none")          # 학습은 표를 적용하지 않은 원본으로
        for f in sorted(glob.glob(pat)):
            it = yaml.safe_load(pathlib.Path(f).read_text(encoding="utf-8"))
            texts = [s.get("body") or "" for s in it.get("stimuli") or []] + [it.get("stem") or ""]
            texts += [s.get("stem") or "" for s in it.get("subparts") or []]
            for t in texts:
                segs.extend(segments(src, t))
    before = len(known)
    known, left = learn(segs, known)
    # 이미 채택된 글자와 모순되는 조각 비율(학습 품질 지표)
    conflict = 0
    for st0, dt in segs:
        st = [c for c in st0 if known.get(c) != ""]
        if len(st) == len(dt) and all(not is_pua(x) or x in known for x in st) and \
                any((known.get(x, x) if is_pua(x) else x) != c for x, c in zip(st, dt)):
            conflict += 1
    print(f"조각 {len(segs)} · 표 {before}→{len(known)}글자 · 표로 다 풀리는 조각 중 불일치 {conflict}"
          f" (분수·첨자 순서차 또는 DB 오기)")
    for ch, c in sorted(left.items()):
        if ch not in known:
            print(f"  보류 {ord(ch):04X} {dict(c)}")
    inf = interpolate({k: v for k, v in known.items() if v})
    print("구조 글리프(무시): " + " ".join(f"{ord(k):04X}" for k, v in sorted(known.items()) if v == ""))
    print(f"블록 보간 {len(inf)}글자: " + " ".join(f"{ord(k):04X}={v}" for k, v in sorted(inf.items())))
    if a.write:
        mp = {f"{ord(k):04X}": {"char": v} for k, v in known.items()}
        mp.update(seeds)
        mp.update({f"{ord(k):04X}": {"char": v, "inferred": True} for k, v in inf.items()})
        out = {"note": "PDF 수식 글꼴 PUA → 글자. tools/pua_learn.py 가 생성. 손으로 고치지 말 것.",
               "map": dict(sorted(mp.items()))}
        PUA_PATH.write_text(yaml.dump(out, allow_unicode=True, sort_keys=False), encoding="utf-8")
        print("저장", PUA_PATH.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
