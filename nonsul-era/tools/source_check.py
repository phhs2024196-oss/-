#!/usr/bin/env python3
"""원본 ↔ DB 대조기 — 신뢰성 우선(놓침 최소화). 모델은 여기서 '확인 필요'로 나온 곳만 이미지로 본다.

사용 (시험지 1장 = 원본 1개 + 그 시험지의 문항 YAML 전부):
  python3 tools/source_check.py --src G24_am.pdf items/AJOU/PAST/AJOU-G24A-0*.yaml
  --src 는 .pdf(문제 쪽 자동 선택, --pages a-b 로 지정 가능) 또는 pdftotext 결과 .txt, HWP 변환 .txt(--src-kind hwp)

방법:
  1) 필드(제시문·발문·소문항 발문)마다 한글 글자열로 원본 위치를 찾는다.
  2) 찾은 필드를 원본 순서대로 이어 붙여, 원본의 해당 범위 전체와 '한 번에' 대조한다
     (필드 경계에서 생기는 거짓 경보를 없애고, 필드 사이에서 빠진 글도 잡는다).
     - 한글: 글자 단위 완전 일치만 통과.
     - 수식: 내용 기호열(숫자 자리·변수 문자·연산·괄호·그리스). PDF 수식 글꼴은 db/pua_map.yaml 로 해독.
  3) 배점: 소문항 points 가 그 소문항 원본 구간의 'n점' 중에 있어야 통과.
출력(차이만, 임계값으로 숨기지 않음):
  DIFF    내용이 다름(한글 글자, 수식 기호가 빠지거나 다르거나 더 있음) → 이미지 확인 필수
  ORDER   문항 전체로는 기호가 같고 자리만 다름(분수·첨자·연립식 배치에서 흔함) → 기본은 확인 대상
          (--lenient 면 제외. 단 a/b↔b/a 같은 뒤바뀜은 ORDER 로만 보이므로 기본값 유지 권장)
  UNREAD  원본 수식 글꼴 글자를 해독표로 못 읽음 → 그 부분 이미지 확인
  NOALIGN 필드 위치를 못 찾음(한글 6자 미만 등) → 필드 전체 이미지 확인
  PTS     배점 불일치/미발견
  OUTSIDE 어느 문항에도 대응되지 않은 원본 글(30자 이상) → 통째로 빠진 제시문·물음 의심
종료코드: 확인 필요가 없으면 0. 신뢰성 시험: tools/tests/test_source_check.py
"""
from __future__ import annotations

import argparse
import collections
import difflib
import pathlib
import re
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from textnorm import POINTS, hangul, is_pua, math_seq, plain  # noqa: E402

MIN_HANGUL = 6
ANSWER_PAGE = re.compile(r"모범\s*답안|예시\s*답안|채점\s*기준|해설|풀이")
PAGE_NO = re.compile(r"^\s*[-–]\s*\d+\s*[-–]\s*$|^\s*\d+\s*/\s*\d+\s*$")


def read_src(path: str, pages: str | None = None) -> str:
    """.txt 는 그대로, .pdf 는 pdftotext(읽기 순서). pages 'a-b' 가 없으면 2쪽부터
    '모범답안·예시답안·채점기준·해설·풀이' 가 첫머리에 나오는 쪽 직전까지(문제 쪽만)."""
    if not path.lower().endswith(".pdf"):
        return pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    import subprocess
    allp = subprocess.run(["pdftotext", path, "-"], capture_output=True, text=True, check=True).stdout.split("\f")
    if pages:
        a, b = (int(v) for v in pages.split("-"))
    else:
        a, b = 2 if len(allp) > 2 else 1, len(allp)
        for i in range(a, len(allp) + 1):
            if ANSWER_PAGE.search(allp[i - 1][:300]):
                b = i - 1
                break
    return "\f".join(allp[a - 1:b])


def strip_running(raw: str) -> str:
    """쪽 번호 줄과, 2쪽 이상에 똑같이 나오는 머리글·꼬리글 줄을 지운다(쪽 구분 \\f)."""
    pages = raw.split("\f")
    cnt = collections.Counter()
    for pg in pages:
        cnt.update({ln.strip() for ln in pg.splitlines() if ln.strip()})
    rep = {ln for ln, c in cnt.items() if c >= 2 and len(pages) >= 2 and hangul(ln)}
    return "\n".join(ln for ln in raw.splitlines() if ln.strip() not in rep and not PAGE_NO.match(ln))


class Source:
    def __init__(self, raw: str, kind: str):
        self.text = plain(strip_running(raw), kind)
        mask = {i for m in POINTS.finditer(self.text) for i in range(*m.span())}
        self.pos = [i for i, ch in enumerate(self.text) if "가" <= ch <= "힣" and i not in mask]
        self.h = "".join(self.text[i] for i in self.pos)
        self.covered = (0, 0)

    def raw(self, hs: int, he: int) -> str:
        """한글 [hs,he) 와 그 앞뒤 한글 사이의 수식까지."""
        a = self.pos[hs - 1] + 1 if hs > 0 else 0
        b = self.pos[he] if he < len(self.pos) else len(self.text)
        return self.text[a:b]


def locate(src: Source, fh: str) -> tuple[int, int] | None:
    """가장 긴 공통 조각을 닻으로 위치를 잡는다. 닻이 필드의 1/4 미만(최소 6자)이면 None."""
    m = difflib.SequenceMatcher(None, src.h, fh, autojunk=False).find_longest_match(0, len(src.h), 0, len(fh))
    if m.size < max(MIN_HANGUL, len(fh) // 4):
        return None
    slack = max(10, len(fh) // 5)
    s = max(0, m.a - m.b - slack)
    e = min(len(src.h), m.a + (len(fh) - m.b) + slack)
    blocks = [b for b in difflib.SequenceMatcher(None, src.h[s:e], fh, autojunk=False).get_matching_blocks() if b.size]
    return s + blocks[0].a, s + blocks[-1].a + blocks[-1].size


def show(toks) -> str:
    return " ".join(f"⟨{ord(c):04X}⟩" if is_pua(c) else c for c in toks) or "∅"


class Field:
    def __init__(self, item_id: str, label: str, text: str, points=None):
        self.item_id, self.label, self.points = item_id, label, points
        self.plain = POINTS.sub(" ", plain(text, "latex"))
        self.h = hangul(self.plain)
        self.span = None
        self.occ = []

    @property
    def name(self) -> str:
        return f"{self.item_id} {self.label}"


def item_fields(f: pathlib.Path) -> list[Field]:
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    iid = it.get("id", f.stem)
    out = [Field(iid, f"stimuli[{s.get('label')}]",
                 (f"({s['label']}) " if re.fullmatch(r"[가-하]", str(s.get("label") or "")) else "") + (s.get("body") or ""))
           for s in it.get("stimuli") or []]
    out.append(Field(iid, "stem", it.get("stem") or ""))
    out += [Field(iid, f"sub[{s['id']}]", s.get("stem") or "", s.get("points")) for s in it.get("subparts") or []]
    return out


def place(src: Source, fields: list[Field]) -> list[tuple[str, str, str]]:
    """필드 위치 잡기. 원본에 같은 글이 여러 번 나오는 짧은 필드(예: '다음 제시문을 읽고…')는
    같은 문항의 다른 필드와 가장 가까운 곳에 둔다."""
    out, amb = [], []
    for fd in fields:
        if len(fd.h) < MIN_HANGUL:
            out.append(("NOALIGN", fd.name, f"한글 {len(fd.h)}자 — 위치 판정 불가, 필드 전체 확인"))
            continue
        occ = [m.start() for m in re.finditer(re.escape(fd.h), src.h)]
        if len(occ) > 1:
            fd.occ = occ
            amb.append(fd)
            continue
        fd.span = (occ[0], occ[0] + len(fd.h)) if occ else locate(src, fd.h)
        if not fd.span:
            out.append(("NOALIGN", fd.name, "원본에서 위치를 찾지 못함 — 필드 전체 확인"))
    taken = [f.span for f in fields if f.span]
    for fd in amb:
        mates = [f.span[0] for f in fields if f.span and f.item_id == fd.item_id]
        free = [o for o in fd.occ if not any(a < o + len(fd.h) and o < b for a, b in taken)] or fd.occ
        o = min(free, key=lambda v: min((abs(v - m) for m in mates), default=v))
        fd.span = (o, o + len(fd.h))
        taken.append(fd.span)
    return out


def compare(src: Source, fields: list[Field], check_points: bool = True) -> list[tuple[str, str, str]]:
    """반환: (등급, 필드 이름, 설명) 목록."""
    out = place(src, fields)
    placed = sorted((f for f in fields if f.span), key=lambda f: f.span[0])
    if not placed:
        return out
    for a, b in zip(placed, placed[1:]):
        if b.span[0] < a.span[1]:
            out.append(("DIFF", b.name, f"{a.name} 와 원본 구간이 겹침(같은 글을 두 번 입력?)"))
    lo, hi = placed[0].span[0], max(f.span[1] for f in placed)
    src.covered = (lo, hi)

    # 한글: 원본 [lo,hi) 전체 ↔ 필드를 원본 순서로 이어 붙인 것. 한 글자라도 다르면 DIFF.
    db_h, owner_h = "", []
    for f in placed:
        db_h += f.h
        owner_h += [f] * len(f.h)
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, src.h[lo:hi], db_h, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        who = owner_h[min(j1, len(owner_h) - 1)]
        ctx = src.h[lo + max(0, i1 - 6):lo + i1]
        out.append(("DIFF", who.name, f"한글 …{ctx}[원본 '{src.h[lo + i1:lo + i2]}' → DB '{db_h[j1:j2]}']"))

    # 수식: 원본 범위 전체 ↔ 필드 이어 붙인 것. 어긋난 조각을 필드별로 모아
    #   순수 차이(원본에만 − DB에만)가 있으면 DIFF/UNREAD, 기호는 같고 자리만 다르면 ORDER.
    s_seq = math_seq(POINTS.sub(" ", src.raw(lo, hi)))
    d_seq, owner_m = [], []
    for f in placed:
        ms = math_seq(f.plain)
        d_seq += ms
        owner_m += [f] * len(ms)
    per = collections.defaultdict(lambda: [collections.Counter(), collections.Counter(), []])
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, s_seq, d_seq, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        who = owner_m[min(j1, len(owner_m) - 1)] if owner_m else placed[0]
        rec = per["시험지"]          # 기호가 문항 경계를 넘어 추출되는 경우가 있어 시험지 전체로 상쇄
        rec[0].update(s_seq[i1:i2])
        rec[1].update(d_seq[j1:j2])
        rec[2].append((who.name, f"…{show(s_seq[max(0, i1 - 3):i1])} [원본 {show(s_seq[i1:i2])} → DB {show(d_seq[j1:j2])}]"))
    for iid, (sc, dc, pieces) in per.items():
        only_s, only_d = sc - dc, dc - sc          # 문항 전체에서 상쇄되지 않는 기호 = 실제 차이 후보
        if not only_s and not only_d:
            for name, msg in pieces:
                out.append(("ORDER", name, "수식 " + msg))
            continue
        bad = set(only_s) | set(only_d)
        unread = any(is_pua(c) for c in only_s)
        for name, msg in pieces:
            hit = any(t in msg for t in map(show, [[c] for c in bad]))
            g = ("UNREAD" if unread and "⟨" in msg else "DIFF") if hit else "ORDER"
            out.append((g, name, "수식 " + msg))
        out.append(("DIFF" if not unread or only_d else "UNREAD", "(시험지 합계)",
                    f"원본에만 {show(sorted(only_s.elements()))} · DB에만 {show(sorted(only_d.elements()))}"))

    if check_points:
        # 배점: 소문항 글(마지막 한글)보다 앞에 찍힌 'n점' 중 마지막 것이 그 소문항(또는 묶음)의 배점.
        # 원본에 배점이 따로 없는 소문항(공식 채점기준으로 DB 가 더 잘게 나눈 경우)은 앞 묶음에 합쳐
        # '묶음 합 = 원본 배점' 으로 확인한다. 끝에 배점을 적는 양식이면 글 바로 뒤 'n점'도 인정.
        info = []
        for f in (f for f in placed if f.points is not None):
            a = src.pos[f.span[0] - 1] + 1 if f.span[0] > 0 else 0
            z = src.pos[f.span[1] - 1]
            before = [int(v) for v in re.findall(r"(\d+)\s*점", src.text[a:z])]
            after = [int(v) for v in re.findall(r"^[^가-힣]{0,15}?(\d+)\s*점", src.text[z + 1:z + 30])]
            info.append((f, before, after))
        # 양식(배점이 글 앞/뒤)은 시험지마다 다르다: 두 방식으로 다 맞춰 보고 어긋남이 적은 쪽을 쓴다(동률이면 앞)
        def check(tail):
            groups, bad = [], []
            for f, before, after in info:
                own = (after[0] if after else None) if tail else (before[-1] if before else None)
                if own is not None or not groups:
                    groups.append([own, [f]])
                else:
                    groups[-1][1].append(f)
            for target, fs in groups:
                tot = sum(int(f.points) for f in fs)
                if tot != target:
                    bad.append(("PTS", fs[0].name, f"배점 DB {'+'.join(str(f.points) for f in fs)}={tot} "
                                f"({', '.join(f.label for f in fs)}) ≠ 원본 {target if target is not None else '없음'}"))
            return bad
        b0, b1 = check(False), check(True)
        out += b0 if len(b0) <= len(b1) else b1
    return out


def run(src_path: str, items: list[str], pages=None, kind="pdf", points=True, raw_text=None):
    src = Source(raw_text if raw_text is not None else read_src(src_path, pages), kind)
    fields = [fd for p in items for fd in item_fields(pathlib.Path(p))]
    res = compare(src, fields, points)
    lo, hi = src.covered
    for s, e in ((0, lo), (hi, len(src.h))):
        if e - s >= 30:
            res.append(("OUTSIDE", "-", f"원본 한글 {e - s}자가 어느 문항에도 없음: {src.h[s:s + 40]}…"))
    return fields, res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--pages")
    ap.add_argument("--src-kind", default="pdf", choices=["pdf", "hwp", "txt"])
    ap.add_argument("--no-points", action="store_true", help="배점이 원본에 없는 시험지")
    ap.add_argument("--lenient", action="store_true", help="ORDER(같은 기호, 순서만 다름)는 확인 필요에서 뺀다 — 기본은 셈(신뢰성)")
    ap.add_argument("items", nargs="+")
    a = ap.parse_args()
    fields, res = run(a.src, a.items, a.pages, a.src_kind, not a.no_points)
    need = [r for r in res if not a.lenient or r[0] != "ORDER"]
    for g, name, msg in sorted(res, key=lambda r: (r[1], r[0])):
        print(f"{g:7} {name}  {msg}")
    cnt = collections.Counter(g for g, _, _ in res)
    ok_fields = len(fields) - len({n for _, n, _ in need} & {f.name for f in fields})
    print(f"SUMMARY source_check: 필드 {len(fields)}개 중 자동통과 {ok_fields} · " +
          " ".join(f"{k} {v}" for k, v in sorted(cnt.items())))
    return 1 if need else 0


if __name__ == "__main__":
    sys.exit(main())
