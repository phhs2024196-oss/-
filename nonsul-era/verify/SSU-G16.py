#!/usr/bin/env python3
"""숭실대 SSU-G16-01 — sympy 검산. 규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수는 자동 실행된다. 소문항 정답은 yaml 의 subparts[].answer.value 에서 읽는다."""
import re
import sys

import sympy as sp

PREFIX = "SSU-G16"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answers(seq):
    """소문항 id → answer.value(문자열). 증명 소문항은 None."""
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(f"items/SSU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    return {str(s["id"]): s["answer"]["value"] for s in d["subparts"]}


def p01():
    """A: 격자 점에서 신호 세기 비교, B: 삼각형∩{y<x} 넓이를 shapely 없이 다각형 클리핑으로 직접 계산해 F, F'(2), argmax G 비교."""
    ans = load_answers(1)
    # A
    Xs, Ys = sp.symbols("x y", real=True)
    reg = sp.sympify(ans["A"], locals={"x": Xs, "y": Ys})
    import random
    random.seed(2); okA = True
    for _ in range(2000):
        px, py = random.uniform(-10, 16), random.uniform(-10, 10)
        if px*px + py*py < 1e-9 or (px-4)**2 + py*py < 1e-9:
            continue
        okA &= (3/(px*px+py*py) > 1/((px-4)**2+py*py)) == bool(reg.subs({Xs: px, Ys: py}))
    # B: Sutherland–Hodgman 클리핑
    def clip(poly, keep):            # keep(p)->value, 반평면 value<0 유지
        out = []
        for i in range(len(poly)):
            P, Q = poly[i], poly[(i+1) % len(poly)]
            fp, fq = keep(P), keep(Q)
            if fp <= 0: out.append(P)
            if (fp < 0 < fq) or (fq < 0 < fp):
                s = fp / (fp - fq); out.append((P[0] + s*(Q[0]-P[0]), P[1] + s*(Q[1]-P[1])))
        return out
    def area(poly):
        return abs(sum(poly[i][0]*poly[(i+1)%len(poly)][1] - poly[(i+1)%len(poly)][0]*poly[i][1] for i in range(len(poly)))) / 2 if len(poly) > 2 else sp.Integer(0)
    def Fnum(tv):
        tri = [(tv-2, sp.Integer(0)), (tv, sp.Integer(0)), (tv-2, sp.Integer(4))]
        return area(clip(tri, lambda p: p[1] - p[0]))
    def Gnum(tv):
        tri = [(tv-2, sp.Integer(0)), (tv, sp.Integer(0)), (tv-2, sp.Integer(4))]
        return area(clip(clip(tri, lambda p: p[1] - p[0]), lambda p: (p[0] - 2) - p[1]))
    Fmd = sp.sympify(ans["B1"], locals={"t": t})
    okB1 = all(sp.simplify(Fnum(sp.Rational(k, 7)) - Fmd.subs(t, sp.Rational(k, 7))) == 0 for k in range(-14, 60))
    hs = [sp.Rational(1, 10**j) for j in range(3, 7)]
    left = [(Fnum(2-h) - Fnum(2))/(-h) for h in hs]; right = [(Fnum(2+h) - Fnum(2))/h for h in hs]
    okB2 = all(abs(l - sp.Rational(4, 3)) < 1e-2 for l in left) and all(abs(r - sp.Rational(4, 3)) < 1e-2 for r in right)
    # G = F(t)-F(t-2) 확인 + 격자 최대
    okG = all(sp.simplify(Gnum(sp.Rational(k, 5)) - (Fnum(sp.Rational(k, 5)) - Fnum(sp.Rational(k, 5) - 2))) == 0 for k in range(-5, 50))
    grid = [sp.Rational(k, 300) for k in range(0, 2401)]
    tbest = max(grid, key=lambda v: Gnum(v))
    okB3 = okG and abs(tbest - sp.sympify(ans["B3"])) <= sp.Rational(1, 300)
    report(1, okA and okB1 and okB2 and okB3,
           f"A 무작위 2000점 일치={okA} | B1 다각형 넓이 일치={okB1} | B2 좌우 차분몫→4/3 [증명-수치확인]={okB2} | B3 G=F(t)-F(t-2) {okG}, 격자 argmax={tbest} md={ans['B3']}")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    m = re.search(r"-s(\d+)\.py$", __file__)
    label = PREFIX + (f"-s{m.group(1)}" if m else "")
    print(f"SUMMARY {label}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
