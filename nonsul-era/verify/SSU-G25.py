#!/usr/bin/env python3
"""숭실대 2025학년도 논술고사(자연) 1~4 (SSU-G25-01~04) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "SSU-G25"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
a, b, k = sp.symbols("a b k", real=True)
SYM = {"x": x, "t": t, "n": n, "a": a, "b": b, "k": k}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")




def load_answer(seq, sub=0, raw=False):
    """yaml 의 subparts[sub].answer.value 를 읽는다(raw=True 면 문자열 그대로)."""
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/SSU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    v = d["subparts"][sub]["answer"]["value"]
    return v if raw else sp.sympify(v, locals=SYM)


def p01():
    """g: 최고차 1, 정수 계수, 최솟값 조건 / h: 근이 모두 1..10 정수 — 전수 열거 후 곱 f 를 다항식으로 중복 제거."""
    import itertools
    gs = []
    for deg_g in (1, 2, 3):                     # deg h = 4 - deg_g >= 1
        if deg_g % 2 == 1:
            continue                            # 홀수 차수 다항식은 최솟값 없음
        for p, q in itertools.product(range(-20, 21), repeat=2):
            g = x**2 + p * x + q
            bb = sp.Rational(-p, 2)
            if 0 <= bb <= 1 and 1 <= g.subs(x, bb) <= 2:
                gs.append(g)
    hs = [sp.expand((x - r) * (x - s)) for r in range(1, 11) for s in range(r, 11)]
    fs = {tuple(sp.Poly(sp.expand(g * h), x).all_coeffs()) for g in gs for h in hs}
    ans = load_answer(1)
    report(1, len(fs) == ans, f"g {len(gs)}개, h {len(hs)}개, 서로 다른 f {len(fs)}개, md={ans} (오답 재현: 중근 누락 {len(gs)*45})")


def p02():
    """f=e^{-x}|cos x|, 현 AB·BC 와 곡선 사이 넓이를 정적분으로 직접 계산(위아래는 격자에서 판정)."""
    f1 = sp.exp(-x) * sp.cos(x)            # [0, pi/2]
    f2 = -sp.exp(-x) * sp.cos(x)           # [pi/2, pi]
    A, B, C = (0, f1.subs(x, 0)), (sp.pi / 2, 0), (sp.pi, f2.subs(x, sp.pi))
    line = lambda P, Q: P[1] + (Q[1] - P[1]) / (Q[0] - P[0]) * (x - P[0])
    l1, l2 = line(A, B), line(B, C)
    pts = [sp.pi / 2 * i / 200 for i in range(1, 200)]
    up1 = all((l1 - f1).subs(x, p).evalf() >= 0 for p in pts)
    up2 = all((f2 - l2).subs(x, sp.pi / 2 + p).evalf() >= 0 for p in pts)
    S = sp.integrate(l1 - f1, (x, 0, sp.pi / 2)) + sp.integrate(f2 - l2, (x, sp.pi / 2, sp.pi))
    ans = load_answer(2)
    ok = up1 and up2 and sp.simplify(S - ans) == 0
    report(2, ok, f"현 위치 판정 {up1},{up2}, 넓이={sp.simplify(S)} ≈ {sp.N(S, 12)}, md={ans}")


def p03():
    """(1) x=-1 에서 좌우극한 일치 방정식을 풀고, (2) b 격자에서 f(g(x)+3) 연속 ⟺ min g ≥ -4 를 직접 판정."""
    A_, C_ = sp.symbols("A C", real=True)
    gL = lambda X: X**2 + A_ * X + C_
    gR = lambda X: X**2 + b * X + C_
    g = lambda X: gL(X) if X < 0 else gR(X)
    eq1 = sp.Eq((-1) * gL(-1), 2 * gL(-1))          # f(x)g(x): 좌 f=-1, 우 f=2
    eq2 = sp.Eq(gL(-1), gR(2))                       # g(f(x)): 좌 g(-1), 우 g(2)
    sol = sp.solve([eq1, eq2], [A_, C_], dict=True)[0]
    ans1 = load_answer(3, 0)
    ok1 = sp.simplify(sol[A_] - ans1[0]) == 0 and sp.simplify(sol[C_] - ans1[1]) == 0

    def cont_ok(bv):
        # g 의 최솟값(두 조각 각각 꼭짓점 또는 x=0)을 정확히 계산
        av, cv = sol[A_].subs(b, bv), sol[C_].subs(b, bv)
        cands = [sp.Integer(0)]
        vl, vr = -av / 2, -sp.Rational(bv) / 2
        if vl < 0:
            cands.append(vl)
        if vr > 0:
            cands.append(vr)
        m = min((X**2 + (av if X < 0 else bv) * X + cv) for X in cands)
        return m >= -4                               # g+3 이 -1 아래로 내려가지 않음
    ans2 = load_answer(3, 1)
    grid = [sp.Rational(i, 40) for i in range(-400, 200)]
    mism = [bv for bv in grid if cont_ok(bv) != bool(ans2.contains(bv))]
    report(3, ok1 and not mism, f"(1) a={sol[A_]}, c={sol[C_]} md={ans1}; (2) 격자 {len(grid)}점 불일치 {len(mism)}, md={ans2}")


def p04():
    """h ≥ 3 + 54/r^2 제약에서 겉넓이 최소: 기호 최소 + 2변수 격자 탐색."""
    r, h = sp.symbols("r h", positive=True)
    A = sp.pi * r**2 + 2 * sp.pi * r * (3 + 54 / r**2)
    crit = [c for c in sp.solve(sp.diff(A, r), r) if c.is_positive]
    Amin = min(A.subs(r, c) for c in crit)
    import math
    best = min(math.pi * rv**2 + 2 * math.pi * rv * hv
               for rv in [0.5 + 0.01 * i for i in range(800)]
               for hv in [3 + 54 / rv**2 + 0.05 * j for j in range(5)])
    ans = load_answer(4)
    ok = sp.simplify(Amin - ans) == 0 and abs(best - float(ans)) < 1e-3 * float(ans)
    report(4, ok, f"임계점 r={crit}, 최솟값={Amin}, 격자 최소≈{best:.5f}, md={ans} (오답 재현: 제약 무시 최소 {sp.nsimplify(sp.minimum(sp.pi*r**2+108*sp.pi/r, r, sp.Interval.open(0, sp.oo)))})")


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
