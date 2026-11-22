#!/usr/bin/env python3
"""HIU-G23A (홍익대 2023학년도 수시 논술 자연계열 서울 오전) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys
from fractions import Fraction
from math import floor

import sympy as sp

PREFIX = "HIU-G23A"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HIU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def kv(s: str) -> dict:
    return {a.strip(): b.strip() for a, b in (seg.split("=", 1) if "=" in seg else seg.split(":", 1)
                                              for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    x, y = sp.symbols("x y", real=True)
    # (1) 접기: 아랫변 위의 점 Q=(q,-1)을 O에 겹치면 접힌 선 l3 는 OQ 의 수직이등분선, l2 는 x=q.
    q = sp.symbols("q", real=True)
    Py = sp.solve(sp.Eq((q - 0) ** 2 + (y - 0) ** 2, (q - q) ** 2 + (y + 1) ** 2), y)[0]  # P 는 l3 위 → |PO|=|PQ|
    # P=(q,Py) 가 O 와 직선 y=-1 까지 같은 거리인지 확인
    ok_eq = sp.simplify(sp.sqrt(q**2 + Py**2) - sp.Abs(Py + 1)).subs(q, sp.Rational(3, 7)) == 0
    d1 = md_answer(1, "1").split(";")
    focus = sp.sympify(d1[0].split(":", 1)[1])
    lhs, rhs = d1[1].split(":", 1)[1].split("=")
    directrix = sp.sympify(lhs, locals={"y": y}) - sp.sympify(rhs, locals={"y": y})               # y+1
    ok1 = ok_eq and tuple(focus) == (0, 0) and sp.simplify(directrix - (y + 1)) == 0
    # (2) 초점 O, 준선 네 변인 포물선 집합: x^2+y^2 = (거리)^2
    loci = [x**2 + y**2 - (y + 1) ** 2, x**2 + y**2 - (x - 1) ** 2, x**2 + y**2 - (x + 1) ** 2, x**2 + y**2 - (y - 1) ** 2]
    eqs = []
    for seg in md_answer(1, "2").split(";"):
        a, b = seg.split("=")
        eqs.append(sp.expand(sp.sympify(a, locals={"x": x, "y": y}) - sp.sympify(b, locals={"x": x, "y": y})))
    ok2 = len(eqs) == 4 and all(any(sp.simplify(e / L).is_number for e in eqs) for L in loci)
    # (3) 영역 R = 네 포물선 안쪽(O 가 각 변보다 가까운 점). 1사분면 넓이 ×4 를 수치 적분
    import mpmath as mp
    mp.mp.dps = 30
    def top(xv):   # 1사분면에서 y 상한: y<=(1-x^2)/2 그리고 x<=(1-y^2)/2 → y<=sqrt(1-2x)
        return min((1 - xv**2) / 2, mp.sqrt(max(1 - 2 * xv, 0)))
    xc = mp.findroot(lambda t: (1 - t**2) / 2 - mp.sqrt(1 - 2 * t), 0.4)
    area = 4 * (mp.quad(lambda t: (1 - t**2) / 2, [0, xc]) + mp.quad(lambda t: mp.sqrt(1 - 2 * t), [xc, 0.5]))
    area = mp.re(area)
    ok3 = abs(area - mp.mpf(str(sp.N(sp.sympify(md_answer(1, "3")), 30)))) < 1e-20
    report(1, ok1 and ok2 and ok3, f"focus={tuple(focus)} parabolas ok={ok2} area={mp.nstr(area, 15)}")


def p02():
    def err(m, p):
        return sum(sp.binomial(m, k) * p**k * (1 - p) ** (m - k) for k in range((m + 1) // 2, m + 1))
    p1, p2 = sp.Rational(1, 10), sp.Rational(2, 10)
    e1, e2 = err(3, p1), err(3, p2)
    ok1 = sp.nsimplify(md_answer(2, "1")) == e1
    d = kv(md_answer(2, "2"))
    ratio = round(float(e2 / e1), 1)
    ok2 = sp.nsimplify(d["prob"]) == e2 and abs(float(d["ratio"]) - ratio) < 1e-12
    mmin = next(m for m in range(3, 101, 2) if err(m, p2) < e1)
    ok3 = int(md_answer(2, "3")) == mmin
    report(2, ok1 and ok2 and ok3,
           f"e1={e1} e2={e2} ratio={float(e2/e1):.4f} err(5,7,9)={[float(err(m,p2)) for m in (5,7,9)]} m_min={mmin}")


def _brute(fmin_max, a, b, n):
    """정사각형을 직접 세기: 열 [i/n,(i+1)/n](양옆 포함) × 행 [j/n,(j+1)/n)(아래 포함, 위 제외)."""
    s = t = 0
    for i in range(n * a, n * b):
        lo, hi = fmin_max(Fraction(i, n), Fraction(i + 1, n))
        for j in range(0, int(hi * n) + 3):
            y0, y1 = Fraction(j, n), Fraction(j + 1, n)
            meet = lo < y1 and hi >= y0          # [lo,hi] ∩ [y0,y1) ≠ ∅
            if meet:
                s += 1
            elif y1 <= lo:                        # 그래프 아래, x축과 곡선 사이
                t += 1
    return s, t


def p03():
    k, n_, a_, b_ = sp.symbols("k n a b", integer=True)
    # (1) [nc]/n 이 c 이하 최대 분수인지 무작위 확인
    import random
    random.seed(1)
    expr1 = sp.sympify(md_answer(3, "1"), locals={"c": sp.Symbol("c"), "n": sp.Symbol("n")})
    ok1 = True
    for _ in range(300):
        c = Fraction(random.randint(-5000, 5000), random.randint(1, 97))
        nn = random.randint(1, 30)
        best = max(Fraction(m, nn) for m in range(floor(c * nn) - 3, floor(c * nn) + 4) if Fraction(m, nn) <= c)
        val = expr1.subs({sp.Symbol("c"): sp.Rational(c.numerator, c.denominator), sp.Symbol("n"): nn})
        ok1 &= sp.Rational(best.numerator, best.denominator) == val
    # 꺾은선 함수의 열별 최소·최대 (꼭짓점 포함, 정확한 유리수)
    def mk(f, knots):
        def mm(x0, x1):
            pts = [x0, x1] + [kx for kx in knots if x0 < kx < x1]
            vals = [f(p) for p in pts]
            return min(vals), max(vals)
        return mm
    fx = lambda xv: xv                                           # 예시 f(x)=x on [0,1]
    f3 = lambda xv: 2 * xv if xv < 1 else -xv / 2 + Fraction(5, 2)   # (3)의 f on [0,3]
    ok_ex = _brute(mk(fx, []), 0, 1, 2) == (4, 1) and _brute(mk(fx, []), 0, 1, 3) == (6, 3)
    # (2) md 공식(M(k), m(k) 함수 기호)을 구체 함수에 대입해 직접 센 값과 비교
    d2 = kv(md_answer(3, "2"))
    Mf, mf = sp.Function("M"), sp.Function("m")
    loc = {"M": Mf, "m": mf, "n": n_, "a": a_, "b": b_, "k": k}
    S2, T2 = sp.sympify(d2["s"], locals=loc), sp.sympify(d2["t"], locals=loc)
    def apply(F, f, knots, a, b, nn):
        mm = mk(f, knots)
        def Mk(kk):
            lo, hi = mm(Fraction(a * nn + kk - 1, nn), Fraction(a * nn + kk, nn))
            return hi
        def mk_(kk):
            lo, hi = mm(Fraction(a * nn + kk - 1, nn), Fraction(a * nn + kk, nn))
            return lo
        total = 0
        G = F.subs({n_: nn, a_: a, b_: b})
        summand, (kv_, lo_, hi_) = G.args[0], G.args[1]
        for kk in range(int(lo_), int(hi_) + 1):
            Mv, mv = Mk(kk), mk_(kk)
            e = summand.subs(kv_, kk).replace(Mf, lambda z: sp.Rational(Mv.numerator, Mv.denominator))
            e = e.replace(mf, lambda z: sp.Rational(mv.numerator, mv.denominator))
            total += int(e)
        return total
    ok2 = True
    fv = lambda xv: abs(3 * xv - 2) + Fraction(1, 3)            # 추가 표본(꺾은선, 꼭짓점 x=2/3)
    for (f, knots, a, b) in [(fx, [], 0, 1), (f3, [1], 0, 3), (fv, [Fraction(2, 3)], 0, 2)]:
        for nn in (1, 2, 3, 5, 8):
            bs, bt = _brute(mk(f, knots), a, b, nn)
            ok2 &= apply(S2, f, knots, a, b, nn) == bs and apply(T2, f, knots, a, b, nn) == bt
    # (3) 직접 센 s(n), t(n) 와 md 다항식 비교 (n=1..40)
    d3 = kv(md_answer(3, "3"))
    n = sp.Symbol("n")
    sP, tP = sp.sympify(d3["s"]), sp.sympify(d3["t"])
    ok3 = all(_brute(mk(f3, [1]), 0, 3, nn) == (int(sP.subs(n, nn)), int(tP.subs(n, nn))) for nn in range(1, 41))
    # (4) 극한 + 넓이(정적분) + 길이
    x = sp.Symbol("x")
    d4 = kv(md_answer(3, "4"))
    L1, L2, L3 = sp.limit(sP / n**2, n, sp.oo), sp.limit(tP / n**2, n, sp.oo), sp.limit(sP / n, n, sp.oo)
    area = sp.integrate(2 * x, (x, 0, 1)) + sp.integrate(-x / 2 + sp.Rational(5, 2), (x, 1, 3))
    length = sp.integrate(sp.sqrt(1 + 2**2), (x, 0, 1)) + sp.integrate(sp.sqrt(1 + sp.Rational(1, 4)), (x, 1, 3))
    ok4 = (sp.sympify(d4["lim_s_n2"]) == L1 and sp.sympify(d4["lim_t_n2"]) == L2 and sp.sympify(d4["lim_s_n"]) == L3
           and sp.sympify(d4["area"]) == area == L2 and sp.simplify(sp.sympify(d4["length"]) - length) == 0
           and sp.simplify(length - L3) != 0)
    report(3, ok1 and ok_ex and ok2 and ok3 and ok4,
           f"example s,t ok={ok_ex} formula ok={ok2} s,t poly ok(n<=40)={ok3} lims=({L1},{L2},{L3}) area={area} length={length}")


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
