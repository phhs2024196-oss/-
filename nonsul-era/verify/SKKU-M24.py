#!/usr/bin/env python3
"""SKKU-M24 (성균관대 2024학년도 모의논술 자연계) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys
from fractions import Fraction

import sympy as sp

PREFIX = "SKKU-M24"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SKKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    k = sp.symbols("k", real=True)
    al = sp.symbols("alpha", real=True)
    up = x - x**2
    ell = sp.diff(up, x).subs(x, k) * (x - k) + up.subs(x, k)
    # (i) k=0
    l0 = ell.subs(k, 0)
    q0 = max(r for r in sp.solve(sp.Eq(l0, x**2 - x), x) if r > 0)
    A0 = sp.integrate(l0 - up, (x, 0, 1))          # 곡선 y=x-x^2, 선분 PQ(=y=x 위), x=1
    B0 = sp.integrate(l0 - (x**2 - x), (x, 1, q0))
    a_, b_ = md_answer(1, "1").split(":")
    ok1 = sp.simplify(A0 / B0 - sp.Rational(a_) / sp.Rational(b_)) == 0
    # (ii) alpha = 양의 교점, alpha>1, B(k)
    roots = sp.solve(sp.Eq(ell, x**2 - x), x)
    alpha = [r for r in roots if r.subs(k, sp.Rational(1, 3)) > 0][0]
    okgt1 = all(alpha.subs(k, v) > 1 for v in [0, sp.Rational(1, 7), sp.Rational(1, 2), sp.Rational(9, 10)])
    Bk = sp.integrate(ell - (x**2 - x), (x, 1, al))          # alpha 를 기호로 두고 적분
    md2 = sp.sympify(md_answer(1, "2"), locals={"k": k, "alpha": al})
    ok2 = sp.expand(Bk - md2) == 0 and okgt1
    # (iii) k=1/n 대입 → C + D sqrt(n^2-2n+2), D>0
    okd = True
    Dsym = sp.sympify(md_answer(1, "3").split("=", 1)[1], locals={"n": sp.Symbol("n")})
    for nv in range(2, 12):
        kv = sp.Rational(1, nv)
        s = sp.sqrt(nv**2 - 2 * nv + 2)
        av = sp.nsimplify(alpha.subs(k, kv))
        val = sp.expand(Bk.subs({al: av, k: kv}))
        r = sp.Symbol("r")
        poly = sp.expand(val.subs(s, r)) if s.is_Symbol else None
        # 계수 추출: val = C + D*s  (s 가 무리수가 아닐 수도 있으므로 직접 분해)
        valr = sp.expand(sp.expand(Bk.subs({al: 1 - kv + kv * r, k: kv})))
        valr = sp.Poly(valr, r)
        rem = valr.rem(sp.Poly(r**2 - (nv**2 - 2 * nv + 2), r))
        cs = rem.all_coeffs()
        D = cs[0] if len(cs) == 2 else 0
        C = cs[-1]
        okd &= (D.is_rational and C.is_rational and D > 0
                and sp.simplify(D - Dsym.subs("n", nv)) == 0
                and sp.simplify(C + D * s - val) == 0)
    report(1, ok1 and ok2 and okd, f"A0={A0} B0={B0} alpha>1:{okgt1} D(n=2..11) 검사={okd}")


def make_f(A, B):
    """<제시문 3>의 f (A>1): x 를 [1,A) 로 축소해 계산."""
    A = sp.nsimplify(A); B = sp.nsimplify(B)

    def f(v):
        v = sp.nsimplify(v)
        s = 0
        while v >= A:
            v /= A; s += 1
        while v < 1:
            v *= A; s -= 1
        return A**s * (B - 2 * B / (A - 1) * sp.Abs(v - (A + 1) / 2))
    return f


def integ_f(A, B, lo, hi):
    """조각마다 선형이므로 꺾이는 점(A^k, 중점)을 모아 사다리꼴로 정확 적분."""
    f = make_f(A, B)
    A = sp.nsimplify(A)
    pts = {sp.nsimplify(lo), sp.nsimplify(hi)}
    for kk in range(-10, 10):
        for p in (A**kk, A**kk * (A + 1) / 2):
            if lo < p < hi:
                pts.add(p)
    pts = sorted(pts, key=lambda t: float(t))
    return sp.nsimplify(sum((pts[i + 1] - pts[i]) * (f(pts[i]) + f(pts[i + 1])) / 2 for i in range(len(pts) - 1)))


def p02():
    # (i)
    v1 = integ_f(3, 1, sp.Rational(2, 3), 9)
    ok1 = sp.simplify(v1 - sp.Rational(md_answer(2, "1"))) == 0
    # (ii) g∘f 의 극소점 개수: 1<=a<=100 에서 촘촘한 격자로 직접 판정
    import numpy as np

    def fnum(v):
        s = 0
        while v >= 2:
            v /= 2; s += 1
        while v < 1:
            v *= 2; s -= 1
        return 2**s * (0.5 - abs(v - 1.5))
    g = lambda y: y**3 - 9 * y**2 + 18 * y
    xs = np.arange(20000, 4040001) / 40000.0   # 이진 분수 격자(2^k 정확히 포함)
    hv = np.array([g(fnum(t)) for t in xs])
    cnt = 0
    for i in range(1, len(xs) - 1):
        if 1 <= xs[i] <= 100 and hv[i] <= hv[i - 1] and hv[i] < hv[i + 1] or \
           (1 <= xs[i] <= 100 and hv[i] < hv[i - 1] and hv[i] <= hv[i + 1]):
            cnt += 1
    ok2 = cnt == int(md_answer(2, "2"))
    # (iii) 정수 P,Q 를 전수 탐색해 조건 만족하는 (A,B) 집합을 구성
    sols = []
    for P in range(-60, 61):
        for Q in range(1, 40):                         # A>1 → Q=2log3A>0
            if 6 * P + 15 * Q != 117:
                continue
            A = sp.sqrt(3)**Q; a1 = sp.Integer(3)**P
            tot = sum(a1 * (A**2)**n for n in range(6))
            L = sp.log(tot, 3)
            if 37 < float(L) < 38:
                B = 2 * a1 / (A - 1)
                # 정의로 a_n 재계산 확인
                an = [integ_f(A, B, A**(n - 1), A**n) for n in (1, 2)]
                assert sp.simplify(an[0] - a1) == 0 and sp.simplify(an[1] / an[0] - A**2) == 0
                sols.append((sp.nsimplify(A), sp.nsimplify(B)))
    md = sp.sympify(md_answer(2, "3"))
    ok3 = len(sols) == 1 and sp.simplify(sols[0][0] - md[0]) == 0 and sp.simplify(sols[0][1] - md[1]) == 0
    report(2, ok1 and ok2 and ok3, f"int={v1} 극소개수={cnt} sols={sols}")


def p03():
    fr = Fraction(54484, 333333)
    digs = []
    r = fr.numerator % fr.denominator
    for _ in range(40):
        r *= 10; digs.append(r // fr.denominator); r %= fr.denominator
    a = lambda n: digs[n - 1]
    fn = lambda n: a(n) - a(n + 4)
    parts = {kk.strip(): vv for kk, vv in (seg.split("=", 1) for seg in md_answer(3, "1").split(";"))}
    ok1 = [a(n) for n in range(1, 21)] == [eval(parts["a"])[(n - 1) % 6] for n in range(1, 21)] and \
        [fn(n) for n in range(1, 21)] == [eval(parts["f"])[(n - 1) % 6] for n in range(1, 21)]
    g = x - x**2

    def h(v):
        v = sp.nsimplify(v)
        while v >= 7:
            v -= 6
        m = int(sp.floor(v))
        return fn(m) * g.subs(x, v - m), fn(m), m
    # (ii) 이음점 x=2..99 에서 좌·우 미분계수 비교
    nd = 0
    for j in range(2, 100):
        _, cl, ml = h(j - sp.Rational(1, 2)); _, cr, mr = h(j + sp.Rational(1, 2))
        left = cl * sp.diff(g, x).subs(x, 1); right = cr * sp.diff(g, x).subs(x, 0)
        nd += left != right
    ok2 = nd == int(md_answer(3, "2"))
    # (iii) 각 단위구간 적분 합
    tot = sum(fn(((j - 1) % 6) + 1) * sp.integrate(g, (x, 0, 1)) for j in range(1, 100))
    ok3 = sp.simplify(tot - sp.Rational(md_answer(3, "3"))) == 0
    report(3, ok1 and ok2 and ok3, f"digits={digs[:6]} 미분불가={nd} 적분={tot}")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
