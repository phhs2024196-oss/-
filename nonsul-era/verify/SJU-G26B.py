#!/usr/bin/env python3
"""SJU-G26B (세종대 2026학년도 논술고사 자연계열B) — sympy/mpmath 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import math
import re
import sys
from fractions import Fraction

import mpmath as mp
import numpy as np
import sympy as sp

PREFIX = "SJU-G26B"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)
mp.mp.dps = 25


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def md_parts(no, sub):
    out = {}
    for seg in md_answer(no, sub).split(";"):
        k, v = seg.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def golden(fun, lo, hi, maximize=False, it=150):
    sgn = -1 if maximize else 1
    gr = (mp.sqrt(5) - 1) / 2
    a_, b_ = mp.mpf(lo), mp.mpf(hi)
    c_, d_ = b_ - gr * (b_ - a_), a_ + gr * (b_ - a_)
    fc, fd = sgn * fun(c_), sgn * fun(d_)
    for _ in range(it):
        if fc < fd:
            b_, d_, fd = d_, c_, fc
            c_ = b_ - gr * (b_ - a_); fc = sgn * fun(c_)
        else:
            a_, c_, fc = c_, d_, fd
            d_ = a_ + gr * (b_ - a_); fd = sgn * fun(d_)
    return (a_ + b_) / 2


def p01():
    lo, hi = sp.pi / 2, 2 * sp.pi
    w = sp.cos(t / 2)
    # 정의에서 직접: x 위치별 절댓값 풀어 기호 적분
    f_left = sp.integrate((t - x) * w, (t, lo, hi))                       # x <= pi/2
    f_mid = sp.integrate((x - t) * w, (t, lo, x)) + sp.integrate((t - x) * w, (t, x, hi))
    f_right = sp.integrate((x - t) * w, (t, lo, hi))                      # x >= 2pi
    fnum = lambda X: mp.quad(lambda s: abs(X - s) * mp.cos(s / 2), [mp.pi / 2, min(max(X, mp.pi / 2), 2 * mp.pi), 2 * mp.pi])
    agree = all(abs(float(e.subs(x, v)) - float(fnum(v))) < 1e-12 for e, v in
                [(f_left, -1.0), (f_mid, 2.0), (f_mid, 5.5), (f_right, 7.0)])
    m1 = md_parts(1, "1")
    f0 = sp.simplify(f_left.subs(x, lo))
    fp = sp.simplify(sp.diff(f_left, x))
    ok1 = agree and sp.simplify(f0 - sp.sympify(m1["f0"])) == 0 and sp.simplify(fp - sp.sympify(m1["fp"])) == 0
    # (1-2) 최댓점 수치 탐색
    fm = sp.lambdify(x, f_mid, "mpmath")
    grid = [mp.pi / 2 + (3 * mp.pi / 2) * i / 600 for i in range(601)]
    g0 = max(grid, key=fm)
    al = golden(fm, g0 - mp.pi / 400, g0 + mp.pi / 400, maximize=True)
    ok2 = abs(mp.cos(al / 2) - sp.N(sp.sympify(md_answer(1, "2")), 25)) < 1e-9 and fm(al) > float(f_left.subs(x, lo)) and fm(al) > float(f_right.subs(x, 7))
    # (1-3) H(x)=f(x)-f(pi/2)-sqrt2 x-k 의 근 개수를 세고 md 집합과 대조
    F = lambda X: (f_left if X <= np.pi / 2 else (f_mid if X <= 2 * np.pi else f_right))
    fl = {id(e): sp.lambdify(x, e) for e in (f_left, f_mid, f_right)}
    Fv = lambda X: fl[id(F(X))](X) - float(f0)
    left_const = Fv(-3.0) - math.sqrt(2) * -3.0
    assert abs(left_const - (Fv(0.5) - math.sqrt(2) * 0.5)) < 1e-12           # 왼쪽은 기울기 sqrt2 직선
    xs = np.linspace(np.pi / 2, 40, 200001)
    Fx = np.array([Fv(X) for X in xs])
    def count(k):
        if abs(left_const - k) < 1e-12:
            return math.inf
        H = Fx - math.sqrt(2) * xs - k
        c = int(np.sum(np.sign(H[:-1]) * np.sign(H[1:]) < 0))
        if c == 0 and abs(H.max()) < 1e-6:
            c = 1                                                            # 접함
        return c
    s = md_answer(1, "3").replace(" ", "")
    pt, iv = s.split("U")
    kstar = float(sp.sympify(pt.strip("{}")))
    kbound = float(sp.sympify(iv.strip("()").split(",")[1]))
    in_md = lambda k: abs(k - kstar) < 1e-9 or k < kbound - 1e-12
    ks = [-8 + 0.0131 * i for i in range(900)] + [kstar, kbound]
    bad = [k for k in ks if (count(k) == 1) != in_md(k)]
    # 접선 값 kstar 에서 H 의 최댓값이 0 인지 정밀 확인
    Hm = golden(lambda X: fm(X) - f0 - mp.sqrt(2) * X - kstar, mp.pi, 2 * mp.pi, maximize=True)
    tang = abs(fm(Hm) - f0 - mp.sqrt(2) * Hm - kstar) < 1e-9
    ok3 = not bad and tang
    report(1, ok1 and ok2 and ok3, f"f(pi/2)={f0} f'={fp} alpha={float(al):.10f} k*={kstar:.10f} (tangent x={float(Hm):.8f}) mismatch={bad[:4]}")


def extremum_info(a, b, c):
    """g = -x/2 (x<=0), a x^3+b x^2+c x (x>0) 의 극값을 갖는 x 목록."""
    X = sp.symbols("X")
    fp = sp.Poly(3 * a * X**2 + 2 * b * X + c, X)
    pts = []
    # x=0 오른쪽 부호
    eps_sign = next((sp.sign(co) for co in reversed(fp.all_coeffs()) if co != 0), 0)
    if eps_sign > 0:
        pts.append(sp.Integer(0))                                          # 감소→증가
    for r, mult in sp.roots(fp, X).items():
        if r.is_real and r > 0 and mult % 2 == 1:
            pts.append(r)
    return pts


def p02():
    # (2-1) (가)에서 a=±1, (다)로 c 결정, b 를 훑어 (나)를 만족하는 a 를 찾는다
    ok_a = set()
    bs = [Fraction(i, 8) for i in range(-80, 400)]
    for a in (1, -1):
        for b in bs:
            b = sp.Rational(b.numerator, b.denominator)
            c = 2 * (sp.Rational(-1, 4) - sp.Rational(a, 4) - b / 3)
            if len(extremum_info(a, b, c)) == 1:
                ok_a.add(a)
                break
    ok1 = ok_a == {int(md_answer(2, "1"))}
    # (2-2) a=1, 유효한 b 에서 alpha 수집
    alphas, valid = [], []
    for b in [sp.Rational(i, 16) for i in range(-80, 1600)] + [sp.Integer(10)**4]:
        c = 2 * (sp.Rational(-1, 4) - sp.Rational(1, 4) - b / 3)
        e = extremum_info(1, b, c)
        if len(e) == 1:
            alphas.append(e[0]); valid.append(b)
    s = md_answer(2, "2").replace(" ", "")
    lo = sp.sympify(s[1:-1].split(",")[0]); hi = sp.sympify(s[1:-1].split(",")[1])
    lo_open, hi_closed = s[0] == "(", s[-1] == "]"
    amin, amax = min(alphas, key=float), max(alphas, key=float)
    ok2 = lo_open and hi_closed and all(float(a_) > float(lo) for a_ in alphas) and sp.simplify(amax - hi) == 0 \
        and float(amin) - float(lo) < 1e-3 and min(valid) == sp.Rational(-3, 2)
    # (2-3) 각 유효 b 에서 h=|g-f'(0)x| 의 미분불가능점 개수
    def nondiff(b):
        c = 2 * (sp.Rational(-1, 4) - sp.Rational(1, 4) - b / 3)
        cnt = 0
        left_coef = sp.Rational(-1, 2) - c                                 # p(x)=(−1/2−c)x (x<=0)
        if left_coef != 0:
            cnt += 1                                                      # 좌미분 −|·| ≠ 우미분 0
        if b < 0:
            cnt += 1                                                      # p=x^2(x+b) 의 단순근 x=-b
        return cnt
    s3 = md_answer(2, "3").replace(" ", "")
    iv, pt = s3.split("U")
    lo3, hi3 = [sp.sympify(v) for v in iv.strip("(]").split(",")]
    p3 = sp.sympify(pt.strip("{}"))
    in_md = lambda a_: (float(lo3) < float(a_) <= float(hi3) + 1e-15) or sp.simplify(a_ - p3) == 0
    extra = [sp.Rational(-3, 4), sp.Integer(0)]                          # 경계·특수값
    test = list(zip(valid, alphas)) + [(b, extremum_info(1, b, 2 * (sp.Rational(-1, 2) - b / 3))[0]) for b in extra]
    bad = [(b, a_) for b, a_ in test if (nondiff(b) == 1) != in_md(a_)]
    ok3 = not bad
    report(2, ok1 and ok2 and ok3, f"a={ok_a} alpha in ({float(amin):.5f}..{float(amax)}] n_valid_b={len(valid)} mismatch={bad[:3]}")


def p03():
    def I(nn, mm):                                    # ∫_0^1 x^n (1-x)^m dx (이항 전개)
        return sum(Fraction(math.comb(mm, j) * (-1)**j, nn + j + 1) for j in range(mm + 1))
    a = {k: Fraction(2**k) * I(k, 102 - k) for k in range(1, 102)}
    S = a[98] / 2**98 + 3 * a[99] / 2**99 + 3 * a[100] / 2**100 + a[101] / 2**101
    m1 = md_parts(3, "1")
    ok1 = S == Fraction(int(sp.sympify(m1["q"])), int(sp.sympify(m1["p"])))
    ns = [k for k in range(1, 101) if a[k] / a[k + 1] >= 1]
    s2 = md_answer(3, "2")
    ints = [int(v) for v in re.findall(r"\d+", s2)]
    ok2 = ns == list(range(ints[0], ints[-1] + 1))
    m = min(a, key=lambda k: a[k])
    val = math.comb(102, m) * a[m]
    m3 = md_parts(3, "3")
    ok3 = m == int(m3["m"]) and val == Fraction(int(sp.sympify(m3["q"])), int(sp.sympify(m3["p"])))
    report(3, ok1 and ok2 and ok3, f"S={S} n=1..{ns[-1]} m={m} C*a_m={val}")


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
