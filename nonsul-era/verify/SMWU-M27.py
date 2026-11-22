#!/usr/bin/env python3
"""SMWU-M27 (숙명여대 2027학년도 모의논술 자연계열(일반)) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import random
import re
import sys

import numpy as np
import sympy as sp

PREFIX = "SMWU-M27"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)


def md_item(no: int):
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SMWU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    return yaml.safe_load(f.read_text(encoding="utf-8"))


def md_answer(no: int, sub: str):
    return next(s["answer"] for s in md_item(no)["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # [증명-수치확인] 두 소문항 모두 proof
    ok0 = md_answer(1, "1")["kind"] == "proof" and md_answer(1, "2")["kind"] == "proof"
    # (1-1) sin t <= t (t>=0): g=t-sin t, g(0)=0, g'=1-cos t>=0
    t = sp.symbols("t", nonnegative=True)
    g = t - sp.sin(t)
    ok_a = g.subs(t, 0) == 0 and all(float(sp.diff(g, t).subs(t, v)) >= 0 for v in np.linspace(0, 10, 1001))
    xs = np.linspace(np.pi / 2, np.pi, 1001)
    ok_b = all(v**2 * np.sin(1 / v) <= v + 1e-15 for v in xs)
    bound = sp.integrate(x, (x, sp.pi / 2, sp.pi))
    ok_c = sp.simplify(bound - 3 * sp.pi**2 / 8) == 0
    I1 = sp.Integral(x**2 * sp.sin(1 / x), (x, sp.pi / 2, sp.pi)).evalf(30)
    ok1 = ok_a and ok_b and ok_c and I1 <= bound.evalf(30)
    # (1-2) x^3/(x^6+x^3+1) = 1/(x^3+1/x^3+1) <= 1/3, 등호는 x=1
    h = x**3 / (x**6 + x**3 + 1)
    ok_d = sp.simplify(h - 1 / (x**3 + 1 / x**3 + 1)) == 0
    ok_e = all(float(h.subs(x, v)) <= 1 / 3 + 1e-15 for v in np.linspace(1, 2, 1001))
    ok_f = sp.simplify(sp.integrate(sp.Rational(1, 3), (x, 1, 2)) - sp.Rational(1, 3)) == 0
    I2 = sp.Integral(h, (x, 1, 2)).evalf(30)
    ok2 = ok_d and ok_e and ok_f and I2 <= sp.Rational(1, 3)
    report(1, ok0 and ok1 and ok2, f"[증명-수치확인] I1={float(I1):.6f}<=3pi^2/8={float(bound):.6f}, I2={float(I2):.6f}<=1/3")


def p02():
    a, d = sp.symbols("a d", real=True)
    n = sp.symbols("n", integer=True, positive=True)
    bn = sp.exp(a + (n - 1) * d)
    ratio = sp.simplify(bn.subs(n, n + 1) / bn)               # 공비
    # (2-1) 수렴 조건: 0<ratio<1  ⇔ d<0
    dcond = sp.solve_univariate_inequality(ratio < 1, d, relational=True)
    cond_s, sum_s = [p.strip() for p in md_answer(2, "1")["value"].split(";")]
    md_cond = sp.sympify(cond_s, locals={"d": d})
    md_sum = sp.sympify(sum_s.split("=", 1)[1].strip(), locals={"a": a, "d": d})
    ok1a = sp.solveset(dcond, d, sp.S.Reals) == sp.solveset(md_cond, d, sp.S.Reals)
    S = sp.summation(bn.subs(d, -sp.Rational(1, 3)).subs(a, sp.Rational(1, 2)), (n, 1, sp.oo))
    ok1b = abs(float(S - md_sum.subs({a: sp.Rational(1, 2), d: -sp.Rational(1, 3)}))) < 1e-12
    ok1b = ok1b and abs(float(sp.summation(bn.subs({a: -1, d: -2}), (n, 1, sp.oo)) - md_sum.subs({a: -1, d: -2}))) < 1e-12
    # d=0 이면 b_n 이 0 으로 가지 않아 발산
    ok1c = sp.limit(bn.subs(d, 0), n, sp.oo) != 0
    # (2-2) e^a+e^d=1: d=log(1-e^a), a<0
    md2 = sp.sympify(md_answer(2, "2")["value"], locals={"a": a})
    errs = []
    for av in (-0.1, -0.7, -2.0, -5.0):
        dv = np.log(1 - np.exp(av))
        T = sum(k * np.exp(av + (2 * k - 1) * dv) for k in range(1, 20000))
        errs.append(abs(T - float(md2.subs(a, av))) / abs(T))
    ok2 = max(errs) < 1e-9
    report(2, ok1a and ok1b and ok1c and ok2, f"ratio={ratio} cond={dcond} sum_check ok, (2-2) max rel err={max(errs):.2e}")


def count_intersections(f_left, f_right, m, n):
    """y=mx+n 과 x<0: f_left, x>=0: f_right(이차식 계수 튜플 (A,B,C)) 교점 수"""
    cnt = set()
    for (A, B, C), side in ((f_left, "L"), (f_right, "R")):
        roots = np.roots([A, B - m, C - n])
        for r in roots:
            if abs(r.imag) < 1e-9:
                r = r.real
                if (side == "L" and r < -1e-12) or (side == "R" and r >= -1e-12):
                    cnt.add(round(r, 9))
    return len(cnt)


def p03():
    # (3-1) k>0, 직선 y=x+4+k
    kk = sp.symbols("k", positive=True)
    left = sp.solve(sp.Eq((x + 2)**2, x + 4 + kk), x)
    right = sp.solve(sp.Eq((x - 2)**2 + kk, x + 4 + kk), x)
    ok_sym = sp.expand((x + 2)**2 - (x + 4 + kk)) == x**2 + 3*x - kk   # 카드의 x^2+3x-k=0
    counts = set()
    for kv in [0.01, 0.5, 1, 2, 3.7, 10, 100]:
        nl = sum(1 for r in left if float(r.subs(kk, kv)) < 0)
        nr = sum(1 for r in right if float(r.subs(kk, kv)) >= 0)
        counts.add(nl + nr)
    ok1 = counts == {int(md_answer(3, "1")["value"])}
    # (3-2) f = (x+2)^2 (x<0), -(x-2)^2+k (x>=0): 4점에서 만나는 직선 존재 ⇔ k in 답 집합
    ans = md_answer(3, "2")["value"]
    K = sp.sympify(ans)
    rng = random.Random(7)

    def exists(kv):
        # m<4 에서 m 을 훑고, n 은 두 조건의 공통 열린구간 중점
        for m in np.concatenate([np.linspace(-60, 3.99, 4000)]):
            lo = max(kv - 4, 4 - (m - 4)**2 / 4)
            hi = min(4, (m - 4)**2 / 4 + kv - 4)
            if lo < hi:
                nn = (lo + hi) / 2
                if count_intersections((1, 4, 4), (-1, 4, kv - 4), m, nn) == 4:
                    return True
        return False

    def random_never(kv):
        for _ in range(20000):
            m = rng.uniform(-50, 50); nn = rng.uniform(-200, 200)
            if count_intersections((1, 4, 4), (-1, 4, kv - 4), m, nn) == 4:
                return False
        return True
    ok_in = all(exists(kv) for kv in (-50, -3, 0, 5, 7.5, 7.99) if K.contains(kv))
    ok_out = all(random_never(kv) and not exists(kv) for kv in (8, 8.5, 12, 30) if not K.contains(kv))
    ok_bd = K.contains(7.99) and not K.contains(8) and K.contains(-1000)
    report(3, ok_sym and ok1 and ok_in and ok_out and ok_bd, f"(3-1) counts={counts}; (3-2) 답 {K}: 내부 k 존재, k>=8 무작위 2만 회 반례 없음")


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
