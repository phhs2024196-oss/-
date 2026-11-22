#!/usr/bin/env python3
"""SMWU-M27P (숙명여대 2027학년도 모의논술 자연계열(약학부)) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import random
import re
import sys

import numpy as np
import sympy as sp

PREFIX = "SMWU-M27P"
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
    # (1-1) x^3/(x^6+x^3+1) = 1/(x^3+1/x^3+1) <= 1/3
    h = x**3 / (x**6 + x**3 + 1)
    ok_d = sp.simplify(h - 1 / (x**3 + 1 / x**3 + 1)) == 0
    ok_e = all(float(h.subs(x, v)) <= 1 / 3 + 1e-15 for v in np.linspace(1, 2, 1001))
    I1 = sp.Integral(h, (x, 1, 2)).evalf(30)
    ok1 = ok_d and ok_e and I1 <= sp.Rational(1, 3)
    # (1-2) 0<=x<=pi/6<1: x^2<=x, cos 감소 ⇒ cos(x^2)>=cos x, ∫cos x = 1/2
    ok_f = sp.pi / 6 < 1
    xs = np.linspace(0, np.pi / 6, 1001)
    ok_g = all(v**2 <= v + 1e-15 and np.cos(v**2) >= np.cos(v) - 1e-15 for v in xs)
    ok_h = all(float(sp.diff(sp.cos(x), x).subs(x, v)) <= 0 for v in xs)
    lower = sp.integrate(sp.cos(x), (x, 0, sp.pi / 6))
    I2 = sp.Integral(sp.cos(x**2), (x, 0, sp.pi / 6)).evalf(30)
    ok2 = ok_f and ok_g and ok_h and lower == sp.Rational(1, 2) and I2 >= lower
    report(1, ok0 and ok1 and ok2, f"[증명-수치확인] I1={float(I1):.6f}<=1/3, I2={float(I2):.6f}>=1/2")


def p02():
    a, r = sp.symbols("a r", positive=True)
    k, n = sp.symbols("k n", integer=True, positive=True)
    ak = a * r**(k - 1)
    # (2-1) B3/C3 <= 7/6, C3 = exp(A3) = (a1 a2 a3)^(1/3)
    C3 = sp.simplify(sp.exp(sp.expand_log(sum(sp.log(ak.subs(k, j)) for j in (1, 2, 3)) / 3, force=True)))
    B3 = sum(ak.subs(k, j) for j in (1, 2, 3)) / 3
    sol = sp.solveset(sp.simplify(B3 / C3) - sp.Rational(7, 6) <= 0, r, sp.Interval.open(0, sp.oo))
    md1 = sp.sympify(md_answer(2, "1")["value"])
    ok1 = sol == md1 and sp.simplify(C3 - a * r) == 0
    # (2-2) a+r=1: Σ a_n ln a_n 직접 수치 합 vs md 식
    md2 = sp.sympify(md_answer(2, "2")["value"], locals={"a": a})
    errs = []
    for av in (0.1, 0.35, 0.5, 0.8, 0.97):
        rv = 1 - av
        S = sum(av * rv**(j - 1) * (np.log(av) + (j - 1) * np.log(rv)) for j in range(1, 6000))  # ln(a r^(j-1)) 를 그대로 전개(언더플로 방지)
        errs.append(abs(S - float(md2.subs(a, av))))
    ok2 = all(np.isfinite(errs)) and max(errs) < 1e-9
    for e in md_answer(2, "2")["equivalents"]:
        ok2 = ok2 and abs(float((sp.sympify(e, locals={"a": a}) - md2).subs(a, 0.37))) < 1e-12
    report(2, ok1 and ok2, f"C3={C3} r in {sol}; (2-2) max abs err={max(errs):.2e}")


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
    # (3-1) f = (x+2)^2 (x<0), -(x-2)^2+k (x>=0): 4점에서 만나는 직선 존재 ⇔ k in 답 집합
    K = sp.sympify(md_answer(3, "1")["value"])
    rng = random.Random(7)

    def exists(kv):
        for m in np.linspace(-60, 3.99, 4000):
            lo = max(kv - 4, 4 - (m - 4)**2 / 4)
            hi = min(4, (m - 4)**2 / 4 + kv - 4)
            if lo < hi and count_intersections((1, 4, 4), (-1, 4, kv - 4), m, (lo + hi) / 2) == 4:
                return True
        return False

    def random_never(kv):
        for _ in range(20000):
            m = rng.uniform(-50, 50); nn = rng.uniform(-200, 200)
            if count_intersections((1, 4, 4), (-1, 4, kv - 4), m, nn) == 4:
                return False
        return True
    ok_in = all(exists(kv) for kv in (-50, -3, 0, 5, 7.5, 7.99))
    ok_out = all(random_never(kv) and not exists(kv) for kv in (8, 8.5, 12, 30))
    ok1 = ok_in and ok_out and K.contains(7.99) and not K.contains(8) and K.contains(-1000)
    # (3-2) [증명-수치확인] D/4 = 4n(am+n) 항등식 + 무작위 20점에서 접하는 이차함수 존재
    ok0 = md_answer(3, "2")["kind"] == "proof"
    a, m, n, k = sp.symbols("a m n k", real=True)
    eq = sp.expand((k * a + m)**2 + 4 * k * n)
    ok_id = sp.expand(eq - (a**2 * k**2 + 2 * (a * m + 2 * n) * k + m**2)) == 0
    D4 = sp.expand((a * m + 2 * n)**2 - a**2 * m**2)
    ok_id = ok_id and sp.expand(D4 - 4 * n * (a * m + n)) == 0
    X = sp.symbols("X", real=True)
    good = 0
    for _ in range(20):
        while True:
            av = rng.choice([-1, 1]) * rng.uniform(0.2, 5); mv = rng.uniform(-5, 5); nv = rng.uniform(-5, 5)
            if nv * (av * mv + nv) > 0:            # 선분 PQ 와 만나지 않음
                break
        ks = [kv for kv in np.roots([av**2, 2 * (av * mv + 2 * nv), mv**2]) if abs(kv.imag) < 1e-12 and abs(kv.real) > 1e-9]
        if not ks:
            continue
        kv = ks[0].real
        disc = (kv * av + mv)**2 + 4 * kv * nv     # kx(x-a)=mx+n 의 판별식(중근 ⇔ 0)
        if abs(disc) < 1e-7 * (1 + abs(mv)**2 + abs(kv * av)**2):
            good += 1
    ok2 = ok0 and ok_id and good == 20
    report(3, ok1 and ok2, f"(3-1) 답 {K}: 내부 k 존재, k>=8 반례 없음; (3-2) [증명-수치확인] D/4=4n(am+n), 무작위 {good}/20 접선 이차함수 존재")


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
