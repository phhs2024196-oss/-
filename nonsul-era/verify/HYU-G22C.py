#!/usr/bin/env python3
"""HYU-G22C (한양대 2022학년도 논술 자연계열 오후2) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "HYU-G22C"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HYU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    return dict(seg.split("=", 1) for seg in s.replace(" ", "").split(";"))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    import math
    import random
    from fractions import Fraction as Fr
    # (1) 상태: 직전 면(구분 불필요) — 정확한 분포를 DP 로 계산(최대 3000번)
    # 1번 던진 뒤 '계속' 확률 1; 이후 매번 1/2 로 같은 면이면 종료
    cont, total = Fr(1), Fr(0)
    for k in range(2, 2023):
        stop = cont * Fr(1, 2)
        total += stop
        cont -= stop
    ok1 = total == sp.Rational(sp.sympify(md_answer(1, "1")))
    # 몬테카를로로 규칙 P(A_x)=(1/2)^(x-1) 확인(작은 x)
    random.seed(3)
    cnt = {}
    for _ in range(200000):
        prev, k = random.random() < .5, 1
        while True:
            cur = random.random() < .5; k += 1
            if cur == prev:
                break
            prev = cur
        cnt[k] = cnt.get(k, 0) + 1
    ok1 &= all(abs(cnt.get(k, 0) / 200000 - 0.5**(k - 1)) < 5e-3 for k in range(2, 7))
    # (2) 반지름 1, 중심각 T: 현 x(T), 활꼴 y(T) → dy/dx at x=1
    T = sp.symbols("T", positive=True)
    X = sp.sqrt(2 - 2 * sp.cos(T)); Y = (T - sp.sin(T)) / 2
    T1 = sp.nsolve(X - 1, T, 1)
    d = (sp.diff(Y, T) / sp.diff(X, T)).subs(T, T1)
    # 수치 미분: x 의 역함수 T(x)=2 asin(x/2)
    fx = lambda xx: (2 * math.asin(xx / 2) - math.sin(2 * math.asin(xx / 2))) / 2
    h = 1e-6
    num = (fx(1 + h) - fx(1 - h)) / (2 * h)
    ans2 = float(sp.sympify(md_answer(1, "2")))
    ok2 = abs(float(d) - ans2) < 1e-12 and abs(num - ans2) < 1e-8
    # (3) 증명: f''<0 기호 확인 + 무작위 표본에서 부등식 확인, f(0)=0
    f = sp.log(sp.log(x + sp.E))
    f2 = sp.simplify(sp.diff(f, x, 2))
    target = -(1 + sp.log(x + sp.E)) / ((x + sp.E)**2 * sp.log(x + sp.E)**2)
    ok3 = sp.simplify(f2 - target) == 0 and sp.simplify(f.subs(x, 0)) == 0
    fl = sp.lambdify(x, f, "math")
    random.seed(7)
    for _ in range(20000):
        a, b = math.exp(random.uniform(-8, 8)), math.exp(random.uniform(-8, 8))
        ok3 &= fl(a + b) < fl(a) + fl(b)
    ok3 &= md_answer(1, "3") is None
    report(1, ok1 and ok2 and ok3, f"P={float(total):.6f} f'(1)={float(d):.10f} num={num:.10f}")


def p02():
    # (1) 쌍곡선 x^2/5 - y^2/27 = 1/2, 초점 (±4,0)
    X0, Y0 = sp.symbols("X0 Y0", positive=True)
    hyp = X0**2 / 5 - Y0**2 / 27 - sp.Rational(1, 2)
    c2 = sp.Rational(5, 2) + sp.Rational(27, 2)
    c = sp.sqrt(c2)
    AF = sp.sqrt((X0 - c)**2 + Y0**2); AF_ = sp.sqrt((X0 + c)**2 + Y0**2)
    cosA = (AF**2 + AF_**2 - (2 * c)**2) / (2 * AF * AF_)
    sols = sp.nsolve([hyp, cosA - sp.Rational(7, 25)], [X0, Y0], [2.4, 4.4])
    a1 = parts(md_answer(2, "1"))
    A = sp.sympify(a1["A"]); B = sp.sympify(a1["B"])
    ok1 = abs(sols[0] - A[0]) < 1e-12 and abs(sols[1] - A[1]) < 1e-12
    # B: 제3사분면, cos(F'BF)=7/25 → 대칭점 확인
    BF = sp.sqrt((B[0] - c)**2 + B[1]**2); BF_ = sp.sqrt((B[0] + c)**2 + B[1]**2)
    ok1 &= sp.simplify(hyp.subs({X0: B[0], Y0: B[1]})) == 0 and \
        sp.simplify((BF**2 + BF_**2 - 4 * c2) / (2 * BF * BF_) - sp.Rational(7, 25)) == 0 and B[0] < 0 and B[1] < 0
    # (2) 타원 접선까지 거리 제곱 f(t) — 일반식으로 계산 후 b=√3/2 a
    a, b, tt = sp.symbols("a b tt", positive=True)
    s2 = b**2 * (1 - tt**2 / a**2)
    fT = (a**2 * b**2)**2 / (b**4 * tt**2 + a**4 * s2)            # 접선 b^2 t x + a^2 s y = a^2 b^2
    I = sp.integrate(sp.apart(sp.simplify(fT.subs(b, sp.sqrt(3) / 2 * a)), tt), (tt, 0, a)) / a**3
    ok2 = sp.simplify(sp.expand_log(I - sp.sympify(md_answer(2, "2")), force=True)) == 0
    # 수치 교차검증(a=1.7)
    import mpmath as mp
    fn = sp.lambdify(tt, fT.subs({b: sp.sqrt(3) / 2 * 1.7, a: 1.7}), "mpmath")
    ok2 &= abs(mp.quad(fn, [0, 1.7]) / 1.7**3 - float(sp.sympify(md_answer(2, "2")))) < 1e-12
    # (3) h(t) = PF1*PF2, s = sqrt(s2)
    cc = sp.sqrt(a**2 - b**2)
    import random
    random.seed(5)
    ok3 = True
    md3 = sp.sympify(md_answer(2, "3"), locals={"a": a, "b": b})
    for _ in range(30):
        av = random.uniform(1, 5); bv = random.uniform(0.1, 0.99) * av; tv = random.uniform(-av, av)
        sub = {a: av, b: bv, tt: tv}
        sv = sp.sqrt(s2).subs(sub)
        h = sp.sqrt((tv - cc.subs(sub))**2 + sv**2) * sp.sqrt((tv + cc.subs(sub))**2 + sv**2)
        ok3 &= abs(float(fT.subs(sub) * h - md3.subs(sub))) < 1e-9
    report(2, ok1 and ok2 and ok3, f"A~({float(sols[0]):.6f},{float(sols[1]):.6f}) I={sp.simplify(I)}")


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
