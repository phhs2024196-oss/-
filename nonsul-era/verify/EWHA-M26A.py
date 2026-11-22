#!/usr/bin/env python3
"""EWHA-M26A (이화여대 2026학년도 모의논술 자연계열Ⅰ) — sympy 검산. 정답 하드코딩 비교 금지."""
import re
import sys
import random
import pathlib

import sympy as sp
import yaml

WIDTH = 2
RESULTS = []


def item(no):
    f = pathlib.Path(__file__).resolve().parents[1] / "items/EWHA/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    return yaml.safe_load(f.read_text(encoding="utf-8"))


def md_answer(no, sub):
    return next(s["answer"]["value"] for s in item(no)["subparts"] if str(s["id"]) == sub)


def report(seq, ok, msg):
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

PREFIX = "EWHA-M26A"
random.seed(26)


def amgm(xs):
    s = sum(xs) / len(xs)
    p = 1.0
    for v in xs:
        p *= v / s
    return p          # = prod / mean^N  (<=1, =1 iff all equal)


def p01():
    # [증명-수치확인] (1)(3) 부등식 연쇄, (2)(4) 등호 조건: 무작위 반례 탐색
    ok = True
    for _ in range(3000):
        a = [random.uniform(0.01, 10) for _ in range(4)]
        m12, m34, m = (a[0]+a[1])/2, (a[2]+a[3])/2, sum(a)/4
        ok &= a[0]*a[1]*a[2]*a[3] <= m12**2*m34**2*(1+1e-12) <= m**4*(1+1e-12)
        for k in (1, 2, 3):
            N = 2**k
            b = [random.uniform(0.01, 10) for _ in range(2*N)]
            b1, b2 = sum(b[:N])/N, sum(b[N:])/N
            ok &= amgm(b[:N]) <= 1+1e-12 and amgm(b[N:]) <= 1+1e-12 and b1*b2 <= ((b1+b2)/2)**2*(1+1e-12)
            ok &= amgm(b) < 1 - 1e-12 or max(b)-min(b) < 1e-9
    # 등호: 모두 같으면 정확히 1 (sympy 기호)
    x = sp.symbols("x", positive=True)
    for N in (4, 8, 16):
        ok &= sp.simplify(x**N - (N*x/N)**N) == 0
    # 하나만 바꾸면 엄격 부등식
    ok &= all(amgm([1.0]*(2**k-1)+[1.0+1e-3]) < 1 for k in (2, 3, 4))
    report(1, ok, "[증명-수치확인] AM-GM 2^k 연쇄·등호조건 무작위 3000회 반례 없음")


def seq(a, b, N):
    from fractions import Fraction as F
    s = [None, F(a+b, 2)]
    s.append(2*a*s[1]/(a+s[1]))
    while len(s) <= N:
        n = len(s)                      # n = 2j+1 or 2j+2
        if n % 2 == 1:
            s.append((s[n-2]+s[n-1])/2)
        else:
            s.append(2*s[n-2]*s[n-1]/(s[n-2]+s[n-1]))
    return s


def p02():
    from fractions import Fraction as F
    ok = True
    for (a, b) in [(F(1), F(2)), (F(1, 3), F(7)), (F(5), F(6)), (F(1, 100), F(100))]:
        s = seq(a, b, 60)
        ok &= all(a < s[n] < b for n in range(1, 61))
        ok &= all(s[2*n] < s[2*n+1] for n in range(1, 30))
        ok &= all(s[2*n-1] > s[2*n+1] for n in range(1, 30))
        ok &= all(s[2*n] < s[2*n+2] for n in range(1, 29))
        # 해설의 인수분해식 검증
        n = 3
        ok &= s[2*n+2]-s[2*n] == s[2*n]/(s[2*n]+s[2*n+1])*(s[2*n+1]-s[2*n])
        ok &= s[2*n+3]-s[2*n+2] == s[2*n+1]/(2*(s[2*n]+s[2*n+1]))*(s[2*n+1]-s[2*n])
    report(2, ok, "[증명-수치확인] 유리수 정확계산 4쌍 n<=60: 유계·a2n<a2n+1·홀수항 감소·짝수항 증가")


def p03():
    X, Y, t = sp.symbols("x y t", real=True)
    def eq(s):
        l, r = s.split("=")
        return sp.sympify(l, locals={"x": X, "y": Y}) - sp.sympify(r, locals={"x": X, "y": Y})
    e1, e2 = eq(md_answer(3, "1")), eq(md_answer(3, "2"))
    # (1) P 가 모두 지나고, 초점 (±√3,0)에서 거리 합 일정
    P = (2*sp.cos(t), sp.sin(t))
    ok1 = sp.simplify(e1.subs({X: P[0], Y: P[1]})) == 0
    d = [sp.sqrt((P[0]-c)**2+P[1]**2) for c in (-sp.sqrt(3), sp.sqrt(3))]
    vals = [float((d[0]+d[1]).subs(t, v)) for v in (-1.2, -0.3, 0.4, 1.1)]
    ok1 &= max(vals)-min(vals) < 1e-12
    Q = (3/sp.cos(t), 2*sp.tan(t))
    ok2 = sp.simplify(e2.subs({X: Q[0], Y: Q[1]})) == 0
    d = [sp.sqrt((Q[0]-c)**2+Q[1]**2) for c in (-sp.sqrt(13), sp.sqrt(13))]
    vals = [float(sp.Abs(d[0]-d[1]).subs(t, v)) for v in (-1.2, -0.3, 0.4, 1.1)]
    ok2 &= max(vals)-min(vals) < 1e-12
    # (3) 신발끈 공식으로 넓이 → B_n = 넓이/(5/4)
    n = sp.symbols("n", positive=True, integer=True)
    th = sp.pi/2**n
    A = (-sp.Rational(3, 2), 0)
    Pn, Qn = (2*sp.cos(th), sp.sin(th)), (3/sp.cos(th), 2*sp.tan(th))
    area = sp.Rational(1, 2)*sp.Abs((Pn[0]-A[0])*(Qn[1]-A[1])-(Pn[1]-A[1])*(Qn[0]-A[0]))
    Bmd = sp.sympify(md_answer(3, "3"), locals={"n": n})
    ok3 = all(abs(float((area/sp.Rational(5, 4)-Bmd).subs(n, k))) < 1e-13 for k in range(2, 12))
    # (4) C_n 수치적분, 급수 n=2.. 합 (원문 시그마 n=2 부터)
    import mpmath as mp
    mp.mp.dps = 30
    def C(k):
        B = mp.sin(mp.pi/2**k)
        return mp.quad(lambda x: B/mp.cos(mp.asin(B*x)), [0, 1])
    ok4 = all(abs(C(k) - mp.pi/2**k) < mp.mpf(10)**-20 for k in range(2, 8))
    tot = sp.summation(sp.pi/2**n, (n, 2, sp.oo))
    ok4 &= sp.simplify(tot - sp.sympify(md_answer(3, "4"))) == 0
    report(3, ok1 and ok2 and ok3 and ok4, f"(1)(2) 곡선·초점 확인, (3) 넓이/(5/4)=B_n n=2..11, (4) C_n=pi/2^n, sum={tot}")


if __name__ == "__main__":
    fns = sorted((n for n in dir() if re.fullmatch(r"p\d+", n)), key=lambda s: int(s[1:]))
    for name in fns:
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(int(name[1:]), False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)

