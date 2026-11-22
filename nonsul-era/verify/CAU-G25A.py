#!/usr/bin/env python3
"""CAU-G25A (2025학년도 중앙대 수시 논술 자연계열Ⅰ) — sympy/전수 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import itertools
import re
import sys
from fractions import Fraction as Fr

import mpmath as mp
import sympy as sp

PREFIX = "CAU-G25A"
WIDTH = 2
RESULTS = []
x, y, t, n = sp.symbols("x y t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """주사위 36 × 공 조합 전수(공 10개 모두 구별) → 2회 점수 합 ≥2 확률 (답 2/9)."""
    balls = ["R"] * 4 + ["B"] * 3 + ["G"] * 3
    dist = {}
    for d1 in range(1, 7):
        for d2 in range(1, 7):
            k = 2 if d1 + d2 <= 4 else 1
            combs = list(itertools.combinations(range(10), k))
            for cb in combs:
                sc = sum(balls[i] == "R" for i in cb)
                dist[sc] = dist.get(sc, 0) + Fr(1, 36) * Fr(1, len(combs))
    P = sum(dist[s1] * dist[s2] for s1 in dist for s2 in dist if s1 + s2 >= 2)
    report(1, P == Fr(2, 9), f"1회 분포={ {k: str(v) for k, v in dist.items()} }, P(합≥2)={P}")


def p02():
    """(1) S_1=a_1 에서 a_n=S_n-S_{n-1} 로 S_n 을 차례로 풀어 S_8=2025 (답 499/2)
    (2) f(x)=2f(x+2)+3 으로 [-4,-1] 위 f 를 만들어 적분 (답 15-8/π²)."""
    a1 = sp.symbols("a1")
    S = {1: a1}
    for k in range(2, 9):
        Sk = sp.symbols("Sk")
        an = Sk - S[k - 1]
        eq = sp.Eq(Sk, an / 2 - 2) if k % 2 == 0 else sp.Eq(Sk, 2 * an + 3)
        S[k] = sp.solve(eq, Sk)[0]
    sol = sp.solve(sp.Eq(S[8], 2025), a1)
    ok1 = sol == [sp.Rational(499, 2)]

    def f0(u):
        return sp.Piecewise((-u**2 * sp.cos(sp.pi * u), u < 0), (u**2 * sp.cos(sp.pi * u), True))
    # f(x-2) = 2f(x)+3  ⇒  f(x) = 2 f(x+2) + 3
    I1 = sp.integrate(4 * (x + 4) ** 2 * sp.cos(sp.pi * (x + 4)) + 9, (x, -4, -3))   # f(x)=4f(x+4)+9, x+4∈[0,1)
    I2 = sp.integrate(2 * f0(x + 2) + 3, (x, -3, -2))
    I3 = sp.integrate(2 * f0(x + 2) + 3, (x, -2, -1))
    # I1 의 식도 재귀로 확인: f(x)=2(2f(x+4)+3)+3
    chk = sp.simplify(2 * (2 * f0(sp.Rational(1, 3)) + 3) + 3 - (4 * f0(sp.Rational(1, 3)) + 9)) == 0
    tot = sp.simplify(I1 + I2 + I3)
    ok2 = chk and sp.simplify(tot - (15 - 8 / sp.pi**2)) == 0
    report(2, ok1 and ok2, f"(1) a1={sol}  (2) 적분={tot}")


def p03():
    """(1) 곡선 인수분해 → y=e^x 위 점과 직선 AB 거리 최소 (답 4-ln2)
    (2) 좌표로 삼각형을 만들어 AD 길이 f(x)를 수치로 구하고 적분 (답 √2/6-3/16)."""
    facs = sp.factor_list(y**3 - 3 * sp.exp(2 * x) * y + 2 * sp.exp(3 * x), y)[1]
    # 직선 AB: 2x-y-6=0, 거리 최소
    X = sp.symbols("X", real=True)
    dist = (2 * X - sp.exp(X) - 6) / sp.sqrt(5)
    crit = sp.solve(sp.diff(sp.exp(X) - 2 * X, X), X)
    dmin = sp.Abs(dist.subs(X, crit[0]))
    area = sp.simplify(sp.sqrt(5) * dmin / 2)
    # e^x - (2x-6) > 0 이 항상 성립(한쪽에만 있음) 수치 확인
    side = all(mp.e**v - (2 * v - 6) > 0 for v in mp.linspace(-20, 20, 401))
    ok1 = side and sp.simplify(area - (4 - sp.log(2))) == 0

    def fAD(xx):
        A = mp.matrix([0, 0])
        B = mp.matrix([2, 0])
        C = mp.matrix([mp.cos(3 * xx), mp.sin(3 * xx)])
        d = mp.matrix([mp.cos(xx), mp.sin(xx)])          # 반직선 AD 방향
        # A + s d = B + u (C-B)
        M = mp.matrix([[d[0], -(C[0] - B[0])], [d[1], -(C[1] - B[1])]])
        s, u = mp.lu_solve(M, B - A)
        assert 0 <= u <= 1
        return s
    mp.mp.dps = 30
    val = mp.quad(lambda xx: fAD(xx) * mp.sin(xx) ** 3, [mp.pi / 6, mp.pi / 4])
    ok2 = abs(val - (mp.sqrt(2) / 6 - mp.mpf(3) / 16)) < 1e-20
    report(3, ok1 and ok2, f"(1) 인수 {[str(f) for f, _ in facs]}, 최소넓이={area}  (2) 적분≈{mp.nstr(val, 15)}")


def p04():
    """(1) 정육면체 좌표에서 P(p,0,2), Q(0,q,2) 를 넓이 조건으로 결정 → △EPQ (답 √14/2)
    (2) 쌍곡선 접선·점근선으로 Q,R,S 를 만들고 넓이 이등분 조건 → F 와 직선 F'P 거리 (답 4√3/21)."""
    p, q = sp.symbols("p q", nonnegative=True)
    E_ = sp.Matrix([0, 0, 0]); G_ = sp.Matrix([2, 2, 0])
    P_ = sp.Matrix([p, 0, 2]); Q_ = sp.Matrix([0, q, 2])
    tri = lambda U, V, W: (V - U).cross(W - U).norm() / 2
    ps = [s for s in sp.solve(sp.Eq(tri(E_, P_, G_) ** 2, 10), p) if 0 <= s <= 2]
    qs = [s for s in sp.solve(sp.Eq(tri(E_, Q_, G_) ** 2, 9), q) if 0 <= s <= 2]
    areas = {sp.simplify(tri(E_, P_.subs(p, a), Q_.subs(q, b))) for a in ps for b in qs}
    ok1 = areas == {sp.sqrt(14) / 2}

    a, b = sp.symbols("a b", positive=True)
    Qy = -1 / (3 * b)                         # 접선 ax-3by=1 의 y절편
    Rx = sp.solve(sp.Eq(a * x - 3 * b * x / sp.sqrt(3), 1), x)[0]
    Sx = 1 / a
    area_OQR = sp.Abs(Qy) * Rx / 2            # 밑변 OQ(y축), 높이 R의 x좌표
    area_OQS = sp.Abs(Qy) * Sx / 2
    sols = sp.solve([sp.Eq(area_OQR, 2 * area_OQS), sp.Eq(a**2 - 3 * b**2, 1)], [a, b], dict=True)
    sols = [s for s in sols if s[a] > 0 and s[b] > 0 and (s[a] - sp.sqrt(3) * s[b]) > 0]
    c = sp.sqrt(1 + sp.Rational(1, 3))
    res = set()
    for s in sols:
        Pp = sp.Matrix([s[a], s[b]]); Fp = sp.Matrix([-c, 0]); F = sp.Matrix([c, 0])
        d = sp.Abs((Pp - Fp)[0] * (F - Fp)[1] - (Pp - Fp)[1] * (F - Fp)[0]) / (Pp - Fp).norm()
        res.add(sp.nsimplify(sp.simplify(d)))
    ok2 = res == {4 * sp.sqrt(3) / 21}
    report(4, ok1 and ok2, f"(1) p={ps}, q={qs}, △EPQ={areas}  (2) (a,b)={[(s[a], s[b]) for s in sols]}, 거리={res}")


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
