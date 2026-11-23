import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import itertools
from math import comb

V = Verifier("YSM-G24M", "items/YSM/PAST")
E = lambda v: sp.sympify(v, locals={"x": x, "t": t, "n": n, "k": k})

def subsets(U):
    U = list(U)
    return [frozenset(c) for r in range(len(U) + 1) for c in itertools.combinations(U, r)]

def pairs(m, size):
    """U_m={1..m} 의 두 부분집합 A,B 가 A∪B=U_m, n(A)=size 인 (A,B) 전체"""
    U = frozenset(range(1, m + 1)); S = subsets(U)
    return [(A, B) for A in S if len(A) == size for B in S if A | B == U]

def p01():
    a = lambda kk: len(pairs(kk + 1, 2)); b = lambda kk: len(pairs(kk + 1, 3))
    V.check(1, "1-1 a2", a(2), 12); V.check(1, "1-1 b3", b(3), E(V.ans(1, "1-1", "b3")))
    V.check(1, "1-1 a2(YAML)", a(2), E(V.ans(1, "1-1", "a2")))
    V.check(1, "1-1 b2=2*a1", b(2), 2*a(1))
    c1 = sp.Rational(b(2), a(1))
    c2 = (b(3) - c1*a(1))/sp.Integer(a(2))
    V.check(1, "1-1 c1", c1, 2)
    V.check(1, "1-1 c2", c2, E(V.ans(1, "1-1", "c2")))
    # b3 에 쓰인 <A,B> : A 는 3원소(4개), B ⊇ U4\A (각 8개), 표 2 와 같은 32개
    P3 = pairs(4, 3)
    U4 = frozenset(range(1, 5))
    V.report(1, len(P3) == 32 and len({A for A, _ in P3}) == 4 and all((U4 - A) <= B for A, B in P3)
             and all(sum(1 for A2, _ in P3 if A2 == A) == 8 for A, _ in P3), "1-1 <A,B> 32개 = 4(A)×8(B⊇A^c)")
    # 표 2 왼쪽 위 칸: A={1,2,3} 의 B 8개
    B123 = sorted(sorted(B) for A, B in P3 if A == frozenset({1, 2, 3}))
    exp = sorted(sorted(s) for s in [{4}, {1, 4}, {2, 4}, {3, 4}, {1, 2, 4}, {1, 3, 4}, {2, 3, 4}, {1, 2, 3, 4}])
    V.report(1, B123 == exp, "1-1 A={1,2,3} 일 때 B 목록이 표 2 와 일치")
    kk = sp.symbols("kk", positive=True, integer=True)
    # 직접 센 값으로 닫힌 식 확인(k=1..5)
    ak = lambda m: len(pairs(m + 1, 2)); bk = lambda m: len(pairs(m + 1, 3))
    fa = lambda m: 2*m*(m + 1); fb = lambda m: sp.Rational(4, 3)*(m - 1)*m*(m + 1)
    V.report(1, all(ak(m) == fa(m) for m in range(1, 6)), "1-2 a_k=2k(k+1) (k=1..5 전수)")
    V.report(1, all(bk(m) == fb(m) for m in range(1, 6)), "1-2 b_k=4/3(k-1)k(k+1) (k=1..5 전수)")
    # b_{k+1} = sum c_i a_i 에서 c_k 를 차례로 풀기 (k=1..30), 닫힌 식으로
    cs = []
    for m in range(1, 31):
        ck = (fb(m + 1) - sum(cs[i - 1]*fa(i) for i in range(1, m)))/sp.Integer(fa(m))
        cs.append(ck)
    V.report(1, all(c == 2 for c in cs), f"1-2 c_1..c_30 모두 2 ({set(cs)})")
    V.report(1, all(sp.simplify(sp.prod(cs[:m]) - E(V.ans(1, "1-2")).subs(k, m)) == 0 for m in range(1, 31)), "1-2 곱 = 2^k (k=1..30)")
    V.check(1, "1-2 시그마 항등식", sp.simplify(sp.summation(2*kk*(kk + 1), (kk, 1, k)) - sp.Rational(1, 2)*fb(k + 1)), 0)

def p02():
    # 2-1: 타원 2a=180, 쌍곡선 2a'=60 (같은 초점 ±c), 교점 R 에서 ∠PRQ=90°
    a_, ap, c = sp.symbols("a_ ap c", positive=True)
    u, v = sp.symbols("u v", positive=True)
    sol = sp.solve([sp.Eq(u + v, 180), sp.Eq(v - u, 60), sp.Eq(u**2 + v**2, 4*c**2)], [u, v, c], dict=True)
    V.report(2, len(sol) == 1, f"2-1 해 {sol}")
    cv = sol[0][c]
    V.check(2, "2-1 두 게양대 거리 2c", 2*cv, E(V.ans(2, "2-1", "d")))
    bq = sp.sqrt(90**2 - cv**2); bh = sp.sqrt(cv**2 - 30**2)
    V.check(2, "2-1 b", bq, 60); V.check(2, "2-1 b'", bh, 60)
    # 좌표로 교점 R 을 직접 구해 직각 확인
    X, Y = sp.symbols("X Y", real=True)
    R = sp.solve([X**2/sp.Integer(90**2) + Y**2/bq**2 - 1, X**2/sp.Integer(30**2) - Y**2/bh**2 - 1], [X, Y], dict=True)
    Rp = [r for r in R if r[X] > 0 and r[Y] > 0][0]
    Pm = sp.Matrix([cv, 0]); Qm = sp.Matrix([-cv, 0]); Rm = sp.Matrix([Rp[X], Rp[Y]])
    V.check(2, "2-1 (R-P)·(R-Q)=0 직각", (Rm - Pm).dot(Rm - Qm), 0)
    V.check(2, "2-1 PR+QR=180", sp.sqrt((Rm - Pm).dot(Rm - Pm)) + sp.sqrt((Rm - Qm).dot(Rm - Qm)), 180)
    V.check(2, "2-1 |QR-PR|=60", sp.sqrt((Rm - Qm).dot(Rm - Qm)) - sp.sqrt((Rm - Pm).dot(Rm - Pm)), 60)
    AB = sp.sqrt(sp.simplify(bh**2*(sp.Rational(90**2, 30**2) - 1)))
    V.check(2, "2-1 AB(세로 길이)", AB, E(V.ans(2, "2-1", "h")))
    V.check(2, "2-1 B 가 타원 장축 끝점 x=90 의 쌍곡선 위", (sp.Rational(90**2, 30**2)) - AB**2/bh**2, 1)
    # 2-2
    th = sp.tan(sp.pi/12)
    V.check(2, "2-2 tan15°", sp.simplify(th), 2 - sp.sqrt(3))
    PA = 90 - cv
    BP2 = sp.expand(PA**2 + AB**2)
    h2 = sp.expand(th**2*BP2)
    h2 = sp.expand(sp.radsimp(sp.expand(h2.rewrite(sp.sqrt))))
    e = sp.expand(h2/1800)
    cf = sp.collect(e, [sp.sqrt(3), sp.sqrt(5), sp.sqrt(15)], evaluate=False)
    g = lambda s: cf.get(s, 0)
    hh = [g(sp.Integer(1)), g(sp.sqrt(3)), g(sp.sqrt(5)), g(sp.sqrt(15))]
    V.report(2, hh == [161, -92, -21, 12], f"2-2 h1..h4 = {hh}")
    V.check(2, "2-2 합", sum(hh), E(V.ans(2, "2-2")))
    V.check(2, "2-2 h^2 재조립", 1800*(hh[0] + hh[1]*sp.sqrt(3) + hh[2]*sp.sqrt(5) + hh[3]*sp.sqrt(15)), th**2*BP2)

if __name__ == "__main__": V.run(globals())
