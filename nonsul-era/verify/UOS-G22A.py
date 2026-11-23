import sys, pathlib, itertools, math, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
from fractions import Fraction as F
import mpmath as mp
V = Verifier("UOS-G22A", "items/UOS/PAST")
a, b = sp.symbols("a b")


def area(P, Q, R):
    return abs((Q[0]-P[0])*(R[1]-P[1]) - (R[0]-P[0])*(Q[1]-P[1])) / 2


def p01():
    pt = lambda u: (u, u - u**2)
    O, C = (0, 0), (1, 0)
    # (a) 0<b<a 에서 삼각형 OAB 넓이의 b에 대한 최대 → a**3/8
    S1 = sp.simplify(sp.Rational(1, 2) * (a*(b - b**2) - b*(a - a**2)))
    bs = sp.solve(sp.diff(S1, b), b)[0]
    V.check(1, "1-1 최대(식)", sp.simplify(S1.subs(b, bs)), V.ans(1, "1-1"))
    # 수치: 사각형 ABOC(꼭짓점 A,B,O,C 순서) 넓이 최대 격자 탐색
    best = 0; arg = None
    N = 400
    for i in range(1, N):
        for j in range(1, i):
            A, B = pt(i/N), pt(j/N)
            quad = area(A, B, O) + area(A, O, C)          # 대각선 OA 로 분할
            sh = abs(sum(p[0]*q[1]-q[0]*p[1] for p, q in zip([A, B, O, C], [B, O, C, A])))/2   # 신발끈
            assert abs(quad - sh) < 1e-12
            if sh > best: best, arg = sh, (i/N, j/N)
    V.report(1, abs(best - float(sp.Rational(V.ans(1, "1-2")))) < 1e-4 and abs(arg[0]-2/3) < 5e-3 and abs(arg[1]-1/3) < 5e-3,
             f"1-2 격자 최대 {best:.6f} at {arg} / YAML {V.ans(1, '1-2')}")
    f = (a**3 - 4*a**2 + 4*a)/8
    V.check(1, "1-2 f(2/3)", f.subs(a, sp.Rational(2, 3)), V.ans(1, "1-2"))


def seqs():
    return itertools.product(range(1, 7), repeat=6)


def p02():
    tot = A = AB = B = Bc = ABc = 0
    for s in seqs():
        tot += 1
        no3 = not any(s[i] % 3 == 0 and s[i+1] % 3 == 0 for i in range(5))
        hasle2 = any(v <= 2 for v in s)
        A += no3; B += hasle2
        AB += no3 and hasle2; ABc += no3 and not hasle2
    V.check(2, "2-1 전수", sp.Rational(A, tot), V.ans(2, "2-1"))
    V.check(2, "2-2 전수 P(A|B)", sp.Rational(AB, B), V.ans(2, "2-2"))
    V.check(2, "P(A|B^C)=21/64", sp.Rational(ABc, tot - B), sp.Rational(21, 64))


def p03():
    f = lambda x: abs(3*mp.sqrt(2)*mp.sin(x)**3 - mp.cos(x))
    mp.mp.dps = 30
    a0 = mp.findroot(lambda x: 3*mp.sqrt(2)*mp.sin(x)**3 - mp.cos(x), 0.6)
    val = mp.quad(f, [-mp.pi/2, a0, mp.pi/2])
    V.check(3, "3-1 수치적분", sp.Float(str(val), 25), V.ans(3, "3-1"), tol=1e-12)
    V.report(3, abs(mp.sin(a0) - 1/mp.sqrt(3)) < 1e-20, "sin a = 1/sqrt3, 근의 유일성(증가) 수치")
    xs = [mp.mpf(i)/200*mp.pi/2 for i in range(201)]
    vals = [3*mp.sqrt(2)*mp.sin(x)**3 - mp.cos(x) for x in xs]
    V.report(3, all(v2 > v1 for v1, v2 in zip(vals, vals[1:])), "[0,pi/2]에서 f 증가")


def p04():
    mp.mp.dps = 80
    an = mp.mpf(5); bad = []
    for n in range(1, 400):
        if not (4 < an <= mp.mpf(3)/4**(0) * (mp.mpf(3)/4)**(n-1) + 4): bad.append(("n", n, an))
        an = mp.mpf(3)/4*an + 2/mp.sqrt(an)
    # 보조 부등식 a>4: 0<1-2/sqrt(a)<(a-4)/8
    for i in range(1, 4000):
        aa = mp.mpf(4) + mp.mpf(i)/40
        if not (0 < 1 - 2/mp.sqrt(aa) < (aa-4)/8): bad.append(("aux", aa))
    V.proof(4, "n=1..399 수열 부등식 및 보조 부등식", bad)
    x = sp.symbols("x", positive=True)
    V.report(4, sp.simplify((1-2/sp.sqrt(x)) - (x-4)/(sp.sqrt(x)*(sp.sqrt(x)+2))) == 0, "유리화 항등식")


if __name__ == "__main__": V.run(globals())
