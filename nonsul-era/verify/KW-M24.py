"""광운대 2024학년도 논술우수자전형 모의논술 자연계열 [문제 1]·[문제 2] 검산 — 기대값은 YAML(V.ans)에서만 읽는다."""
import sys, pathlib, itertools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
import numpy as np

V = Verifier("KW-M24", "items/KW/PAST")
p_ = sp.symbols("p", integer=True, positive=True)
LOC = {"x": x, "n": n, "k": k, "p": p_, "E": sp.E, "pi": sp.pi, "binomial": sp.binomial}


def ex(s, **kw):
    return sp.sympify(s, locals={**LOC, **kw})


def p01():
    # [1](1) 교점, 접선, 넓이
    cands = [sp.pi / 4 + j * sp.pi for j in range(-3, 4)]            # sin x = cos x 의 해
    x0 = min(cands, key=lambda c: abs(float(c)))
    y0 = sp.sin(x0)
    yB = sp.simplify(sp.cos(x0) + (-sp.sin(x0)) * (0 - x0))
    yC = sp.simplify(sp.sin(x0) + sp.cos(x0) * (0 - x0))
    area = sp.simplify(sp.Rational(1, 2) * abs(yB - yC) * abs(x0))
    V.check(1, "1-1 넓이", area, V.ans(1, "1-1"))
    # [1](2) 외접원 반지름: 두 방법
    Ax, Ay = x0, y0
    b_ = sp.sqrt((Ax - 0) ** 2 + (Ay - yB) ** 2)
    c_ = sp.sqrt((Ax - 0) ** 2 + (Ay - yC) ** 2)
    a_ = abs(yB - yC)
    cosA = (b_ ** 2 + c_ ** 2 - a_ ** 2) / (2 * b_ * c_)
    R1 = a_ / (2 * sp.sqrt(1 - cosA ** 2))
    R2 = a_ * b_ * c_ / (4 * area)
    V.check(1, "1-2 R (사인법칙)", sp.simplify(R1), V.ans(1, "1-2"))
    V.check(1, "1-2 R (abc/4S)", sp.simplify(R2), V.ans(1, "1-2"))
    # [2](1) 귀납 명제 수치 확인
    V.proof(1, "2-1 n! >= 2^(n-1)", [m for m in range(1, 200) if not (sp.factorial(m) >= 2 ** (m - 1))])
    # [2](2)
    mp.mp.dps = 60
    a2024 = mp.fsum(mp.mpf(1) / mp.factorial(j) for j in range(1, 2025))
    m_ = int(mp.floor(a2024 - 2024))
    V.check(1, "2-2 m", m_, V.ans(1, "2-2"))
    V.report(1, 1 <= a2024 < 2, "2-2 1 ≤ a_2024 < 2")
    # [3](1) 역함수가 존재하는 k 의 최솟값 = -min_x e^{x+1}(x^2+(n+2)x+1)
    xs = np.linspace(-80, 12, 2000001)
    for nv in (1, 2, 3, 5, 8):
        h = np.exp(xs + 1) * (xs ** 2 + (nv + 2) * xs + 1)
        kmin = -h.min()
        V.check(1, f"3-1 n={nv} k 최솟값", sp.Float(float(kmin), 8), sp.N(ex(V.ans(1, "3-1")).subs(n, nv), 8), tol=1e-6)
    # 도함수 식과 x=-1 에서 최소
    fx = sp.exp(x + 1) * (x ** 2 + n * x - n + 1) + sp.Symbol("kk") * x
    h = sp.diff(sp.exp(x + 1) * (x ** 2 + (n + 2) * x + 1), x)
    V.check(1, "3-1 f' 의 도함수", sp.simplify(sp.diff(fx, x, 2) - h), 0)
    V.check(1, "3-1 h' 식", sp.simplify(h - sp.exp(x + 1) * (x + 1) * (x + n + 3)), 0)
    # [3](2)
    m = sp.Symbol("m", integer=True, positive=True)
    S = sp.summation(1 / (m * (m + 1) * (m + 2)), (m, 1, sp.oo))
    V.check(1, "3-2 급수의 합", S, V.ans(1, "3-2"))
    # [4]
    def cond(kv):
        if kv <= 0 or abs(np.log(kv) - 1) < 1e-14:
            return False
        L = np.log(kv) - 1
        okA = (L * L - L) >= 0
        okB = 2 * np.sin(np.pi / (2 * np.e) * kv - np.pi / 4) ** 2 <= 1 + 1e-12
        return bool(okA and okB)
    ks = np.concatenate([np.linspace(-np.e / 2, 3 * np.e / 2, 400001), [0.0, np.e]])
    good = np.array([cond(v) for v in ks])
    lo, hi = ks[good].min(), ks[good].max()
    V.check(1, "4 하한", sp.Float(float(lo), 3), sp.N(ex(V.ans(1, "4", "lower")), 3), tol=1e-3)
    V.check(1, "4 상한", sp.Float(float(hi), 3), sp.N(ex(V.ans(1, "4", "upper")), 3), tol=1e-3)
    V.report(1, (not cond(0.0)) and (not cond(float(np.e))) and cond(float(np.e) - 1e-6) and cond(1e-6), "4 양 끝점 제외·내부 포함")


def p02():
    # [1](1) 극한으로 정의된 f 와 구간별 식
    mp.mp.dps = 40
    def fn(xv, N=400):
        xv = mp.mpf(xv)
        if xv == 0:
            return mp.mpf(0)
        return (xv + xv ** (2 * N)) * mp.log(abs(xv)) / (xv ** (2 * N) + 1)
    def piece(xv):
        xv = mp.mpf(xv)
        if xv in (-1, 0, 1):
            return mp.mpf(0)
        return xv * mp.log(abs(xv)) if abs(xv) < 1 else mp.log(abs(xv))
    pts = [-3, -1.5, -1, -0.9, -0.5, -0.1, 0, 0.1, 0.5, 0.9, 1, 1.1, 2, 5]
    V.proof(2, "1-1 구간별 식 = 극한 정의(n=400)", [t for t in pts if abs(fn(t) - piece(t)) > 1e-8])
    V.proof(2, "1-1 연속(±1, 0 근방 값 → 0)", [(c, d) for c in (-1, 0, 1) for d in (1e-9, -1e-9) if abs(piece(c + d)) > 1e-6])
    # [1](2)
    right = limit(sp.log(1 + x) / x, x, 0, "+")
    left = limit((1 + x) * sp.log(1 + x) / x, x, 0, "-")
    V.check(2, "1-2 우미분계수", right, V.ans(2, "1-2"))
    V.check(2, "1-2 좌미분계수", left, V.ans(2, "1-2"))
    # [1](3)
    val = mp.quad(lambda t: t * mp.log(t), [mp.mpf(1) / mp.e, 1]) + mp.quad(mp.log, [1, mp.e])
    val2 = mp.quad(lambda t: fn(t, 600), [mp.mpf(1) / mp.e, 1, mp.e])
    want = sp.N(ex(V.ans(2, "1-3")), 25)
    V.check(2, "1-3 정적분(구간별 식)", sp.Float(str(val), 25), want, tol=1e-12)
    V.check(2, "1-3 정적분(극한식 n=600)", sp.Float(str(val2), 25), want, tol=1e-6)
    # [2]
    a_, b_, c1 = sp.symbols("a_ b_ c1")
    gx = sp.exp(x ** 3 + x + 1)
    Q = x + c1  # 임의의 몫
    f_ = x ** 2 * Q + a_ * x + b_
    g_inv_e = 0   # g(0)=e
    V.check(2, "2 g(0)=e", gx.subs(x, 0), sp.E)
    sol = sp.solve([f_.subs(x, g_inv_e) - sp.pi, sp.diff(f_, x).subs(x, 0) / sp.diff(gx, x).subs(x, 0) - 1], [a_, b_], dict=True)[0]
    V.check(2, "2 a", sol[a_], V.ans(2, "2", "a"))
    V.check(2, "2 b", sol[b_], V.ans(2, "2", "b"))
    # [3](1) 독립시행: 승/패 열거와 비교
    nv = 6
    Pk_enum = {kk: sum(sp.Rational(3, 4) ** sum(s) * sp.Rational(1, 4) ** (nv - sum(s)) for s in itertools.product((0, 1), repeat=nv) if sum(s) == kk) for kk in range(nv + 1)}
    ok = all(Pk_enum[kk] == ex(V.ans(2, "3-1")).subs({n: nv, k: kk}) for kk in range(nv + 1))
    V.report(2, ok, "3-1 P_k 공식 = 승패열 전수(n=6)")
    # [3](2)
    xx = sp.Symbol("xx")
    for nv in range(1, 9):
        Pk = lambda kk: ex(V.ans(2, "3-1")).subs({n: nv, k: kk})
        F = 4 ** nv * sum(Pk(kk) * xx ** (kk + 1) for kk in range(nv + 1))
        V.check(2, f"3-2 n={nv} f'(1)", sp.diff(F, xx).subs(xx, 1), ex(V.ans(2, "3-2")).subs(n, nv))
    # [3](3)
    for pv in range(1, 9):
        tot = 0
        for nv in range(1, pv + 1):
            tot += 4 ** nv * sum(ex(V.ans(2, "3-1")).subs({n: nv, k: kk}) * xx ** (kk + 1) for kk in range(nv + 1))
        coef = sp.Poly(sp.expand(tot), xx).coeff_monomial(xx ** 2)
        V.check(2, f"3-3 p={pv} x^2 계수", coef, ex(V.ans(2, "3-3")).subs(p_, pv))


if __name__ == "__main__":
    V.run(globals())
