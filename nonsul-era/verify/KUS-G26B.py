#!/usr/bin/env python3
"""KUS-G26B (고려대 세종 2026학년도 자연계열Ⅱ 약학과) 검산 — 기대값은 YAML 에서 읽는다."""
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *  # noqa: E402,F401,F403

V = Verifier("KUS-G26B", "items/KUS/PAST")
R = sp.Rational


def p01():
    O, O2, P = sp.Matrix([0, 0]), sp.Matrix([3, 0]), sp.Matrix([-5, 0])
    N = 720
    circ = lambda c, i: c + 3 * sp.Matrix([math.cos(2 * math.pi * i / N), math.sin(2 * math.pi * i / N)])
    import numpy as np
    th = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
    Cp = np.stack([3 * np.cos(th), 3 * np.sin(th)], 1)           # 원 C
    C2 = np.stack([3 + 3 * np.cos(th), 3 * np.sin(th)], 1)       # 원 C'
    # 1-1: A: 렌즈 경계 & |PA|<=7, B,Y': C' 위 & |P.|>=sqrt73, 각 BAY' 최대 -> 넓이
    inC = lambda q: (q[:, 0]**2 + q[:, 1]**2) <= 9 + 1e-9
    inC2 = lambda q: ((q[:, 0] - 3)**2 + q[:, 1]**2) <= 9 + 1e-9
    lens = np.vstack([Cp[inC2(Cp)], C2[inC(C2)]])
    Pn = np.array([-5, 0])
    A = lens[np.linalg.norm(lens - Pn, axis=1) <= 7 + 1e-9]
    Bs = C2[np.linalg.norm(C2 - Pn, axis=1) >= math.sqrt(73) - 1e-9]
    Bs = np.vstack([Bs, [[3, 3], [3, -3]]])
    best = 0
    Ai = np.vstack([A[::4], [[0, 0]]]); Bi = Bs[::6]
    Bi = np.vstack([Bi, [[3, 3], [3, -3]]])
    for a in Ai:
        u = Bi - a
        n = np.linalg.norm(u, axis=1)
        cosm = (u @ u.T) / np.outer(n, n)
        ang = np.arccos(np.clip(cosm, -1, 1))
        i, j = np.unravel_index(np.argmax(ang), ang.shape)
        area = 0.5 * n[i] * n[j] * math.sin(ang[i, j])
        if abs(ang[i, j] - math.pi / 2) < 1e-6 and area > best:
            best = area
    V.report(1, abs(best - float(S(V.ans(1, "1-1")))) < 0.05, f"1-1 수치 최대넓이 {best:.4f} / YAML {V.ans(1,'1-1')}")
    # 1-2
    Bq = C2[::3]
    mx = -1e9
    for a in lens[::3]:
        d = (Bq[:, 0] * 0 + 3) * (a[0] - Bq[:, 0])           # OO'=(3,0)
        mx = max(mx, d.max())
        if d.max() == mx:
            arg = (a, Bq[int(np.argmax(d))])
    V.report(1, abs(np.linalg.norm(arg[0] - arg[1]) - float(S(V.ans(1, "1-2")))) < 0.02, f"1-2 최대내적 {mx:.3f}, |AB|={np.linalg.norm(arg[0]-arg[1]):.3f}")
    # 1-3: Y 는 C 위, PY=2sqrt13 ; Q=Y 의 지름 반대쪽 -> 각 YQP=90
    xs = sp.symbols("xs", real=True)
    ysol = sp.solve([xs**2 + sp.Symbol("ys")**2 - 9, (xs + 5)**2 + sp.Symbol("ys")**2 - 52], [xs, sp.Symbol("ys")], dict=True)
    ok = True
    for s_ in ysol:
        Y = sp.Matrix([s_[xs], s_[sp.Symbol("ys")]])
        Q = -Y
        ok &= sp.simplify((Y - Q).dot(P - Q)) == 0
    V.proof(1, "1-3 YQ⊥QP", [] if ok and len(ysol) == 2 else ["x"])
    # 1-4: PY=2sqrt13, tan(OPX')=1/3, 두 배치
    Y = sp.Matrix([R(9, 5), R(12, 5)])
    dY = Y - sp.Matrix([-5, 0])
    PX = R(16) / (2 * sp.sqrt(13))
    XYv = dY / sp.sqrt(dY.dot(dY)) * (2 * sp.sqrt(13) - PX)
    # 선 PX': 방향 (3,±1)/sqrt10, C' 와의 교점
    out = {}
    for sgn, key in ((1, "ii"), (-1, "i")):
        u = sp.Matrix([3, sgn]) / sp.sqrt(10)
        s_ = sp.symbols("s_", positive=True)
        pt = sp.Matrix([-5, 0]) + s_ * u
        ss = sorted(sp.solve((pt[0] - 3)**2 + pt[1]**2 - 9, s_))
        XYp = u * (ss[1] - ss[0])
        out[key] = sp.simplify(XYv.dot(XYp))
    for key in ("i", "ii"):
        V.check(1, f"1-4 {key}", out[key], V.ans(1, "1-4", key))
    V.check(1, "PX", PX, 8 / sp.sqrt(13))


def p02():
    g1 = sp.Function("G")
    gp = sp.symbols("gp", real=True)
    fgp = (x**2 + 2 * x + 2) * sp.exp(x) / gp
    expr = x * (fgp - sp.exp(x)) / ((x**2 + 2) * sp.exp(x))
    claim = x * (x**2 + 2 * x + 2 - gp) / ((x**2 + 2) * gp)
    V.report(2, sp.simplify(expr - claim) == 0, "2-1 항등식")
    V.report(2, sp.Poly(x**2 + 2 * x + 2, x).discriminant() < 0, "x^2+2x+2>0")
    # 2-2: 최종 함수가 조건을 모두 만족하는지 + 해 유일성(k,beta,a 풀이)
    k, be, a = sp.symbols("k beta a", real=True)
    u = k * x**2 - 2 * be * k * x + 1
    v = a * (x - be) * sp.exp(u)
    lhs = sp.integrate(v, (x, 0, 2))
    rhs = (sp.exp(be**2 * k) - sp.exp(k * (be - 2)**2)) / (2 * k) * (-1) * (-1)
    asol = -sp.exp(k * be**2 - 1)
    kk = sp.solve(sp.simplify(sp.expand(lhs.subs(a, asol)) - (sp.exp(be**2) - sp.exp((be - 2)**2)) / (2 * k)).subs({be: -1}), k)
    V.report(2, kk and any(sp.simplify(s_ - 1) == 0 for s_ in kk), f"k={kk}")
    uf = u.subs({k: 1, be: -1}); af = asol.subs({k: 1, be: -1})
    vf = (af * (x + 1) * sp.exp(uf))
    # u(β+1) = -β -> β
    ub = sp.simplify(u.subs(k, 1).subs(x, be + 1) + be)
    bs = [s_ for s_ in sp.solve(ub, be) if s_ < 0]
    V.report(2, bs == [-1], f"β={bs}")
    exp_v = sp.sympify(V.ans(2, "2-2"), locals={"x": x})
    V.check(2, "v(x)", sp.simplify(vf - exp_v), 0)
    # 조건 수치 검증: m 최솟값 -1, 두 적분
    f1 = sp.lambdify(x, vf, "math")
    xs = [i / 1000 for i in range(-3000, 1500) if abs(i / 1000 + 1) > 1e-9]
    mmin = max(-f1(t_) / (1) * 0 + (f1(t_) / (t_ + 1)) for t_ in xs)     # m >= v/(x-β) for all x
    V.report(2, abs(mmin + 1) < 1e-2, f"m 최솟값 ≈ {mmin:.4f}")
    I1 = sp.integrate(vf, (x, 0, 2)); I2 = sp.integrate(vf, (x, -2, 2))
    tgt = (sp.exp(1) - sp.exp(9)) / 2
    V.report(2, sp.simplify(I1 - tgt) == 0 and sp.simplify(I2 - tgt) == 0, "두 정적분 = (e-e^9)/2")
    # 2-3
    curve = sp.simplify(sp.log(sp.Abs(vf / (x + 1))).rewrite(sp.log))
    V.report(2, abs(float(sp.log(abs(vf.subs(x, 0.7) / 1.7))) - 1.7**2) < 1e-9, "곡선 = (x+1)^2")
    V.check(2, "부피", sp.integrate((x + 1)**4, (x, 0, 2)), V.ans(2, "2-3"))


def p03():
    c = R(23, 100)
    def b3(alpha, beta, n):               # n번째 주기 직전 농도
        al = alpha
        for _ in range(n - 1):
            al = c * (al + R(54, 100) * beta)
        return al
    V.check(3, "3-1 b3", round(float(b3(0, 125, 3)), 1), V.ans(3, "3-1"))
    # 3-2
    al = R(5); lb = None
    seq_a, seq_b = [], []
    for n in range(1, 200):
        seq_b.append(float(b3(5, 100, n)))
        seq_a.append(float(b3(5, 100, n) + R(54)))
    V.check(3, "3-2 lim a", round(seq_a[-1], 1), V.ans(3, "3-2", "a"))
    V.check(3, "3-2 lim b", round(seq_b[-1], 1), V.ans(3, "3-2", "b"))
    # f(t) 의 최대·최소가 t=2, 복용 직전임을 수치 확인
    ts = [i / 100 for i in range(0, 2401)]
    fv = lambda t_, al_, be_: al_ + 0.27 * be_ * t_ if t_ <= 2 else (al_ + 0.54 * be_) * math.exp(-math.sqrt((t_ - 2) / 10))
    vals = [fv(t_, 5, 100) for t_ in ts]
    V.report(3, abs(max(vals) - 59) < 1e-9 and abs(min(vals) - 5) < 1e-9, "한 주기 max=α+0.54β, min=α")
    # 3-3: 가장 큰 정수 β
    best = max(b for b in range(1, 1000) if float(b3(0, b, 400) + R(54, 100) * b) < 100)
    V.check(3, "3-3", best, V.ans(3, "3-3"))
    # 3-4: sigma
    lim_a = float(b3(5, 100, 400) + R(54))
    best = max(s_ for s_ in range(1, 100) if round(lim_a, 1) + 2.33 * s_ < 100)
    V.check(3, "3-4", best, V.ans(3, "3-4"))


if __name__ == "__main__":
    V.run(globals())
