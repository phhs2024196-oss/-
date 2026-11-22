#!/usr/bin/env python3
"""KAU-G26E-01~02 (한국항공대 2026 공학계열) — sympy/mpmath 독립 재계산. 기대값은 YAML 의 answer.value."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
import mpmath as mp  # noqa: E402
from sympy.parsing.sympy_parser import parse_expr  # noqa: E402
from vlib import *  # noqa: E402,F403

V = Verifier("KAU-G26E", "items/KAU/PAST")
mp.mp.dps = 30


def P(s, **loc):
    return parse_expr(s, local_dict={"x": x, "t": t, "r": sp.Symbol("r", positive=True), **loc})


def parts(no, sub):
    return dict(p.split("=", 1) for p in V.ans(no, sub).replace(" ", "").split(";"))


def close(a, b, tol=1e-12):
    return abs(complex(sp.N(a - b, 30))) < tol


def numsame(ea, eb, var, pts):
    return all(close(ea.subs(var, p), eb.subs(var, p), 1e-12) for p in pts)


def p01():
    # 1-1 (1)(2)(3)
    num, den = x**3 - 4 * x**2 + 5 * x - 2, x**2 - 1
    a = sp.limit(num / den, x, 1)
    V.check(1, "1-1(1) a=lim f(x)", a, V.ans(1, "1"))
    h = sp.Symbol("h", real=True)
    f_def = sp.cancel(num / den)                      # x≠1 에서의 식(연속 확장 값 a 와 일치해야 함)
    fp1 = sp.limit((f_def.subs(x, 1 + h) - a) / h, h, 0)   # 미분계수의 정의
    V.check(1, "1-1(2) f'(1) (정의로 계산)", fp1, V.ans(1, "2"))
    V.check(1, "1-1(2) f'(1) (몫의 미분 후 대입)", sp.limit(sp.diff(num / den, x), x, 1), V.ans(1, "2"))
    crit = [c for c in sp.solve(sp.numer(sp.cancel(sp.diff(num / den, x))), x) if c.is_real and c > 0]
    x0 = P(V.ans(1, "3").split(">=")[1])
    ok = len(crit) == 1 and close(crit[0], x0) and all(sp.diff(f_def, x).subs(x, c) > 0 for c in (x0 + 1, x0 + 5)) \
        and all(sp.diff(f_def, x).subs(x, c) < 0 for c in (x0 / 2, x0 * sp.Rational(9, 10)))
    V.report(1, ok, f"1-1(3) 증가 구간 [{x0},oo) : f'=0 근 {crit}")
    # 1-2
    X, Y = (1 + sp.cos(2 * t)) * sp.cos(2 * t), (1 + sp.cos(2 * t)) * sp.sin(2 * t)
    dydx = sp.diff(Y, t) / sp.diff(X, t)
    V.report(1, numsame(dydx, P(V.ans(1, "4"), t=t), t, [sp.Rational(3, 10), sp.Rational(7, 10), sp.Rational(11, 10), 2]),
             "1-2(1) dy/dx 수치 대조")
    speed = sp.sqrt(sp.diff(X, t) ** 2 + sp.diff(Y, t) ** 2)
    grid = [mp.pi / 2 + (mp.pi) * i / 4000 for i in range(4001)]
    sf = sp.lambdify(t, speed, "mpmath")
    vals = [sf(g) for g in grid]
    tmax = grid[max(range(len(vals)), key=vals.__getitem__)]
    ans5 = parts(1, "5")
    V.report(1, abs(tmax - mp.pi) < 1e-3 and all(close(speed.subs(t, p), -4 * sp.cos(p)) for p in [1.7, 2.2, 3.0, 4.0, 4.6]),
             f"1-2(2) 속력 = -4cos t 수치확인, 최대 시각 격자 {mp.nstr(tmax, 6)} vs {ans5['t']}")
    tstar = P(ans5["t"])
    pos = (sp.simplify(X.subs(t, tstar)), sp.simplify(Y.subs(t, tstar)))
    V.check(1, "1-2(2) 위치 x", pos[0], P(ans5["P"].strip("()").split(",")[0]))
    V.check(1, "1-2(2) 위치 y", pos[1], P(ans5["P"].strip("()").split(",")[1]))
    dist = mp.quad(sf, [mp.pi / 2, mp.pi, 3 * mp.pi / 2])
    V.check(1, "1-2(3) 이동 거리(수치적분)", sp.Float(dist, 25), V.ans(1, "6"), tol=1e-12)
    # 1-3
    th = sp.Symbol("theta", real=True)

    def I(a_):
        g = sp.lambdify(th, sp.sin(th) / (1 + a_ * sp.cos(th)), "mpmath")
        return mp.quad(g, [0, mp.pi / 2])
    V.check(1, "1-3(1) I(2) (수치적분)", sp.Float(I(2), 25), P(V.ans(1, "7"), ln=sp.log, **{"log": sp.log}), tol=1e-12)
    pts = [sp.Rational(1, 10), sp.Rational(1, 2), 1, 3, 10, 50]
    V.report(1, all(abs(I(p) - mp.log(1 + p) / p) < 1e-20 for p in pts), "1-3 I(a)=ln(1+a)/a 수치 대조(a=0.1..50)")
    lim0 = sp.limit(sp.log(1 + x) / x, x, 0, "+")
    cont = "연속" if close(lim0, I(0)) else "불연속"
    V.report(1, cont == V.ans(1, "8"), f"1-3(2) lim_(x→0+)I={lim0}, I(0)={mp.nstr(I(0), 8)} → {cont}")
    Ix = sp.log(1 + x) / x
    dnum = sp.numer(sp.together(sp.diff(Ix, x)))
    grid2 = [sp.Rational(i, 20) for i in range(1, 400)]
    mono = "감소" if all(sp.diff(Ix, x).subs(x, g) < 0 for g in grid2) and all(dnum.subs(x, g) < 0 for g in grid2) \
        and all(I(grid2[i]) > I(grid2[i + 40]) for i in range(0, 300, 30)) else "비단조"
    V.report(1, mono == V.ans(1, "9"), f"1-3(3) x>0 에서 I' 분자 부호·수치 적분 비교 → {mono}")


def p02():
    r = sp.Symbol("r", positive=True)
    import math

    def tri(rv):
        """a=1,b=r,c=r²인 삼각형을 좌표로 세워 각을 벡터로 직접 계산(코사인법칙 비사용)."""
        a_, b_, c_ = 1.0, rv, rv * rv
        A = (0.0, 0.0)
        B = (c_, 0.0)
        cx = (b_ * b_ + c_ * c_ - a_ * a_) / (2 * c_)       # C 의 x 좌표(두 원의 교점)
        cy = math.sqrt(b_ * b_ - cx * cx)
        C = (cx, cy)

        def ang(P_, Q_, R_):
            v1, v2 = (Q_[0] - P_[0], Q_[1] - P_[1]), (R_[0] - P_[0], R_[1] - P_[1])
            return math.acos((v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(*v1) * math.hypot(*v2)))
        return ang(A, B, C), ang(B, A, C), ang(C, A, B)
    ok1 = ok2 = ok3 = ok4 = True
    e1, e2, e3, e4 = (P(V.ans(2, s), r=r) for s in ("1", "2", "3", "4"))
    for rv in (0.7, 0.9, 1.0, 1.2, 1.4):
        A_, B_, C_ = tri(rv)
        ok1 &= abs(math.cos(A_) - float(e1.subs(r, rv))) < 1e-9
        ok2 &= abs(math.cos(B_) - float(e2.subs(r, rv))) < 1e-9
        ok3 &= abs(math.sin(B_) / math.sin(A_) - float(e3.subs(r, rv))) < 1e-9
        ok4 &= abs(math.sin(C_) / math.sin(A_) - float(e4.subs(r, rv))) < 1e-9
    V.report(2, ok1 and ok2 and ok3 and ok4, "2-1(1)~(4) 좌표로 만든 삼각형의 각과 대조(r=0.7~1.4)")
    # (5): 2sinA = sin(B-A)+sinC 의 근을 (0.618,1.618) 에서 격자·이분법으로 찾아 r² 와 비교
    def F(rv):
        A_, B_, C_ = tri(rv)
        return 2 * math.sin(A_) - math.sin(B_ - A_) - math.sin(C_)
    xs = [0.62 + i * (1.617 - 0.62) / 2000 for i in range(2001)]
    roots = []
    for u, v in zip(xs, xs[1:]):
        if F(u) * F(v) < 0:
            lo, hi = u, v
            for _ in range(80):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if F(lo) * F(mid) > 0 else (lo, mid)
            roots.append((lo + hi) / 2)
    exp5 = P(V.ans(2, "5"))
    V.report(2, len(roots) == 1 and abs(roots[0] ** 2 - float(exp5)) < 1e-9, f"2-1(5) 등차 조건의 근 격자탐색 r={roots}, r²={[q*q for q in roots]} vs {exp5}")
    V.check(2, "2-1(5) r⁴-r²-1=0 의 양의 근", sp.solve(r**4 - r**2 - 1, r)[0] ** 2, exp5)
    # 2-2 좌표: A(0,0) B(s,0) C(s,s) D(0,s), P=(px,py)
    s_, px, py = sp.symbols("s px py", positive=True)
    sol = sp.solve([px**2 + py**2 - 25, (px - s_) ** 2 + py**2 - 64, (px - s_) ** 2 + (py - s_) ** 2 - 169], [s_, px, py], dict=True)
    inside = [q for q in sol if 0 < q[px] < q[s_] and 0 < q[py] < q[s_]]
    V.report(2, len(inside) == 1, f"2-2 정사각형 내부의 점 P 조건을 만족하는 해 {len(inside)}개")
    q = inside[0]
    S_, PX, PY = q[s_], q[px], q[py]
    Pt, Bt, Ct = sp.Matrix([PX, PY]), sp.Matrix([S_, 0]), sp.Matrix([S_, S_])
    Qt = Bt + sp.Matrix([PY, S_ - PX])                      # B 중심 시계 방향 90° 회전: (u,v)→(v,-u)
    PQ = sp.sqrt(((Pt - Qt).T * (Pt - Qt))[0])
    V.check(2, "2-2(1) PQ", sp.simplify(PQ), V.ans(2, "6"))
    V.check(2, "2-2 QC=PA=5", sp.simplify(sp.sqrt(((Qt - Ct).T * (Qt - Ct))[0])), 5)

    def cos_ang(O, U, W):
        a_, b_ = U - O, W - O
        return sp.simplify((a_.T * b_)[0] / (sp.sqrt((a_.T * a_)[0]) * sp.sqrt((b_.T * b_)[0])))

    def sin_ang(O, U, W):
        a_, b_ = U - O, W - O
        return sp.simplify(abs(a_[0] * b_[1] - a_[1] * b_[0]) / (sp.sqrt((a_.T * a_)[0]) * sp.sqrt((b_.T * b_)[0])))
    V.check(2, "2-2(2) cos∠CPQ", cos_ang(Pt, Ct, Qt), V.ans(2, "7"))
    V.check(2, "2-2(3) sin∠CPQ", sin_ang(Pt, Ct, Qt), V.ans(2, "8"))
    V.check(2, "2-2(4) cos∠BPC", cos_ang(Pt, Bt, Ct), V.ans(2, "9"))
    V.check(2, "2-2(5) 넓이 = s²", sp.simplify(S_**2), V.ans(2, "10"))
    V.check(2, "2-2(5) 코사인법칙 BC²=PC²+PB²-2·PC·PB·cos∠BPC", 169 + 64 - 2 * 13 * 8 * cos_ang(Pt, Bt, Ct), V.ans(2, "10"))


if __name__ == "__main__":
    V.run(globals())
