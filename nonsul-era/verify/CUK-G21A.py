#!/usr/bin/env python3
"""CUK-G21A (가톨릭대 2021 수시 논술 자연·공학/간호 1~3) — 문제 조건에서 독립 재계산."""
import sys, pathlib, itertools, random
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp
V = Verifier("CUK-G21A", "items/CUK/PAST")
mp.mp.dps = 30


def p01():
    x, y, a, b, c, m, k, t = sp.symbols("x y a b c m k t", real=True)
    # 논제 1: 원점 대칭 (x,y)->(-x,-y), 평행이동 x->x-1, y->y+6 을 변환식 그대로 적용
    X, Y = sp.symbols("X Y")
    # 원 도형 y=ax^2+bx+2 의 점 (u, a u^2+b u+2) 의 상 (X,Y)
    u = sp.symbols("u")
    Xi = -u + 1
    Yi = -(a * u**2 + b * u + 2) - 6
    ux = sp.solve(sp.Eq(Xi, x), u)[0]
    img = sp.expand(Yi.subs(u, ux) - (-x**2 + 4 * x + c))
    sol = sp.solve(sp.Poly(img, x).all_coeffs(), [a, b, c], dict=True)[0]
    got = f"a={sol[a]}; b={sol[b]}; c={sol[c]}"
    ex = {k_: sp.Rational(v) for k_, v in (p.split("=") for p in V.ans(1, "1").replace(" ", "").split(";"))}
    V.report(1, ex == {"a": sol[a], "b": sol[b], "c": sol[c]}, f"(논제1) {got} / YAML {V.ans(1,'1')}")
    # 논제 2: 공통접선을 직선 y=mx+k 로 두고 두 포물선에 접할 조건(중근)으로 독립 계산
    f1 = sol[a] * x**2 + sol[b] * x + 2
    f2 = -x**2 + 4 * x + sol[c]
    d1 = sp.discriminant(sp.expand(f1 - (m * x + k)), x)
    d2 = sp.discriminant(sp.expand(f2 - (m * x + k)), x)
    sols = sp.solve([d1, d2], [m, k], dict=True)
    l1 = [s for s in sols if s[m] < 0]; l2 = [s for s in sols if s[m] > 0]
    assert len(sols) == 2 and len(l1) == 1 and len(l2) == 1
    L1 = l1[0][m] * x + l1[0][k]; L2 = l2[0][m] * x + l2[0][k]
    xi = sp.solve(sp.Eq(L1, L2), x)[0]
    # 둘러싸인 도형 = 곡선 f1 과 max(L1,L2) 사이 (접점 사이)
    t1 = sp.solve(sp.Eq(sp.diff(f1, x), l1[0][m]), x)[0]; t2 = sp.solve(sp.Eq(sp.diff(f1, x), l2[0][m]), x)[0]
    S = sp.integrate(f1 - L1, (x, t1, xi)) + sp.integrate(f1 - L2, (x, xi, t2))
    # 수치 확인: f1 - max(L1,L2) 의 [t1,t2] 적분
    Sn = mp.quad(lambda z: f1.subs(x, z) - max(L1.subs(x, z), L2.subs(x, z)), [t1, xi, t2])
    exp = sp.Rational(V.ans(1, "2"))
    V.report(1, S == exp and abs(mp.mpf(str(sp.N(Sn, 25))) - mp.mpf(str(sp.N(exp, 25)))) < 1e-15 and (t1, t2) == (-2, 3) and (L1, L2) == (-2*x - 2, 8*x - 7),
             f"(논제2) l1={L1}, l2={L2}, 접점 {t1},{t2}, S={S} / YAML {exp}")


def p02():
    x, t = sp.symbols("x t", real=True)
    al, be = sp.symbols("alpha beta", real=True)
    # 논제 1: 실계수 → 서로 다른 두 허근은 켤레: conj(2z+9) = -z-3i/4, z=al+be i
    z = al + be * sp.I
    e = sp.expand(sp.conjugate(2 * z + 9) - (-z - sp.Rational(3, 4) * sp.I))
    s = sp.solve([sp.re(e), sp.im(e)], [al, be], dict=True)[0]
    zz = z.subs(s)
    r1, r2 = sp.expand(2 * zz + 9), sp.expand(-zz - sp.Rational(3, 4) * sp.I)
    quad = sp.expand((x - r1) * (x - r2))
    A_, B_ = sp.symbols("A B", real=True)
    f = B_ * (x - A_) * quad
    sol = sp.solve([sp.Eq(f.subs(x, sp.Rational(1, 2)), sp.Rational(17, 4)), sp.Eq(sp.diff(f, x, 2).subs(x, A_), -12)], [A_, B_], dict=True)
    sol = [q for q in sol if q[B_] != 0]
    assert len(sol) == 1
    F = sp.expand(f.subs(sol[0]))
    V.report(2, sp.expand(F - sp.sympify(V.ans(2, "1"), locals={"x": x})) == 0 and F.subs(x, sp.Rational(1, 2)) == sp.Rational(17, 4),
             f"(논제1) f={F} / YAML {V.ans(2,'1')}")
    # 논제 2: g=f+36/f on [1,2], f>0 확인, 고정밀 격자 + 임계점 정확값
    g = F + 36 / F
    pts = [sp.Integer(1), sp.Integer(2)] + [r for r in sp.solve(sp.diff(g, x), x) if r.is_real and 1 <= r <= 2]
    vals = [sp.nsimplify(g.subs(x, p_)) for p_ in pts]
    grid = [mp.mpf(1) + mp.mpf(i) / 20000 for i in range(20001)]
    gf = sp.lambdify(x, g, "mpmath")
    gv = [gf(p_) for p_ in grid]
    M, m_ = max(vals), min(vals)
    ok = abs(mp.mpf(str(sp.N(M, 25))) - max(gv)) < 1e-6 and abs(mp.mpf(str(sp.N(m_, 25))) - min(gv)) < 1e-6
    V.report(2, ok and sp.simplify(M - sp.Rational(V.ans(2, "2", "M"))) == 0 and sp.simplify(m_ - sp.Rational(V.ans(2, "2", "m"))) == 0,
             f"(논제2) M={M}, m={m_} / YAML {V.ans(2,'2')}")


def p03():
    def nxt(a, b):
        s = a + b
        return 3 * s + 1 if s % 2 else s // 2

    def seq(a1, a2, n=14):
        v = [a1, a2]
        while len(v) < n:
            v.append(nxt(v[-2], v[-1]))
        return v
    # 논제 1: 명제 p [증명-수치확인] — 모든 (a1,a2) 와 k 에 대해 반례 탐색
    bad = []
    for a1 in range(1, 80):
        for a2 in range(1, 80):
            v = seq(a1, a2)
            for k in range(1, 12):
                if v[k - 1] % 2 and v[k + 1] % 2 and not all(v[i] % 2 for i in range(k + 2)):
                    bad.append((a1, a2, k))
    V.proof(3, "명제 p(k=1..11, a1,a2<80)", bad)
    # 논제 2: a3=19, a6=22 인 모든 a1 (a1,a2 양의 정수 전수)
    A = sorted({a1 for a1 in range(1, 600) for a2 in range(1, 600) if seq(a1, a2)[2] == 19 and seq(a1, a2)[5] == 22})
    txt = V.ans(3, "2").strip().strip("{}")
    exp = sorted(int(s) for s in txt.split(",") if s.strip())
    V.report(3, A == exp, f"(논제2) A={A} / YAML {V.ans(3,'2')}")


if __name__ == "__main__":
    V.run(globals())
