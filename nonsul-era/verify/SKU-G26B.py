#!/usr/bin/env python3
"""서경대 2026학년도 수시 논술 B형 수학 5~8번 — sympy 검산(기대값은 YAML 정답에서 읽음)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("SKU-G26B", "items/SKU/PAST")


def E(s, **loc):
    return S(s, locals={"x": x, "k": k, **loc})


def p05():
    # 정의를 그대로 전수 계산
    a = {}
    for n in range(1, 51):
        a[n] = 2 * n + 1 if n % 2 == 1 else a[n - 1] + 1
    V.check(5, "sum a_k (k=1..50)", sum(a.values()), V.ans(5, "1"))
    # 공식 해설의 일반항 확인
    kk = sp.symbols("kk", integer=True, positive=True)
    V.check(5, "sum(8k-1,k=1..25)", sp.summation(8 * kk - 1, (kk, 1, 25)), V.ans(5, "1"))


def p06():
    # f=f1+(x-1)*..., g 를 미지의 1차식으로 놓고 조건에서 값·기울기를 풀어 확인
    f0, f1, g0, g1 = sp.symbols("f0 f1 g0 g1")
    f = f0 + f1 * (x - 1)
    g = g0 + g1 * (x - 1)
    # 극한이 존재하려면 x=1 에서 분자 0
    s0 = sp.solve([(f - g).subs(x, 1), (f + g - 4).subs(x, 1)], [f0, g0])
    V.check(6, "1 f(1)", s0[f0], V.ans(6, "1"))
    V.check(6, "2 g(1)", s0[g0], V.ans(6, "2"))
    fs, gs = f.subs(s0), g.subs(s0)
    s1 = sp.solve([sp.limit((fs - gs) / (x - 1), x, 1) - 2, sp.limit((fs + gs - 4) / (x - 1), x, 1) - 6], [f1, g1])
    V.check(6, "3 f'(1)", s1[f1], V.ans(6, "3"))
    V.check(6, "4 g'(1)", s1[g1], V.ans(6, "4"))
    h = sp.expand((fs * gs).subs(s1))
    V.check(6, "5 h'(1)", sp.diff(h, x).subs(x, 1), V.ans(6, "5"))


def p07():
    f = x**2 / 2 + 2 * x + 5
    m = -1 / sp.diff(f, x).subs(x, -4)       # 수직인 직선의 기울기
    l = m * (x + 4) + f.subs(x, -4)
    V.check(7, "1 l(x)", sp.expand(l), E(V.ans(7, "1")))
    roots = sorted(sp.solve(f - l, x))
    area = sp.integrate(l - f, (x, roots[0], roots[1]))
    V.check(7, "교점이 -4, 1", roots, [-4, 1])
    V.check(7, "l>=f on [-4,1]", all((l - f).subs(x, t) >= 0 for t in [-3, -2, 0, 1]), True)
    V.check(7, "2 넓이", area, V.ans(7, "2"))


def p08():
    kp = sp.symbols("kp", positive=True)
    f = lambda X: 3 * sp.cos(2 * kp * X)
    # 주기: f(x+T)=f(x) 가 모든 x 에서 성립하는 최소 양수 T = 2π/(2k)
    g = sp.pi / kp
    ok = all(sp.simplify(f(t + g) - f(t)) == 0 for t in (0, 1, sp.Rational(1, 3)))
    V.check(8, "g 가 주기", ok, True)
    V.check(8, "1 g(k)", g, E(V.ans(8, "1"), pi=sp.pi).subs(k, kp))
    h = f(g) + sp.cos(g / 2 + sp.Rational(4, 3) * sp.pi)
    V.check(8, "2 h(k)", sp.simplify(h), sp.simplify(E(V.ans(8, "2"), pi=sp.pi).subs(k, kp)))
    # 각 θ(k)=π/(2k)+4π/3 의 값이 cos 최대(2π의 배수)/최소(홀수 π) 가 되는 k 중 최대
    th = lambda kv: sp.pi / (2 * kv) + sp.Rational(4, 3) * sp.pi
    ks_max = [sp.solve(sp.Eq(th(kp), 2 * sp.pi * n), kp)[0] for n in range(1, 7)]
    ks_min = [sp.solve(sp.Eq(th(kp), sp.pi * (2 * n + 1)), kp)[0] for n in range(1, 7)]
    V.check(8, "3 a=max k (cos=1)", max(ks_max), V.ans(8, "3"))
    V.check(8, "4 b=max k (cos=-1)", max(ks_min), V.ans(8, "4"))
    V.check(8, "h(a)=4", sp.simplify(h.subs(kp, max(ks_max))), 4)
    V.check(8, "h(b)=2", sp.simplify(h.subs(kp, max(ks_min))), 2)


if __name__ == "__main__":
    V.run(globals())
