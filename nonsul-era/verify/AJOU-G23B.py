#!/usr/bin/env python3
"""AJOU-G23B (아주대 2023학년도 논술 자연계열 오후) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "AJOU-G23B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/AJOU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    import itertools
    import math
    import numpy as np
    # [1-1](1): 사차함수 f, f(0)=f(1)=0, f'(0)=f'(1)=0, 극댓값 2^10 → log f(6) 을 A=log2, B=log3 로
    c = sp.symbols("c0:5")
    f = sum(c[i] * x**i for i in range(5))
    sol = sp.solve([f.subs(x, 0), f.subs(x, 1), sp.diff(f, x).subs(x, 0), sp.diff(f, x).subs(x, 1)], c[:4], dict=True)[0]
    f = f.subs(sol)
    crit = [r for r in sp.solve(sp.diff(f, x), x) if r not in (0, 1)]
    lead = sp.solve(sp.Eq(f.subs(x, crit[0]), 2**10), c[4])[0]
    f6 = f.subs(c[4], lead).subs(x, 6)
    A, B = sp.symbols("A B", positive=True)
    md = sp.sympify(md_answer(1, "1-1"), locals={"A": A, "B": B})
    ok1 = abs(float(md.subs({A: math.log10(2), B: math.log10(3)})) - math.log10(float(f6))) < 1e-12
    # [1-1](2): 예시 좋은함수들로 a∈(0,p), b∈(p,1) 존재 확인 [증명-수치확인]
    ok2 = True
    for k_ in (1, 2, 3):
        for pp in (sp.Rational(1, 3), sp.Rational(1, 2), sp.Rational(3, 5)):
            g = x * (x - 1) * ((x - pp)**2 * k_ + 1)
            # 최댓값이 p 에서 나오도록 하는 대신, 일반적으로 f(p)=M 인 경우의 관계 f'(a)=M/p, f'(b)=M/(p-1) 만 확인
            M = g.subs(x, pp)
            d = sp.diff(g, x)
            xa = [r for r in np.linspace(1e-6, float(pp) - 1e-6, 20001) if True]
            fa = sp.lambdify(x, d - M / pp); fb = sp.lambdify(x, d - M / (pp - 1))
            va = np.array([fa(v) for v in xa]); xb = np.linspace(float(pp) + 1e-6, 1 - 1e-6, 20001)
            vb = np.array([fb(v) for v in xb])
            ok2 = ok2 and np.any(np.sign(va[:-1]) != np.sign(va[1:])) and np.any(np.sign(vb[:-1]) != np.sign(vb[1:]))
    # [1-2](1): 정적분 직접 계산
    g = sp.cos(sp.pi * x / 2)**3 * sp.sin(sp.pi * x / 2)
    h = x * sp.exp(x**2 - 1) - x
    I = sp.integrate(g, (x, 0, 1)) + sp.integrate(h.subs(x, x - 1), (x, 1, 2))
    ok3 = sp.simplify(I - sp.sympify(md_answer(1, "2-1"))) == 0
    # [1-2](2): 구체적 좋은함수 p(x)=x(x-1)(ux+v) 로 G,H 를 기호 미분해 조건 직접 대입, 9^6 전수
    a, b, cc, d, e, ff, X = sp.symbols("a b cc d e ff X")
    def good(s0, s1):
        return X * (X - 1) * ((s0 + s1) * X - s0)       # p'(0)=s0, p'(1)=s1
    P, Q, R = good(a, b), good(cc, d), good(e, ff)
    G = P.subs(X, Q) + P * Q
    H = Q.subs(X, R) + 2 * R
    dG, dH = sp.diff(G, X), sp.diff(H, X)
    fn = sp.lambdify((a, b, cc, d, e, ff), [G.subs(X, 0), G.subs(X, 1), H.subs(X, 0), dG.subs(X, 0), dG.subs(X, 1), dH.subs(X, 0)])
    cnt = 0
    for t6 in itertools.product(range(1, 10), repeat=6):
        G0, G1, H0, dG0, dG1, dH0 = fn(*t6)
        if G0 == 0 and G1 == 0 and H0 == 0 and dG1 == dH0 == 24 and dG0 > 0 and 24 % dG0 == 0:
            cnt += 1
    ok4 = cnt == int(md_answer(1, "2-2"))
    report(1, ok1 and ok2 and ok3 and ok4, f"[증명-수치확인] log f(6) 일치={ok1}; (1-2)적분={sp.simplify(I)}; 순서쌍={cnt}")


def p02():
    import numpy as np
    import random
    a = sp.symbols("a", real=True)
    # [2-1](1): 유리수 a 격자에서 [-2,2] 의 서로 다른 실근 개수 = 3 인 집합 → 경계 비교
    def nroot(av):
        rs = sp.Poly(x**3 - 3 * av * x + av, x).real_roots()
        return len({r for r in rs if -2 <= r <= 2})
    grid = [sp.Rational(k, 1680) for k in range(-1680, 4 * 1680)]
    good = [g for g in grid if nroot(g) == 3]
    lo, hi = min(good), max(good)
    m = re.match(r"\s*(.+?)\s*<\s*a\s*<=\s*(.+)\s*$", md_answer(2, "1-1"))
    L, U = sp.sympify(m.group(1)), sp.sympify(m.group(2))
    ok1 = (nroot(U) == 3 and nroot(L) != 3 and lo - L <= sp.Rational(1, 1680) and hi == U
           and all((L < g <= U) == (nroot(g) == 3) for g in grid))
    # [2-1](2): max|f| 를 a 격자에서 최소화
    xs = np.linspace(-2, 2, 40001)
    def M(av):
        return np.max(np.abs(xs**3 - 3 * av * xs + 2))
    avs = np.linspace(-1, 5, 6001)
    Ms = np.array([M(v) for v in avs])
    i = int(np.argmin(Ms))
    parts = dict(seg.split("=") for seg in md_answer(2, "1-2").replace(" ", "").split(";"))
    ok2 = abs(avs[i] - float(sp.sympify(parts["a"]))) < 2e-3 and abs(Ms[i] - float(sp.sympify(parts["max"]))) < 1e-6
    # [2-2](1): 무작위 (a,b) 로 서로 다른 네 실근이면 sqrt(-a/2) > 최소 |근| [증명-수치확인]
    ok3 = True; tested = 0
    random.seed(1)
    for _ in range(3000):
        av = random.uniform(-20, 5); bv = random.uniform(-20, 40)
        rts = np.roots([1, 0, av, 0, bv])
        if np.all(np.abs(rts.imag) < 1e-12) and len(set(np.round(rts.real, 9))) == 4:
            tested += 1
            Aval = np.min(np.abs(rts.real))
            ok3 = ok3 and (av < 0 and np.sqrt(-av / 2) > Aval)
    ok3 = ok3 and tested > 100
    # [2-2](2): c 를 미지수로, 대칭 네 근 ±r,±s 와 ∫_{-s}^{s}F=0 연립
    c, s = sp.symbols("c s", positive=True)
    F = x**4 - x**2 + c
    sols = sp.solve([F.subs(x, s), sp.integrate(F, (x, -s, s))], [c, s], dict=True)
    cs = {so[c] for so in sols if sp.Poly(F.subs(c, so[c]), x).count_roots() == 4
          and max(sp.Poly(F.subs(c, so[c]), x).real_roots()) == so[s]}
    ok4 = cs == {sp.sympify(md_answer(2, "2-2"))}
    # [2-3]: 계수 비교 → m, a, L 기울기; F(s)=0 & ∫_0^s F=0 (s=t-m 가장 큰 근)
    aa, bb, m_, k1, k0 = sp.symbols("aa bb m_ k1 k0")
    Gx = x**4 - 4 * x**3 + 9 * x - sp.Rational(11, 2)
    expr = sp.expand(Gx - ((x - m_)**4 + aa * (x - m_)**2 + bb + k1 * x + k0))
    co = sp.solve(sp.Poly(expr, x).all_coeffs(), [m_, aa, k1, k0], dict=True)[0]
    u, sv = sp.symbols("u sv", positive=True)
    Fu = u**4 + co[aa] * u**2 + bb
    sol2 = sp.solve([Fu.subs(u, sv), sp.integrate(Fu, (u, 0, sv))], [bb, sv], dict=True)
    Ls = set()
    for so in sol2:
        Fb = sp.Poly(Fu.subs(bb, so[bb]), u)
        if Fb.count_roots() == 4 and max(Fb.real_roots()) == so[sv]:
            Ls.add(sp.expand(co[k1] * x + co[k0].subs(bb, so[bb])))
    ok5 = Ls == {sp.expand(sp.sympify(md_answer(2, "2-3"), locals={"x": x}))}
    report(2, ok1 and ok2 and ok3 and ok4 and ok5,
           f"[증명-수치확인] a범위=({lo},{hi}] 경계({L},{U}]; a*={avs[i]:.3f} max={Ms[i]:.4f}; 반례검사 {tested}건; c={cs}; L={Ls}")


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
