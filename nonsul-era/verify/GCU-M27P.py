#!/usr/bin/env python3
"""가천대 2027학년도 의약학 논술 모의고사 한의예·약학 수학(문제 1~4) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산."""
import re
import sys

import sympy as sp

PREFIX = "GCU-M27P"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)




def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    a = sp.symbols("a", positive=True)
    b = 2 * a / 3                                   # 9^a = 27^b
    vals = []
    for sgn in (1, -1):
        for av in sp.solve(sp.Eq(a + b - a * b**2, sgn), a):
            if av.is_real and av > 0:
                vals.append(sp.nsimplify(sp.Abs(av**2 - 3 * b.subs(a, av))))
    vals = [sp.radsimp(v) for v in vals]
    mx, mn = max(vals, key=lambda v: float(v)), min(vals, key=lambda v: float(v))
    ok = sp.simplify(mx - (7 * sp.sqrt(33) - 33) / 8) == 0 and sp.simplify(mn - (9 - sp.sqrt(33)) / 8) == 0
    report(1, ok, f"|a^2-3b| 후보={vals}, 최댓값={mx}, 최솟값={mn} (YAML (7√33-33)/8, (9-√33)/8)")


def p02():
    import mpmath as mpm
    # 조건을 만족하는 g 두 개(대칭 조건 g(x)+g(π/4-x)=3)로 f 를 구성: f = h'' + h (h = 우변), 적분값이 g 에 무관하게 7π/16 인지
    t = sp.symbols("t", real=True)
    res = []
    for g in (sp.Rational(3, 2) + 0 * x, sp.Rational(3, 2) + (x - sp.pi / 8)**3, sp.Rational(3, 2) + sp.sin(8 * x)):
        assert sp.simplify(g + g.subs(x, sp.pi / 4 - x) - 3) == 0
        h = 1 - sp.cos(4 * x) + sp.sin(4 * x)**2 * g
        f = sp.diff(h, x, 2) + h
        # 원래 적분방정식 확인(수치): ∫_0^x sin(x-t) f(t) dt = h(x)
        fl, hl = sp.lambdify(x, f, "mpmath"), sp.lambdify(x, h, "mpmath")
        chk = abs(mpm.quad(lambda tt: mpm.sin(0.5 - tt) * fl(tt), [0, 0.5]) - hl(0.5)) < 1e-12
        res.append((chk, sp.nsimplify(sp.N(sp.integrate(f, (x, 0, sp.pi / 4)), 30), [sp.pi])))
    ok = all(c for c, _ in res) and {v for _, v in res} == {7 * sp.pi / 16}
    report(2, ok, f"(방정식 확인, ∫f)={res} (YAML 7π/16)")


def p03():
    a, b, c, d = sp.symbols("a b c d", real=True)
    # f = x^3 + b x^2 + c x + d, 변곡점 x=a+1/3, f(a)=g(a) (h(a)=1), h'(a)=0
    f = x**3 + b * x**2 + c * x + d
    sols = []
    for av in (sp.Rational(1, 2), 1, 2, sp.Rational(7, 3)):      # 여러 양수 a 에 대해 넓이가 같은지
        e1 = sp.Eq(sp.diff(f, x, 2).subs(x, av + sp.Rational(1, 3)), 0)
        # f 증가(역함수 존재)이면 f(f(a))=a ⇒ f(a)=a ; h'(a)=0 ⇒ 1/f'(a) = f'(a) ⇒ f'(a)=1
        e2 = sp.Eq(f.subs(x, av), av)
        e3 = sp.Eq(sp.diff(f, x).subs(x, av), 1)
        s = sp.solve([e1, e2, e3], [b, c, d], dict=True)[0]
        fs = f.subs(s)
        mono = sp.discriminant(sp.diff(fs, x), x) <= 0               # 증가함수 확인
        # h(a)=1, h'(a)=0 수치 확인 (g 는 f 의 역함수)
        g_at = lambda y: sp.nsolve(fs - y, x, av)
        hval = g_at(av) / fs.subs(x, av)
        rts = sorted(set(sp.real_roots(sp.Poly(fs - x, x))))
        area = 2 * sp.integrate(sp.Abs(fs - x), (x, rts[0], rts[-1]))
        sols.append((av, bool(mono), float(hval), sp.nsimplify(area)))
    ok = all(m and abs(hv - 1) < 1e-12 and ar == sp.Rational(1, 6) for _, m, hv, ar in sols)
    report(3, ok, f"(a, 증가, h(a), 넓이)={sols} (YAML 1/6)")


def p04():
    import mpmath as mpm
    t, k = sp.symbols("t k", positive=True)
    f = lambda u: 4 * u * sp.exp(-u**2)
    xm = [r for r in sp.solve(sp.diff(f(x), x), x) if r > 0][0]
    M = sp.simplify(f(xm))
    # g(x)=t 가 서로 다른 두 실근: 0<t<M 이면 4개, t=M 이면 2개(두 극대점), t≤0 이면 x=0 근 등 — 수치로 근 개수 확인
    def nroots(tv, kv):
        g = lambda u: float(f(-kv * u)) if u < 0 else float(f(u))
        xs = [i / 200 for i in range(-4000, 4001)]
        c = 0
        for u, v in zip(xs, xs[1:]):
            gu, gv = g(u) - tv, g(v) - tv
            if abs(gu) < 1e-13 or gu * gv < 0:
                c += 1
        return c
    kv_sol = sp.solve(sp.Eq(xm + xm / k, 3 * sp.sqrt(2) / 2), k)
    # t=M: x>0 에서 f-M ≤ 0 이고 등호는 x=xm 뿐(최댓값) — 수치 격자로 확인
    fm = [float(f(sp.Rational(i, 1000)) - M) for i in range(1, 6000)]
    ok_cnt = (max(fm) <= 1e-15 and sum(1 for v in fm if v > -1e-9) <= 3 and nroots(float(M) * 0.5, 0.5) == 4
              and nroots(-0.3, 0.5) == 0 and nroots(float(M) * 1.01, 0.5) == 0)
    # 0<t<M 이면 4개, t<0 또는 t>M 이면 0개, t=0 이면 1개 → 근 2개는 t=M 뿐
    report(4, M == 2 * sp.sqrt(2) * sp.exp(-sp.Rational(1, 2)) and kv_sol == [sp.Rational(1, 2)] and ok_cnt,
           f"극댓값 t={M}, k={kv_sol}, 근 개수 확인={ok_cnt} (YAML t=2√2e^(-1/2), k=1/2)")


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
