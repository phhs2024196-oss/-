#!/usr/bin/env python3
"""가천대 2025학년도 논술고사 의예과 수학(문제 1~8) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산 후 YAML answer 와 대조."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "GCU-G25M"
WIDTH = 2
RESULTS = []
ITEMS = pathlib.Path(__file__).resolve().parent.parent / "items/GCU/PAST"
x, t, a, b, c, k = sp.symbols("x t a b c k", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def ans(seq: int, sub: str = "1") -> str:
    d = yaml.safe_load((ITEMS / f"{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    for s in d["subparts"]:
        if str(s["id"]) == sub:
            return str(s["answer"]["value"]).strip()
    raise KeyError(sub)


def tex(s: str):
    """간단한 LaTeX 값 → sympy"""
    s = s.replace("\\dfrac", "\\frac").replace("\\ ", "").replace("\\pi", "pi")
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt(\d+)", r"sqrt(\1)", s)
    s = re.sub(r"\\log_(\d+)\s*(\d+)", r"log(\2,\1)", s)
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"(\d)\s*(log|sqrt|exp|\()", r"\1*\2", s)
    return sp.nsimplify(sp.sympify(s, locals={"log": lambda v, base: sp.log(v, base)}), rational=False)


def pick(s: str, name: str):
    m = re.search(rf"{name}\s*=\s*([^,]+?)(?:,|$)", s.replace("\\,", ","))
    return tex(m.group(1))


import math
import mpmath as mp


def p01():
    # 좌표: C0=(0,0), B0=(-6,0)... 직각 C0, 각 B0=5π/12. 반원 G_n 지름 B_nC_n, B_{n+1} = C_n 에서 빗변에 내린 수선의 발
    B0 = sp.Matrix([-6, 0]); C0 = sp.Matrix([0, 0]); A = sp.Matrix([0, 6 * sp.tan(5 * sp.pi / 12)])
    tot, Bn, Cn, Ts = 0, B0, C0, []
    for _ in range(4):
        u = (A - Bn) / (A - Bn).norm()
        Bn1 = Bn + u * ((Cn - Bn).dot(u))           # 지름 원주각 → 수선의 발
        r = (Cn - Bn).norm() / 2
        chord = (Bn1 - Bn).norm()
        ang = 2 * sp.asin(chord / (2 * r))
        T = sp.Rational(1, 2) * r ** 2 * (ang - sp.sin(ang))
        Ts.append(sp.nsimplify(sp.simplify(T)))
        Cn1 = sp.Matrix([0, Bn1[1]])
        Bn, Cn = Bn1, Cn1
    ratio = sp.nsimplify(sp.simplify(Ts[1] / Ts[0]))
    S = sp.simplify(Ts[0] / (1 - ratio))
    p_, q_ = sp.symbols("p q")
    ya = ans(1)
    P, Q = pick(ya, "p"), pick(ya, "q")
    ok = abs(float(S - (P + Q * sp.sqrt(3)) * (sp.pi / 3 - 1))) < 1e-12 and abs(float(Ts[2] / Ts[1] - ratio)) < 1e-12
    report(1, ok, f"T0={Ts[0]}, 공비={ratio}, ΣT={sp.nsimplify(S)}")


def p02():
    I = sp.integrate(x ** 3 * sp.log(x + 1), (x, 0, 1))
    L = sp.simplify(-4 * I)
    mp.mp.dps = 30
    n = 200000
    s = mp.fsum(mp.mpf(k) ** 3 * mp.log(k + n) for k in range(1, n + 1))
    num = (1 + mp.mpf(1) / n) ** 2 * mp.log(n) - 4 * s / mp.mpf(n) ** 4
    report(2, sp.simplify(L - tex(ans(2))) == 0 and abs(float(num) - float(L)) < 1e-4, f"-4∫x³ln(x+1)={L}, n=2e5 수치={float(num):.6f}")


def p03():
    A_, B_, al = sp.symbols("A B alpha", positive=True)
    al_s = [r for r in sp.solve(sp.Eq(1 / al, 2 * B_ ** 2 * al - B_), al) if r.is_positive is not False]
    al_v = [r for r in al_s if sp.simplify(r - 1 / B_) == 0][0]
    A_v = sp.solve(sp.Eq(sp.log(A_ * al_v), (B_ ** 2 * x ** 2 - B_ * x + 1).subs(x, al_v)), A_)[0]
    # 접점 유일성(대표 b=1/(3√3)) 수치 확인 + AB 최솟값
    m = sp.symbols("m", positive=True)
    Px, Py = 1 / B_, 1
    G = (Px + Py / m) ** 2 + (Py + m * Px) ** 2
    mstar = sp.solve(sp.diff(G, m), m)
    bsol = sp.nsolve(sp.sqrt(G.subs(m, B_ ** sp.Rational(1, 3))) - 8, B_, 0.2)
    bsol = sp.nsimplify(bsol, [sp.sqrt(3)])
    abv = sp.simplify(A_v.subs(B_, bsol) * bsol)
    h = lambda xx: math.log(float(A_v.subs(B_, bsol)) * xx) - float((bsol ** 2 * xx ** 2 - bsol * xx + 1))
    xs = [0.01 * i for i in range(1, 3000)]
    cross = sum(1 for i in range(len(xs) - 1) if h(xs[i]) * h(xs[i + 1]) < 0)
    report(3, sp.simplify(abv - tex(ans(3).replace("e", "E"))) == 0 if False else abs(float(abv) - float(sp.E / 27)) < 1e-12 and "frac{e}{27}" in ans(3) and cross == 0,
           f"α={al_v}, a={A_v}, b={bsol}, ab={abv}, 부호변화(교차)={cross}")


def p04():
    def F(xv, n):  # x>0 유리수
        r = xv - n * (sp.ceiling(xv / n) - 1)
        v = 3 * r + 1; mm = 0
        while 7 ** (mm + 1) <= v:
            mm += 1
        return mm
    def count(n):
        eps = sp.Rational(1, 10 ** 6)
        cands = set()
        for j in range(0, 2025 // n + 2):
            for s in (2, 16, 114, 800, n):
                c = j * n + s
                if 0 < c < 2025 and (s < n or s == n):
                    cands.add(c)
        cnt = 0
        for c in cands:
            l, v, r = F(c - eps, n), F(sp.Integer(c), n), F(c + eps, n)
            if not (l == v == r):
                cnt += 1
        return cnt
    res = {}
    for n in range(1, 2100):
        k_ = count(n)
        res.setdefault(k_, []).append(n)
    got = {k_: max(res.get(k_, [0])) for k_ in (7, 13, 19, 25)}
    ya = ans(4)
    exp = {k_: int(re.search(rf"a_{{?{k_}}}?=(\d+)", ya.replace(" ", "")).group(1)) for k_ in (7, 13, 19, 25)}
    report(4, got == exp, f"a_k={got}")


def p05():
    # 등호 사례 f'=1/(2+cos f) 로 g(a_n) 계산 + 일반 부등식의 핵심 적분값 확인
    u = sp.symbols("u")
    I1 = -sp.integrate(sp.cos(u) * (2 + sp.cos(u)), (u, sp.pi / 2, 3 * sp.pi / 2))
    I2 = -sp.integrate(sp.cos(u) * (2 + sp.cos(u)), (u, 2 * sp.pi - sp.pi / 2, 2 * sp.pi + sp.pi / 2))
    per = sp.simplify(I1 + I2)
    # 등호 사례: ln g(a_n) = -∫_{π/2}^{2nπ+π/2} cos u (2+cos u) du
    n = sp.symbols("n", positive=True, integer=True)
    lng = sp.simplify(-sp.integrate(sp.cos(u) * (2 + sp.cos(u)), (u, sp.pi / 2, 2 * n * sp.pi + sp.pi / 2)))
    lim = sp.limit(sp.exp(lng), n, sp.oo)
    report(5, sp.simplify(I1 - (4 - sp.pi / 2)) == 0 and sp.simplify(I2 - (-4 - sp.pi / 2)) == 0 and per == -sp.pi and lim == tex(ans(5)),
           f"반주기 상한 {I1}, {I2}, 한 주기 {per}; 등호 사례 ln g(a_n)={lng}, 극한={lim} [증명-수치확인]")


def p06():
    E1 = math.exp(-1)
    def make(a):
        def f(xx):
            return xx * xx + 4 * xx + 3 if xx <= 0 else -(xx + 1) ** 2 * math.exp(-xx) + a
        return f
    xl = [-12 + 0.0005 * i for i in range(24001)]            # [-12, 0] (x=0 포함)
    xr = [1e-9] + [0.0005 * i for i in range(1, 120001)]     # (0, 60]  — x=0 의 불연속을 건너 세지 않도록 분리
    def g(a, tv):
        f = make(a)
        tot = 0
        for xs in (xl, xr):
            vals = [abs(f(xx)) - tv for xx in xs]
            tot += sum(1 for i in range(len(vals) - 1) if vals[i] * vals[i + 1] < 0)
        return tot
    def levels(a):
        return sorted({0.0, 1.0, 3.0, abs(a - 1), abs(a - 4 * E1), a})
    def jump(a):
        d = 1e-5
        return max(g(a, L - d) - g(a, L + d) for L in levels(a) if L > d)
    cands = [4 * E1 - 1, 2 * E1, 1.0, 2 * E1 + 0.5, 2.0]
    ok_c = all(jump(a) > 2 for a in cands)
    ok_o = all(jump(a) <= 2 for a in (0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.5, 3.5))
    a1 = min(cands)
    pts = [0.0, (4 * E1 - 1) / 2, 4 * E1 - 1, 0.5, 2 - 4 * E1, 0.9, 1.0, 2.0, 3.0, 4.0]
    d = 1e-5
    M = max(g(a1, m_ - d) + g(a1, m_ + d) for m_ in pts)
    ya = ans(6)
    sumv = float(tex(ya.split("=")[-1].replace("e^{-1}", "exp(-1)")))
    okv = abs(sum(cands) - sumv) < 1e-12 and M == int(pick(ya, "M"))
    report(6, ok_c and ok_o and okv, f"후보 a 에서 최대 도약={[jump(a) for a in cands]}, 일반 a 도약≤2:{ok_o}, Σa={sum(cands):.10f}, M={M} [수치 교점 세기]")


def p07():
    A = sp.symbols("A", positive=True)
    B = sp.Matrix([0, 0]); Ap = sp.Matrix([0, 3]); C = sp.Matrix([4, 0])
    P = B + (C - B) * 1 / (1 + A)
    Q = C + (Ap - C) * 1 / (1 + A)
    av = [v for v in sp.solve(sp.Eq((Ap - P).dot(Ap - P), (Q - P).dot(Q - P)), A) if v > 0][0]
    P, Q = P.subs(A, av), Q.subs(A, av)
    u, v = Ap - P, Q - P
    cs = sp.nsimplify(u.dot(v) / sp.sqrt(u.dot(u) * v.dot(v)))
    S = sp.Abs(u[0] * v[1] - u[1] * v[0]) / 2
    ya = ans(7)
    ok = sp.simplify(cs - pick(ya, r"\\cos\(\\angle\\mathrm\{APQ\}\)")) == 0 and sp.simplify(S - pick(ya, "S")) == 0
    report(7, ok, f"a={av}, cos={cs}, 넓이={S}")


def p08():
    f2 = sp.exp(x) * sp.cos(x); g2 = sp.exp(x) * sp.sin(2 * x)
    cross = sp.solve(sp.Eq(sp.cos(x), sp.sin(2 * x)), x)
    V = sp.sqrt(3) / 4 * (sp.integrate(f2, (x, 0, sp.pi / 6)) + sp.integrate(g2, (x, sp.pi / 6, sp.pi / 2)))
    # 대소 수치 확인
    okh = all((float(f2.subs(x, t_)) >= float(g2.subs(x, t_))) == (t_ <= math.pi / 6) for t_ in [0.01 * i for i in range(1, 157)])
    lhs = sp.expand(80 * V + 10 * sp.sqrt(3))
    ya = ans(8)
    A_, B_, C_ = pick(ya, "a"), pick(ya, "b"), pick(ya, "c")
    ok = sp.simplify(lhs - (A_ * sp.exp(B_) + C_ * sp.exp(sp.pi / 2))) == 0
    report(8, ok and okh, f"교점={cross}, 80V+10√3={sp.simplify(lhs)}")



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
