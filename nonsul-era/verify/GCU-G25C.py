#!/usr/bin/env python3
"""가천대 2025학년도 논술고사 자연C 세트 수학(문제 7~15) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산 후 YAML answer 와 대조."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "GCU-G25C"
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
    s = s.replace("\\dfrac", "\\frac").replace("\\ ", "")
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt(\d+)", r"sqrt(\1)", s)
    s = re.sub(r"\\log_(\d+)\s*(\d+)", r"log(\2,\1)", s)
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"(\d)\s*(log|sqrt)", r"\1*\2", s)
    return sp.nsimplify(sp.sympify(s, locals={"log": lambda v, base: sp.log(v, base)}), rational=False)


def pick(s: str, name: str):
    m = re.search(rf"{name}\s*=\s*([^,]+?)(?:,|$)", s.replace("\\,", ","))
    return tex(m.group(1))


def p07():
    th = sp.symbols("th")
    # cos(pi/2+th)=1/sqrt6 → sin th=-1/sqrt6, cos th<0
    sols = [s for s in sp.solve(sp.Eq(sp.cos(sp.pi / 2 + th), 1 / sp.sqrt(6)), th)]
    cand = []
    for s in sols:
        for kk in range(-2, 3):
            v = sp.nsimplify(s + 2 * sp.pi * kk)
            if 0 <= float(v) < 2 * float(sp.pi) and float(sp.cos(v)) < 0:
                cand.append(v)
    v = cand[0]
    val = sp.nsimplify(sp.simplify(sp.tan(sp.pi + v) ** 2 + sp.sqrt(5) * sp.tan(v) + 3 * sp.sin(sp.Rational(3, 2) * sp.pi + v) ** 2))
    report(7, len(cand) == 1 and sp.simplify(val - tex(ans(7))) == 0, f"θ={v}, 값={val}")


def p08():
    def f(v):
        v = sp.nsimplify(v)
        return sp.Rational(1, 3) * v ** -4 if v <= 2 else sp.log(v, 3)
    tot = sum(f(f(f(n))) for n in (1, 2, 3))
    tot = sp.expand_log(sp.logcombine(tot), force=True)
    report(8, abs(float(tot) - float(tex(ans(8)))) < 1e-12, f"합={sp.nsimplify(tot)} ≈ {float(tot):.10f}")


def p09():
    # 차수 비교: f 일차(f(x)=b x + c)로 놓고 항등식 풀이 (상수·2차·3차 이상은 아래에서 차수 비교로 배제 확인)
    ok_deg = True
    for n in range(0, 5):
        cs = sp.symbols(f"c0:{n+1}")
        f = sum(cs[i] * x ** i for i in range(n + 1))
        eq = sp.expand(sp.integrate(t * f.subs(x, t), (t, 0, x)) - (f + a * x ** 3 + x ** 2 + x - 2))
        sol = sp.solve(sp.Poly(eq, x).coeffs(), list(cs) + [a], dict=True)
        sol = [s for s in sol if n == 0 or s.get(cs[n], cs[n]) != 0]
        if n == 1:
            s1 = sol
        elif sol:
            ok_deg = False
    s = s1[0]
    cs = sp.symbols("c0:2")
    f = cs[0] + cs[1] * x
    fa = f.subs(s).subs(x, s[a])
    report(9, ok_deg and len(s1) == 1 and sp.simplify(fa - tex(ans(9))) == 0, f"f(x)={f.subs(s)}, a={s[a]}, f(a)={fa}")


def p10():
    def F(kk):
        return sum(1 for n in range(2, 7 * kk + 1) if (7 * kk) % n == 0 and (7 * kk) // n >= 3)
    g = F(7) == 2
    n_ = sum(1 for kk in range(1, 6) if F(kk) == 2) == 2
    d = sum(kk for kk in range(1, 11) if F(kk) == 5) == 24
    picked = [lab for lab, ok in zip("ㄱㄴㄷ", (g, n_, d)) if ok]
    yaml_pick = re.findall(r"[ㄱㄴㄷ]", ans(10))
    report(10, picked == yaml_pick, f"f(1..10)={[F(kk) for kk in range(1, 11)]}, 참={picked}")


def p11():
    A_, B_, C_ = sp.symbols("A B C")
    f = A_ * x ** 2 + B_ * x + C_
    e1 = sp.Eq(sp.limit(f / (9 * x ** 2 - 1), x, sp.oo), sp.Rational(1, 3))
    sA = sp.solve(e1, A_)[0]
    f = f.subs(A_, sA)
    # (나): f(1/3)=0, 극한=1/4
    sC = sp.solve(f.subs(x, sp.Rational(1, 3)), C_)[0]
    f = f.subs(C_, sC)
    sB = sp.solve(sp.Eq(sp.limit((9 * x ** 2 - 1) / f, x, sp.Rational(1, 3)), sp.Rational(1, 4)), B_)
    f = sp.factor(f.subs(B_, sB[0]))
    expr = (x - sp.Rational(1, 3)) * (x ** 2 + a * x + 5) / f
    L = sp.limit(-expr, x, sp.Rational(1, 3), "-")
    R = sp.limit(expr, x, sp.Rational(1, 3), "+")
    sa = sp.solve(sp.Eq(L, R), a)[0]
    bb = R.subs(a, sa)
    ya = ans(11)
    report(11, sp.simplify(sa - pick(ya, "a")) == 0 and sp.simplify(bb - pick(ya, "b")) == 0, f"f={f}, a={sa}, b={bb}")


def p12():
    import numpy as np
    vals = set()
    for sgn in (1, -1):
        # f(x)=sgn*(x^3-3x)+d, α=-1, β=1, |f(β)|=2|f(α)|, f(α)f(β)<0
        d = sp.symbols("d", real=True)
        f = sgn * (x ** 3 - 3 * x) + d
        for dv in sp.solve(sp.Eq(f.subs(x, 1) ** 2, 4 * f.subs(x, -1) ** 2), d):
            fv = f.subs(d, dv)
            if fv.subs(x, -1) * fv.subs(x, 1) >= 0:
                continue
            fa = abs(fv.subs(x, -1))
            fv = sp.expand(fv)
            levels = [sp.Integer(0), sp.nsimplify(abs(fv.subs(x, -1))), sp.nsimplify(abs(fv.subs(x, 1)))]
            ts, tot = [], 0
            for L in levels:
                for side in (L, -L) if L != 0 else (0,):
                    for r in sp.Poly(fv - side, x).real_roots():
                        ts.append(r)
            ts = sorted(set(ts))
            # 홀수 개 판정(각 t 에 대해)
            def cnt(tv):
                lv = sp.nsimplify(abs(fv.subs(x, tv)))
                rs = set(sp.Poly(fv - lv, x).real_roots()) | set(sp.Poly(fv + lv, x).real_roots())
                return len(rs)
            good = [tv for tv in ts if cnt(tv) % 2 == 1]
            # 일반 높이(무작위)에서는 짝수임을 표본 확인
            for L in np.linspace(0.05, float(max(levels)) + 2, 37):
                rs = set(np.round(np.roots([float(c) for c in sp.Poly(fv - L, x).all_coeffs()]), 9)) | set(np.round(np.roots([float(c) for c in sp.Poly(fv + L, x).all_coeffs()]), 9))
                rs = {r.real for r in rs if abs(r.imag) < 1e-9}
                if any(abs(L - float(v)) < 1e-6 for v in levels):
                    continue
                assert len(rs) % 2 == 0
            m = len(good)
            vals.add((m, sp.nsimplify(sum(fv.subs(x, tv) for tv in good) / fa)))
    got = {v for _, v in vals}
    ms = {m_ for m_, _ in vals}
    yv = {tex(s) for s in re.findall(r"-?\d+", ans(12))}
    report(12, ms == {11} and got == yv, f"(m, 값)={vals}")


def p13():
    f0 = a * (x - sp.Rational(1, 4)) ** 2 - sp.Rational(9, 16)
    sa = sp.solve(sp.Eq(sp.limit(f0, x, 1, "-"), f0.subs(x, 0) + 1), a)[0]
    f0 = f0.subs(a, sa)
    p = [sp.integrate(f0, (x, 0, 1)) + (n - 1) for n in range(1, 101)]  # f(x+n)=f(x)+n
    y = sp.symbols("y")
    g = sum((y - pn) ** 2 for pn in p)
    al = sp.solve(sp.diff(g, y), y)[0]
    report(13, sp.simplify(al - tex(ans(13))) == 0, f"a={sa}, p1={p[0]}, α={al}")


def p14():
    A_, K = sp.symbols("A K")
    # ①: f(2)=f(-2)≠0 이면 (나) 극한 = 1 ≠ 3
    gen = sp.symbols("g0:4")
    f = sum(gen[i] * x ** i for i in range(4))
    cond = sp.solve(sp.Eq(f.subs(x, 2), f.subs(x, -2)), gen[1])
    f2 = f.subs(gen[1], cond[0])
    lim1 = sp.simplify(f2.subs(x, 2) / f2.subs(x, -2))
    ok1 = lim1 == 1 and "존재하지 않" in ans(14, "1")
    f = A_ * (x + 2) * (x - 2) * (x - K)
    sk = sp.solve(sp.Eq(sp.limit(f.subs(x, x + 2) / f.subs(x, x - 2), x, 0), 3), K)[0]
    f = f.subs(K, sk)
    sa = sp.solve(sp.Eq(sp.limit((f - f.subs(x, -2)) / (x - 2), x, 2), 1), A_)[0]
    f = f.subs(A_, sa)
    I = sp.integrate(f, (x, 0, 3))
    report(14, ok1 and sp.simplify(I - tex(ans(14, "2"))) == 0, f"① f(2)/f(-2)={lim1}; f={sp.factor(f)}, ∫={I}")


def p15():
    ns = [n for n in range(1, 61) if sp.simplify(sp.sin(sp.pi / n * (x + 30)) - sp.sin(sp.pi / n * x)) == 0]
    m = sum(ns)
    h = sp.symbols("h", positive=True)
    R = 4
    OD, OE, OF = [sp.Rational(12) * h, m * h, 6 * (3 * sp.sqrt(5) - 1) * h]
    sinA2 = (R ** 2 - OE ** 2) / R ** 2   # OE ⟂ BC, BC=2R sinA
    sinC2 = (R ** 2 - OD ** 2) / R ** 2
    sinB2 = (R ** 2 - OF ** 2) / R ** 2
    hv = [v for v in sp.solve(sp.Eq(5 * sinA2, 4 * sinC2), h) if v > 0][0]
    oe = sp.nsimplify(OE.subs(h, hv))
    sb = sp.nsimplify(sp.simplify(sinB2.subs(h, hv)))
    # 각의 합 = π 확인(예각삼각형)
    A_ = sp.acos(OE.subs(h, hv) / R); B_ = sp.acos(OF.subs(h, hv) / R); C_ = sp.acos(OD.subs(h, hv) / R)
    tri = abs(float(A_ + B_ + C_ - sp.pi)) < 1e-12
    ya = ans(15)
    ok = sp.simplify(oe - pick(ya, r"\\overline\{\\mathrm\{OE\}\}")) == 0 and sp.simplify(sb - pick(ya, r"\\sin\^2B")) == 0
    report(15, ok and tri, f"n={ns}, m={m}, OE={oe}, sin²B={sb}, A+B+C=π:{tri}")


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
