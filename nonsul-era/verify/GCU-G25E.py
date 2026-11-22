#!/usr/bin/env python3
"""가천대 2025학년도 논술고사 자연계열 자연E(보고서 문항카드 73~81, 문제 7~15) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산 후 YAML answer 와 대조."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "GCU-G25E"
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




def tex3(s: str):
    s = re.sub(r"\\sqrt\[3\]\{([^{}]*)\}", r"cbrt(\1)", s)
    s = re.sub(r"(\d)\s*cbrt", r"\1*cbrt", s)
    s = s.replace("\\frac{", "((").replace("}{", ")/(").replace("}", "))")
    return sp.sympify(s, locals={"cbrt": sp.cbrt})


def p07():
    r = sp.symbols("r", real=True)
    S = lambda n: (r ** n - 1) / (r - 1)   # a1=1 (비는 a1 무관)
    eq = sp.numer(sp.together((S(4) - S(1)) / (S(10) - S(4)) - sp.Rational(64, 9)))
    sols = [v for v in sp.Poly(eq, r).real_roots() if v < 0 and v != -1 and sp.simplify((S(10) - S(4)).subs(r, v)) != 0]
    sols = sorted({sp.nsimplify(sp.N(v, 50), [sp.cbrt(9)]) for v in sols}, key=float)
    report(7, len(sols) == 1 and abs(sp.N(sols[0] - tex3(ans(7)), 30)) < 1e-25, f"r={sols}")


def p08():
    A, B, C = sp.symbols("A B C")
    f = A * x ** 2 + B * x + C
    lim = sp.limit((sp.sqrt(x ** 4 - 5) - f) / (6 * x ** 2 + 1), x, sp.oo)
    sA = sp.solve(sp.Eq(lim, sp.Rational(1, 3)), A)[0]
    f = f.subs(A, sA)
    gl = f.subs(x, 1 - x) - f
    eqs = [sp.Eq(gl.subs(x, 1), f.subs(x, 1)), sp.Eq(sp.diff(gl, x).subs(x, 1), sp.diff(f, x).subs(x, 1))]
    s = sp.solve(eqs, [B, C], dict=True)[0]
    v = f.subs(s).subs(x, 3)
    report(8, sp.simplify(v - tex(ans(8))) == 0, f"A={sA}, {s}, f(3)={v}")


def p09():
    tot = 0
    for kk in range(1, 1000):
        sol = sp.solve(sp.Eq(3 * x + 3, 2 * x + sp.log(kk, 2)), x)[0]   # 5^u 단사
        if 1 <= sol <= 3:
            tot += kk
    report(9, tot == tex(ans(9)), f"합={tot}")


def p10():
    s = sp.symbols("s", real=True)
    # (가): x^2+9>=9 이므로 f(x^2+9)=0 의 실근이 x=0 하나뿐 <=> f(9)=0 이고 다른 근 s<=9 (s=9 중근 포함)
    f = (x - 9) * (x - s)
    num, den = f.subs(x, x - 2), f.subs(x, x ** 2)
    assert den.subs(x, -3) == 0
    cand = sp.solve(sp.Eq(num.subs(x, -3), 0), s)          # 극한이 유한하려면 분자도 0
    cand = [v for v in cand if v <= 9]
    out = []
    for v in cand:
        fv = f.subs(s, v)
        zeros = sp.solve(sp.Eq(fv.subs(x, x ** 2 + 9), 0), x)
        zr = sorted({z for z in zeros if z.is_real})
        bval = sp.limit(num.subs(s, v) / den.subs(s, v), x, -3)
        if len(zr) == 1 and bval.is_finite:
            out.append(zr[0] + bval + fv.subs(x, 3))
    report(10, len(out) == 1 and sp.simplify(out[0] - tex(ans(10))) == 0, f"s 후보={cand}, 값={out}")


def p11():
    p = sp.symbols("p", positive=True)
    O = sp.Matrix([0, 0]); A = sp.Matrix([-3, 0]); B = sp.Matrix([3, 0])
    P = 3 * sp.Matrix([sp.cos(p), sp.sin(p)])
    C = 3 * sp.Matrix([sp.cos(2 * p), sp.sin(2 * p)])        # 호 PB=호 PC
    u, w = sp.symbols("u w")
    sD = sp.solve(list(u * P - (B + w * (C - B))), [u, w], dict=True)[0]
    D = u * P
    D = D.subs(sD)
    # ACPD 평행사변형: A+P = C+D
    eq = sp.simplify((A + P - C - D)[0])
    sols = [v for v in sp.solve(eq, p) if v.is_real and 0 < v < sp.pi / 2]
    res = []
    for v in sols:
        Pv, Cv, Dv = P.subs(p, v), C.subs(p, v), D.subs(p, v)
        assert sp.simplify((A + Pv - Cv - Dv).norm()) == 0
        AP = sp.sqrt(sp.simplify((Pv - A).dot(Pv - A)))
        u1, u2 = Pv - B, Cv - B
        cosang = sp.simplify(u1.dot(u2) / (u1.norm() * u2.norm()))
        res.append((sp.nsimplify(sp.simplify(AP)), sp.nsimplify(sp.simplify(sp.sqrt(1 - cosang ** 2)))))
    a_ = ans(11)
    exp = (tex(re.search(r"=\s*([^,]+),", a_).group(1)), tex(a_.split("=")[-1]))
    report(11, len(res) == 1 and all(sp.simplify(res[0][i] - exp[i]) == 0 for i in range(2)), f"p={sols}, (AP, sin)={res}")


def p12():
    cc = sp.symbols("cc")
    f = x ** 3 - 3 * x ** 2 + 2 * x * cc + (cc / sp.sqrt(2)) ** 2
    cs = sp.solve(sp.Eq(sp.integrate(f, (x, 0, 2)), cc), cc)
    good = []
    for v in cs:
        h = sp.expand(f.subs(cc, v) + 3 * x ** 2)            # (나) <=> h 증가함수
        crit = [z for z in sp.solve(sp.diff(h, x), x) if z.is_real]
        dh = sp.diff(h, x)
        inc = all(dh.subs(x, z + sp.Rational(1, 10)) > 0 and dh.subs(x, z - sp.Rational(1, 10)) > 0 for z in crit) and dh.subs(x, 0) >= 0 if crit else True
        if inc:
            good.append(f.subs(cc, v))
    fv = good[0]
    A, B = sp.symbols("A B")
    g0 = fv + A * x ** 2                                    # [0,1)
    e1 = sp.Eq(g0.subs(x, 1), g0.subs(x, 0) + B)            # x=1 연속
    I23 = sp.integrate(g0, (x, 0, 1)) + 2 * B               # g(x+2)=g(x)+2b
    s = sp.solve([e1, sp.Eq(I23, sp.Rational(79, 12))], [A, B], dict=True)[0]
    v = tex(ans(12).split("=")[-1])
    report(12, len(good) == 1 and s[A] == v and s[B] == v, f"c 후보={cs}, f={fv}, {s}")


def _sets(bb, D):
    f2 = sp.integrate(-4 * t ** 3 - 12 * t ** 2 - 9 * t - 2, (t, 0, x)) + D
    fb = f2.subs(x, bb)
    f1 = fb - 2 * (x - bb)
    fp2 = -4 * x ** 3 - 12 * x ** 2 - 9 * x - 2
    Bs = {z for z in sp.Poly(fp2, x).real_roots() if z <= bb}
    Zs = {z for z in sp.Poly(f2, x).real_roots() if z <= bb} | {z for z in sp.solve(f1, x) if z > bb}
    Zs = {sp.nsimplify(z) if not isinstance(z, sp.CRootOf) else z for z in Zs}
    As = Bs | Zs
    F = lambda z: f2.subs(x, z) if z <= bb else f1.subs(x, z)
    Cs = {sp.nsimplify(sp.Abs(sp.N(F(z), 40)), tolerance=1e-30, rational=True) for z in As}
    return As, Bs, Cs


def p13():
    bs = [v for v in sp.solve(sp.Eq(-4 * b ** 3 - 12 * b ** 2 - 9 * b - 2, -2), b)]   # f' 연속
    found = []
    for bb in bs:
        h = sp.integrate(-4 * t ** 3 - 12 * t ** 2 - 9 * t - 2, (t, 0, x))
        crit = sorted({-h.subs(x, z) for z in (-2, sp.Rational(-1, 2)) if z <= bb} | {-h.subs(x, bb)})
        cand = set(crit) | {crit[0] - 1, crit[-1] + 1} | {(crit[i] + crit[i + 1]) / 2 for i in range(len(crit) - 1)}
        for D in sorted(cand):
            As, Bs, Cs = _sets(bb, D)
            if len(As) == 3 and len(Bs) == 2 and len(Cs) == 2:
                found.append((bb, D, Cs))
    exp = {tex(v.strip()) for v in re.search(r"\\left\\\{(.*)\\right\\\}", ans(13)).group(1).replace("\\ ", "").split(",")}
    report(13, len(found) == 1 and found[0][2] == exp, f"b 후보={bs}, (b,D,C)={found}")


def p14():
    v1 = -t ** 2 / 2 + sp.Rational(5, 2) * t
    C0 = sp.symbols("C0")
    v2 = sp.integrate(2 * t - sp.Rational(1, 2), t) + C0
    kk = [v for v in sp.solve(sp.Eq((2 * t - sp.Rational(1, 2)), 7 * sp.diff(v1, t)), t) if v >= 0]
    out = []
    for kv in kk:
        c0 = sp.solve(sp.Eq(v1.subs(t, kv), v2.subs(t, kv)), C0)[0]
        w2 = v2.subs(C0, c0)
        def dist(v):
            pts = sorted({sp.S(0), kv} | {z for z in sp.solve(v, t) if z.is_real and 0 < z < kv})
            return sum(sp.Abs(sp.integrate(v, (t, pts[i], pts[i + 1]))) for i in range(len(pts) - 1))
        d1, d2 = dist(v1), dist(w2)
        out.append(sp.Abs(sp.simplify(d1 - d2)))
    report(14, len(out) == 1 and sp.simplify(out[0] - tex(ans(14))) == 0, f"k={kk}, 차={out}")


def p15():
    A = sp.symbols("A", real=True)
    th = sp.symbols("th", real=True)
    # sinθ+cosθ=√5/3 를 만족하는 θ∈(π/2,π)
    sols = [v for v in sp.solve(sp.Eq(sp.sin(th) + sp.cos(th), sp.sqrt(5) / 3), th)]
    sols = [v for v in [sp.nsimplify(s0) for s0 in sols] + [s0 + 2 * sp.pi for s0 in sols] if sp.pi / 2 < sp.N(v) < sp.pi]
    v = sols[0]
    av = sp.nsimplify(sp.N(3 * sp.sin(v) * sp.cos(v), 50), rational=True)
    chk = sp.N(3 * sp.sin(v) ** 2 - sp.sqrt(5) * sp.sin(v) + av, 30)
    bv = sp.N(sp.sin(v) - sp.cos(v), 50)
    prod = sp.N(av * bv, 40)
    report(15, len(sols) == 1 and abs(chk) < 1e-20 and abs(prod - sp.N(tex(re.sub(r"\\sqrt\{(\d+)\}", r"\\sqrt\1", ans(15))), 40)) < 1e-30, f"θ={sp.N(v)}, a={av}, ab≈{prod}")

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
