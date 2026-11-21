#!/usr/bin/env python3
"""SJU-G25B (세종대 2025학년도 수시 논술고사 B형) — sympy/수치 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import mpmath as mp
import numpy as np
import sympy as sp

PREFIX = "SJU-G25B"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    return {k.strip(): sp.sympify(v.strip()) for k, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def nondiff_points(f):
    """다항식 f 에 대해 |f| 가 미분불가능한 점 = f 의 실근 중 f'≠0 인 점(근사)."""
    rts = sp.Poly(f, x).real_roots()
    fp = sp.diff(f, x)
    return sorted({round(float(r), 9) for r in rts if abs(float(fp.subs(x, r))) > 1e-9})


def parse_interval(s):
    s = s.replace(" ", "")
    lo, hi = s[1:-1].split(",")
    return s[0], float(sp.sympify(lo)), float(sp.sympify(hi)), s[-1]


def p01():
    p, q = sp.symbols("p q", real=True)
    f = x * (x - 1) * (x**2 + p * x + q)               # |f| 가 0,1 에서 미분불가능 ⇒ 0,1 은 근
    qs = sp.solve(sp.Eq(f.subs(x, -1), 2), q)[0]
    f = sp.expand(f.subs(q, qs))
    good = []
    for pv in [sp.Rational(k, 50) for k in range(-300, 401)]:
        if nondiff_points(f.subs(p, pv)) == [0.0, 1.0]:
            good.append(sp.diff(f, x).subs({p: pv, x: 0}))
    lb, lo, hi, rb = parse_interval(md_answer(1, "1"))
    ok1 = abs(float(min(good)) - lo) < 1e-12 and lb == "[" and abs(float(max(good)) - hi) <= 0.02 + 1e-12 and \
        rb == ")" and all(float(v) < hi for v in good)
    # (2) f'(0)=-3 일 때 e^{|f|} 의 좌·우 차분몫
    pv = sp.solve(sp.Eq(sp.diff(f, x).subs(x, 0), -3), p)[0]
    fl = sp.lambdify(x, f.subs(p, pv))
    E = lambda h: (mp.e**abs(fl(h)) - 1) / h
    rd, ld = E(1e-8), E(-1e-8)
    ok2 = abs(rd - 3) < 1e-5 and abs(ld + 3) < 1e-5
    # (3) 이중접선: f-(mx+n)=(x-α)^2(x-β)^2 를 sympy 로 풀어 m(a) 를 얻고 (0,4] 격자에서 최소
    al, be, m, n = sp.symbols("alpha beta m n")
    a = sp.symbols("a", real=True)
    fa = f.subs(p, a)
    eqs = sp.Poly(sp.expand(fa - (m * x + n) - (x - al)**2 * (x - be)**2), x).coeffs()
    sols = sp.solve(eqs, [al, be, m, n], dict=True)
    mexpr = sp.simplify(sols[0][m])
    ml = sp.lambdify(a, mexpr)
    agrid = np.linspace(1e-6, 4, 400001)
    agrid = agrid[np.abs(agrid - 1) > 1e-6]                    # a=1 이면 α=β
    amin = agrid[np.argmin(ml(agrid))]
    ok3 = abs(-amin - float(sp.sympify(md_answer(1, "3")))) < 1e-4
    report(1, ok1 and ok2 and ok3, f"f'(0)∈[{float(min(good))},{float(max(good))}] (0 제외) 우/좌={float(rd):.6f}/{float(ld):.6f} m(a)={mexpr} argmin a={amin:.6f}")


def p02():
    p, q, c, b = sp.symbols("p q c b", real=True)
    f = x**3 + p * x**2 + q * x + c
    ps = sp.solve(sp.Poly(sp.expand(f.subs(x, -x) - c + (f - c)), x).coeffs(), p, dict=True)[0]
    f = f.subs(ps)
    v1 = sp.integrate(f, (x, -1, 1))
    ok1 = sp.simplify(v1 - sp.sympify(md_answer(2, "1"), locals={"c": c})) == 0
    # (2) c=0, b 격자: f=|x| 의 서로 다른 실근 개수 직접 계산
    def roots_abs(fx):
        r = set()
        for r0 in sp.Poly(sp.expand(fx - x), x).real_roots():
            if r0 >= 0: r.add(round(float(r0), 10))
        for r0 in sp.Poly(sp.expand(fx + x), x).real_roots():
            if r0 < 0: r.add(round(float(r0), 10))
        return sorted(r)
    ks = []
    for bv in [sp.Rational(k, 100) for k in range(-300, 301)]:
        rs = roots_abs(f.subs({q: bv, c: 0}))
        if len(rs) == 2:
            ks.append(max(abs(v) for v in rs))
    lb, lo, hi, rb = parse_interval(md_answer(2, "2"))
    ok2 = lb == "(" and rb == "]" and abs(max(ks) - hi) < 1e-9 and min(ks) > lo and min(ks) < 0.11
    # (3) md f 가 (가)(나)(f(0)≠0) 와 적분 조건을 만족 + f(0)<0 접선형은 적분 < 0
    fm = sp.sympify(md_answer(2, "3"), locals={"x": x})
    rs = roots_abs(fm)
    beta = max(rs)
    I = sp.integrate(fm - x, (x, 0, beta)) + sp.integrate(fm + x, (x, -beta, 0))
    ok3 = sp.simplify(fm.subs(x, -x) - fm.subs(x, 0) + (fm - fm.subs(x, 0))) == 0 and len(rs) == 2 and fm.subs(x, 0) != 0 and abs(float(I)) < 1e-12
    for gv in (-0.2, -0.5, -1.0, -2.0):                          # f(0)<0: x=γ<0 에서 y=-x 에 접함
        gg = sp.nsimplify(gv)
        fg = x**3 + (-1 - 3 * gg**2) * x + 2 * gg**3
        rr = roots_abs(fg)
        if len(rr) == 2:
            bb = max(rr)
            Ig = sp.integrate(fg - x, (x, 0, bb)) + sp.integrate(fg + x, (x, -bb, 0))
            ok3 &= float(Ig) < 0
    report(2, ok1 and ok2 and ok3, f"∫f={v1} k∈({min(ks):.3f}..{max(ks):.6f}] f={fm} roots={rs}")


def p03():
    Cs = sp.symbols("C", positive=True)
    fams = [lambda t: mp.e**(-t**2), lambda t: 1 / (1 + t**2), lambda t: mp.cosh(t / 2)]
    ok1 = True
    for f in fams:
        C = mp.quad(f, [-1, 1]); F = lambda v: mp.quad(f, [-1, v])
        for v in (-0.7, 0.2, 1.5, 3.0):
            ok1 &= abs(F(v) + F(-v) - float(sp.sympify(md_answer(3, "1"), locals={"C": Cs}).subs(Cs, C))) < 1e-12
    # (2) g'(x)>=0 를 (-1,1) 격자에서 직접 확인하며 a 이분 탐색
    f = lambda t: np.exp(-t**2)
    C = float(mp.quad(lambda t: mp.e**(-t**2), [-1, 1]))
    import math
    xs = np.concatenate([np.linspace(-1, 1, 200001)[1:-1], 1 - np.logspace(-9, -5, 50)])
    Fx = np.array([math.sqrt(math.pi) / 2 * (math.erf(v) + math.erf(1)) for v in xs])   # ∫_{-1}^x e^{-t^2}dt
    def incr(a):
        gp = -2 * a**2 * xs * np.exp(-a**2 * xs**2) * Fx + np.exp(-(a**2 + 1) * xs**2)
        return np.all(gp >= -1e-13)
    lo, hi = 0.1, 2.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if incr(mid) else (lo, mid)
    ok2 = abs(lo - float(sp.sympify(md_answer(3, "2"), locals={"C": Cs}).subs(Cs, C))) < 1e-6
    # (3) 여러 우함수에서 F(a)=2C/3 인 a 를 찾아 a∫g 직접 계산
    ok3 = True; vals = []
    for f in fams:
        C = mp.quad(f, [-1, 1]); F = lambda v: mp.quad(f, [-1, v])
        a = mp.findroot(lambda v: F(v) - 2 * C / 3, 0.2)
        val = a * mp.quad(lambda v: f(a * v) * F(v), [-1, 1])
        vals.append(float(val))
        ok3 &= a > 0 and abs(val - float(sp.sympify(md_answer(3, "3"), locals={"C": Cs}).subs(Cs, C))) < 1e-10
    report(3, ok1 and ok2 and ok3, f"a_max~{lo:.6f} a∫g={vals}")


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
