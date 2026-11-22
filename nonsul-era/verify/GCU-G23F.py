#!/usr/bin/env python3
"""가천대 2023학년도 논술전형 자연F 수학(문제 7~15) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산하고 YAML answer 와 대조."""
import pathlib
import re
import sys

import sympy as sp
import yaml

PREFIX = "GCU-G23F"
WIDTH = 2
RESULTS = []
ITEMS = pathlib.Path(__file__).resolve().parent.parent / "items" / "GCU" / "PAST"
x, t, k, a, b, c, n, th = sp.symbols("x t k a b c n theta", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def yaml_answer(seq: int) -> str:
    it = yaml.safe_load((ITEMS / f"{PREFIX}-{seq:02d}.yaml").read_text(encoding="utf-8"))
    return str(it["subparts"][0]["answer"]["value"])


def tex(s: str):
    """간단한 LaTeX 수(분수·근호) → sympy."""
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\ ", "")
    for _ in range(5):
        s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"((\1)/(\2))", s)
        s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"(\d)\(", r"\1*(", s)
    s = re.sub(r"(\d)sqrt", r"\1*sqrt", s)
    s = re.sub(r"\)\(", ")*(", s)
    s = re.sub(r"\)sqrt", ")*sqrt", s)
    return sp.nsimplify(sp.sympify(s))


def same(v, s) -> bool:
    return sp.simplify(sp.nsimplify(v) - tex(s)) == 0


def nums(s: str):
    """YAML 집합형 답에서 수(분수·근호 포함)를 순서대로 추출."""
    parts = re.split(r"[①②③④]", s.replace("\\ ", " "))
    return [p.strip().strip(",").strip() for p in parts if p.strip()]


def cubic_level_k(f, lo, hi, minval):
    """닫힌구간 [lo,hi] 최솟값 = minval 로 상수 a 결정 후, y=k 와 교점 2개인 k 전체."""
    crit = [r for r in sp.solve(sp.diff(f, x), x) if r.is_real]
    cands = [f.subs(x, p) for p in crit if lo <= p <= hi] + [f.subs(x, lo), f.subs(x, hi)]
    at0 = [sp.simplify(e.subs(a, 0)) for e in cands]
    amin = cands[at0.index(min(at0))]
    aval = sp.solve(sp.Eq(amin, minval), a)[0]
    g = f.subs(a, aval)
    ks = []
    for kk in sorted({sp.nsimplify(g.subs(x, p)) for p in crit} | {sp.Integer(0)}):
        roots = sp.Poly(g - kk, x).real_roots()
        if len(set(roots)) == 2:
            ks.append(kk)
    return aval, ks


def p07():
    ms = []
    for mv in range(-30, 31):
        fp = sp.diff(x**3 - mv*x**2 + (mv + sp.Rational(4, 3))*x + 5, x)
        xm = sp.solve(sp.diff(fp, x), x)[0]
        if fp.subs(x, xm) >= 0:          # 이차 f'의 최솟값 ≥ 0 이면 증가(일대일)
            ms.append(mv)
    want = sorted(int(v) for v in re.findall(r"-?\d+", yaml_answer(7)))
    report(7, ms == want, f"정수 m={ms}")


def p08():
    A, B = sp.symbols("A B", real=True)
    vals = set()
    for sg in (1, -1):                     # sqrt(a^4 b^-2) = a^2/|b|
        for s in sp.solve([sp.Eq(1/(A*sg*B), 3), sp.Eq(1/A - 1/B, -5)], [A, B], dict=True):
            if s[A].is_real and s[B].is_real and sg*s[B] > 0 and s[A] != 0:
                assert sp.simplify(s[A]**-3 * sp.sqrt(s[A]**4 * s[B]**-2) - 3) == 0
                vals.add(sp.nsimplify(sp.simplify(s[A]**2 + s[B]**2)))
    v = list(vals)
    report(8, len(v) == 1 and same(v[0], yaml_answer(8)), f"a^2+b^2={v} (b<0 경우는 해 없음)")


def p09():
    sols = sp.solveset(sp.tan(x) + 1/sp.tan(x) + 2, x, sp.Interval.open(0, sp.pi))
    vals = {sp.simplify(1/sp.cos(s) - 1/sp.sin(s)) for s in sols}
    v = list(vals)
    report(9, len(v) == 1 and same(v[0], yaml_answer(9)), f"θ={sols}, 값={v}")


def p10():
    f = lambda z: sp.Piecewise((2*z + a**2, z <= -1), (-z + 2*a + 1, True))
    gfun = f(x**2) * f(2*x - 1)
    right = sp.limit(gfun.subs(x, sp.Symbol("h", positive=True)), sp.Symbol("h", positive=True), 0)
    left = sp.limit(gfun.subs(x, -sp.Symbol("h", positive=True)), sp.Symbol("h", positive=True), 0)
    val0 = gfun.subs(x, 0)
    sol = set(sp.solve(sp.Eq(right, val0), a)) & set(sp.solve(sp.Eq(left, val0), a) or sp.solve(sp.Eq(right, val0), a))
    s = sp.simplify(sum(sol))
    report(10, same(s, yaml_answer(10)), f"a={sol}, 합={s}")


def p11():
    res = []
    for p in range(1, 60):
        for q in range(-400, 0):
            seq = [30]
            for nn in range(1, 4):
                seq.append(seq[-1] + 3*p*nn + q)
            if seq[2] > 0 and seq[3] < 0:
                res.append((p, q))
    want = sorted(tuple(int(v) for v in m) for m in re.findall(r"\((-?\d+),\s*(-?\d+)\)", yaml_answer(11)))
    report(11, sorted(res) == want, f"(p,q)={res}")


def p12():
    B = sp.symbols("B", positive=True)
    bv = [r for r in sp.solve((sp.Rational(5, 3) - B)*(sp.Rational(5, 3) + B) - 1, B)]   # y좌표 부호 반대 ⇔ 진수의 곱 1
    bv = [r for r in bv if sp.Rational(5, 3) - r > 0][0]
    xq = sp.Rational(5, 3) + bv
    yq = [r for r in sp.solve((xq - sp.Rational(5, 3))**2 + x**2 - sp.Rational(17, 9), x) if r > 0][0]  # a>1, x>1 → y>0
    av = sp.solve(sp.Eq(sp.log(xq) / sp.log(a), yq), a)[0]
    # 수치 확인: y=log_a x 와 원의 교점이 정확히 두 점이고 중점이 중심
    import mpmath as mp
    hf = lambda z: (z - mp.mpf(5)/3)**2 + (mp.log(z)/mp.log(float(av)))**2 - mp.mpf(17)/9
    zs = [mp.mpf(i)/1000 for i in range(1, 3400)]
    roots = sorted({round(float(zs[i] + zs[i+1]) / 2, 2) for i in range(len(zs)-1) if hf(zs[i]) * hf(zs[i+1]) <= 0})
    assert len(roots) == 2 and abs(sum(roots) / 2 - 5/3) < 2e-3
    want = nums(yaml_answer(12))
    ok = (same(xq, want[0].replace("b", "(" + str(bv) + ")"))
          and same(bv, want[1]) and same(yq, want[2]) and same(av, want[3]))
    report(12, ok, f"b={bv}, Q=({xq},{yq}), a={av}, 교점 x={roots}")


def p13():
    r1, r2 = sp.symbols("r1 r2")
    f = 4 + 2*(x - 2) + r1*(x - 2)**2          # 조건을 만족하는 임의의 연속함수 대표(고차항 임의)
    g = -2 - (x - 2) + r2*(x - 2)**2
    assert sp.limit((f - 4)/(x - 2), x, 2) == 2 and sp.limit((g + 2)/(x - 2), x, 2) == -1
    L = sp.simplify(sp.limit((f*g + 8)/(x**2 - 4), x, 2))
    report(13, same(L, yaml_answer(13)), f"극한값={L} (r1,r2 무관)")


def p14():
    p, q = sp.symbols("p q")
    f = 3*x**2 + p*x + q
    pv = sp.solve(sp.diff(f, x).subs(x, 3), p)[0]
    f = f.subs(p, pv)
    g = sp.expand((1 - x)*f + sp.integrate(f.subs(x, t), (t, 1, x)))
    assert q not in g.free_symbols

    def count(kv):
        pts = set()
        for sgn in (1, -1):
            for r in sp.Poly(sp.expand(sgn*g - kv*x), x).nroots(n=30):
                if abs(sp.im(r)) < 1e-8 and sgn*g.subs(x, sp.re(r)) >= -1e-8:
                    pts.add(round(float(sp.re(r)), 5))
        return len(pts)
    tt = [r for r in sp.solve(sp.Eq(g.subs(x, t), sp.diff(g, x).subs(x, t)*t), t)]
    cands = sorted({sp.simplify(sp.diff(g, x).subs(x, r)) for r in tt}, key=float)
    good = [kv for kv in cands if count(kv) == 3]
    # 격자 탐색: 접선 기울기 이외의 k 에서는 교점 3개가 나오지 않음
    grid = [kk/4 for kk in range(-240, 241)]
    near = [kk for kk in grid if count(sp.nsimplify(kk)) == 3]
    ok = len(good) == 1 and same(good[0], yaml_answer(14)) and not near
    report(14, ok, f"g={g}, 접선 후보 k={cands}, 교점3개 k={good}, 격자에서 3개={near[:3]}")


def p15():
    fpw = sp.Piecewise((x**2 - 10*x + 9, x**2 - 5*x >= 0), (-x**2 + 9, True))
    f = sp.Abs(x**2 - 5*x) - 5*x + 9
    zs = sorted(r for r in sp.solve(x**2 - 10*x + 9, x) + sp.solve(-x**2 + 9, x) if f.subs(x, r) == 0)
    cuts = sorted({zs[0], zs[-1]} | {r for r in sp.solve(x**2 - 5*x, x) if zs[0] < r < zs[-1]})
    S = 0
    for lo, hi in zip(cuts, cuts[1:]):
        br = x**2 - 10*x + 9 if ((lo + hi)/2)**2 - 5*(lo + hi)/2 >= 0 else -x**2 + 9
        val = sp.integrate(br, (x, lo, hi))
        S += abs(val)                       # 각 구간에서 부호 일정(근은 zs 뿐)
    assert all(f.subs(x, sp.Rational(i, 10)) < 0 for i in range(31, 90))
    report(15, same(S, yaml_answer(15)), f"x축 교점={zs}, 넓이={S}")


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
