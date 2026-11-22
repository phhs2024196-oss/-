#!/usr/bin/env python3
"""서울과기대 2022학년도 모의논술고사(자연계열) (SNUT-M22-01~03) — sympy/mpmath 검산. 정답 하드코딩 비교 없이 문제 조건에서 독립 재계산해 YAML answer 와 비교."""
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "SNUT-M22"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
a, b, d, k = sp.symbols("a b d k", real=True)
theta, theta_k, alpha = sp.symbols("theta theta_k alpha", real=True)
SYM = {"x": x, "t": t, "n": n, "a": a, "b": b, "d": d, "k": k, "theta": theta,
       "theta_k": theta_k, "alpha": alpha, "pi": sp.pi, "E": sp.E}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0, raw=False):
    """yaml 의 subparts[sub].answer.value 를 읽는다(raw=True 면 문자열 그대로)."""
    import pathlib
    import yaml
    d_ = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/SNUT/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    v = d_["subparts"][sub]["answer"]["value"]
    return v if raw else sp.sympify(v, locals=SYM)


def same(u, v, tol=1e-9):
    """수치 동일성(상대오차 tol)."""
    return abs(complex(sp.N(u, 30)) - complex(sp.N(v, 30))) <= tol * max(1, abs(complex(sp.N(v, 30))))

def p01():
    # 1.1
    A, B = sp.symbols("A B", real=True)
    T = lambda tv: 24 + A * sp.exp(B * tv)
    s = sp.solve([sp.Eq(T(0), 88), sp.Eq(T(5), 72)], [A, B], dict=True)[0]
    val = sp.simplify(T(15).subs(s))
    ans = load_answer(1, 0)
    report(1, sp.simplify(val - ans) == 0, f"1.1 a={s[A]}, b={s[B]}, T(15)={val}, yaml={ans}")
    # 1.2
    X, Y = sp.symbols("X Y", real=True)
    F = X ** 3 + X * Y + Y ** 3 - 8
    px = [r for r in sp.solve(F.subs(Y, 0), X) if r.is_real][0]
    m1 = sp.simplify(sp.idiff(F, Y, X).subs({X: px, Y: 0}))
    m2 = -1 / m1
    l1 = lambda xv: m1 * (xv - px)
    l2 = lambda xv: m2 * (xv - px)
    inter = sp.solve(sp.Eq(l1(X), l2(X)), X)
    pts = [(0, l1(0)), (0, l2(0)), (px, 0)]
    area = sp.Abs(sum(pts[i][0] * pts[(i + 1) % 3][1] - pts[(i + 1) % 3][0] * pts[i][1] for i in range(3))) / 2
    ans = load_answer(1, 1)
    report(1, inter == [px] and sp.simplify(area - ans) == 0, f"1.2 P=({px},0), 기울기 {m1}, {m2}, y절편 {l1(0)}, {l2(0)}, 넓이={area}, yaml={ans}")
    # 1.3 항별 정확한 적분의 합
    tot = sum(sp.Integer(kk) ** 2 * sp.Rational(1 - (-1) ** kk, kk) for kk in range(1, 2023))
    chk = sp.integrate(sum(sp.Integer(kk) ** 2 * x ** (kk - 1) for kk in range(1, 8)), (x, -1, 1)) == sum(sp.Integer(kk) ** 2 * sp.Rational(1 - (-1) ** kk, kk) for kk in range(1, 8))
    ans = load_answer(1, 2)
    report(1, chk and tot == ans, f"1.3 합={tot}, yaml={ans}")


def p02():
    aa = sp.symbols("aa", positive=True)
    f = lambda xv, av: xv ** 3 / sp.sqrt(3) - av * xv ** 2 + 1
    g = lambda xv, av: xv ** 3 / sp.sqrt(3) + av * xv ** 2 - 1
    # 2.1 몇 개의 a 에 대해 교점 풀이 후 yaml 식과 비교
    ex = load_answer(2, 0, raw=True)
    exv = sp.sympify(ex, locals=SYM)
    ok = True
    for av in [sp.Rational(1, 3), 2, 5]:
        r = sp.solve(sp.Eq(f(x, av), g(x, av)), x)
        pts = sorted([(sp.nsimplify(rr), sp.simplify(f(rr, av))) for rr in r], key=lambda p: sp.N(p[0]))
        mine = sorted([(sp.N(p[0]), sp.N(p[1])) for p in pts])
        theirs = sorted([(sp.N(q[0].subs(a, av)), sp.N(q[1].subs(a, av))) for q in exv])
        ok = ok and len(mine) == 2 and all(abs(u[0] - v[0]) < 1e-12 and abs(u[1] - v[1]) < 1e-12 for u, v in zip(mine, theirs))
    report(2, ok, "2.1 f=g 의 교점이 a=1/3, 2, 5 에서 yaml 의 (±1/√a, ±1/(a√(3a))) 와 일치")
    # 2.2 모든 교점에서 f'·g' = -1 → a
    r = sp.solve(sp.Eq(f(x, aa), g(x, aa)), x)
    conds = [sp.simplify((sp.diff(f(x, aa), x) * sp.diff(g(x, aa), x)).subs(x, rr) + 1) for rr in r]
    sa = sp.solve(conds[0], aa)
    sa = [s for s in sa if s.is_real and s > 0]
    both = all(sp.simplify(cd.subs(aa, sa[0])) == 0 for cd in conds)
    ans = load_answer(2, 1)
    report(2, len(sa) == 1 and both and sa[0] == ans, f"2.2 양의 해 a={sa} (두 교점 모두 수직 확인={both}), yaml={ans}")
    # 2.3
    a0 = sa[0]
    ineq = sp.solve_univariate_inequality(f(x, a0) - g(x, a0) >= 0, x, relational=False)
    got = f"{ineq.start} <= x <= {ineq.end}" if isinstance(ineq, sp.Interval) else str(ineq)
    ans = load_answer(2, 2, raw=True)
    report(2, got.replace(" ", "") == ans.replace(" ", ""), f"2.3 f≥g 인 x: {got}, yaml={ans}")
    # 2.4 부호가 바뀌는 점으로 구간을 나눠 절댓값 적분
    dif = sp.expand(f(x, a0) - g(x, a0))
    cuts = sorted([-2] + [rr for rr in sp.solve(dif, x) if -2 < rr < 2] + [2])
    area = 0
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        mid = (lo + hi) / sp.Integer(2)
        sg = sp.sign(dif.subs(x, mid))
        area += sp.integrate(sg * dif, (x, lo, hi))
    ans = load_answer(2, 3)
    report(2, sp.simplify(area - ans) == 0, f"2.4 구간 {cuts}, 넓이={area}, yaml={ans}")


def _sim(al, steps=400):
    """정사각형 열을 직접 만들어 l_n, S_n 을 구한다(좌표 계산)."""
    import math
    Aa = (0.0, 1.0); Bb = (0.0, 0.0); Cc = (1.0, 0.0); Dd = (1.0, 1.0)
    L, S = [], []
    mix = lambda P, Q: (P[0] + al * (Q[0] - P[0]), P[1] + al * (Q[1] - P[1]))
    for _ in range(steps):
        An, Bn, Cn, Dn = mix(Aa, Bb), mix(Bb, Cc), mix(Cc, Dd), mix(Dd, Aa)
        L.append(math.dist(Aa, An))
        S.append(abs((An[0] - Aa[0]) * (Dn[1] - Aa[1]) - (An[1] - Aa[1]) * (Dn[0] - Aa[0])) / 2)   # 삼각형 A_(n-1) A_n D_n
        Aa, Bb, Cc, Dd = An, Bn, Cn, Dn
    return L, S


def p03():
    import math
    # 3.1 θ=∠A0 D1 A1, t=tanθ → α
    al = sp.symbols("al", positive=True)
    A0 = sp.Matrix([0, 1]); A1 = sp.Matrix([0, 1 - al]); D1 = sp.Matrix([1 - al, 1])
    v1 = A0 - D1; v2 = A1 - D1
    tanth = sp.simplify((v1[0] * v2[1] - v1[1] * v2[0]) / v1.dot(v2))   # 0<α<1/2 에서 외적>0, 내적>0
    sol = sp.solve(sp.Eq(t, tanth), al)
    rhs = sp.sympify(load_answer(3, 0, raw=True).split("=")[1], locals=SYM)
    report(3, len(sol) == 1 and sp.simplify(sol[0] - rhs) == 0, f"3.1 tanθ={tanth}, α={sol}, yaml={rhs}")
    # 3.2 시뮬레이션: Σ l_n
    ok = True
    ex = load_answer(3, 1)
    for av in [0.1, 0.25, 0.4]:
        L, _ = _sim(av)
        tv = av / (1 - av)
        ok = ok and abs(sum(L) - float(ex.subs(t, tv))) < 1e-9
    report(3, ok, "3.2 정사각형 열을 좌표로 만들어 Σl_n(400항) = yaml 식 (α=0.1, 0.25, 0.4)")
    # 3.3 Σ S_(2n-1) = 2/13 인 α 를 시뮬레이션 + 이분법으로 찾기
    def F(av):
        _, S = _sim(av, 800)
        return sum(S[0::2]) - 2 / 13
    lo, hi = 1e-6, 0.4999
    sg = F(lo) * F(hi) < 0
    for _ in range(80):
        mid = (lo + hi) / 2
        if F(lo) * F(mid) <= 0:
            hi = mid
        else:
            lo = mid
    ans = load_answer(3, 2)
    report(3, sg and abs((lo + hi) / 2 - float(ans)) < 1e-9, f"3.3 이분법 α≈{(lo + hi) / 2:.10f} (0<α<1/2), yaml={ans}")


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
