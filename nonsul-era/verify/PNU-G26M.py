#!/usr/bin/env python3
"""PNU-G26M (부산대 2026학년도 수시 지역인재전형(논술) 의·약학계) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지. YAML answer 를 읽어 독립 재계산값과 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "PNU-G26M"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/PNU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def normal_xint(f, t, x):
    """곡선 y=f(x) 위 (t,f(t))에서의 법선의 x절편(일반식, 접선 기울기 0인 점 포함)."""
    ft, dft = f.subs(x, t), sp.diff(f, x).subs(x, t)
    return sp.expand(t + ft * dft)


def count_level(g, t, c):
    """g(t)=c 의 서로 다른 실근 개수."""
    return len(set(sp.real_roots(sp.Poly(sp.expand(g - c), t))))


def p01():
    x, t, k = sp.symbols("x t k", real=True)
    # (1-1) p = 1,2,3,... 에 대해 n(A_k)=2 인 k 가 존재하는 최소 p 를 직접 탐색
    p1 = None
    for pv in range(1, 20):
        f = x**2 + pv * x + sp.Rational(pv, 2) - sp.Rational(1, 2)
        g = normal_xint(f, t, x)
        crit = sp.solve(sp.diff(g, t), t)
        cands = [c for c in crit if c.is_real]
        if any(count_level(g, t, g.subs(t, c)) == 2 for c in cands):
            p1 = pv
            break
    f = x**2 + p1 * x + sp.Rational(p1, 2) - sp.Rational(1, 2)
    g = normal_xint(f, t, x)
    ks = set()
    for c in [c for c in sp.solve(sp.diff(g, t), t) if c.is_real]:
        ks |= set(sp.real_roots(sp.Poly(g - g.subs(t, c), t)))
    s = sum(ks)
    parts = dict(seg.split("=") for seg in md_answer(1, "1").replace(" ", "").split(";"))
    ok1 = p1 == int(parts["p1"]) and sp.simplify(s - sp.sympify(parts["sum"])) == 0 and len(ks) == 4
    # 법선 x절편 식 자체 확인: 점 (t,f(t)) 에서 기울기 -1/f'(t) 직선의 y=0 해
    X = sp.symbols("X")
    tv = sp.Rational(7, 5)
    sl = sp.diff(f, x).subs(x, tv)
    xi = sp.solve(sp.Eq(0 - f.subs(x, tv), -1 / sl * (X - tv)), X)[0]
    ok1 = ok1 and sp.simplify(xi - g.subs(t, tv)) == 0
    # (1-2) f = x^2+x-7 : k 를 조밀하게 훑어 조건(n(A_k)=2 또는 m_k<k<M_k)을 만족하는 k 에서 M^2+m^2 의 최대·최소
    f2 = x**2 + x - 7
    g2 = normal_xint(f2, t, x)
    vals = []
    kgrid = [sp.Rational(i, 40) for i in range(-200, 201)]
    extra = []
    for c in [c for c in sp.solve(sp.diff(g2, t), t) if c.is_real]:
        extra += list(sp.real_roots(sp.Poly(g2 - g2.subs(t, c), t)))
    for kv in kgrid + extra:
        roots = sorted(set(sp.real_roots(sp.Poly(g2 - g2.subs(t, kv), t))))
        n = len(roots)
        mk, Mk = roots[0], roots[-1]
        if n == 2 or (mk < kv < Mk):
            vals.append(sp.nsimplify(Mk**2 + mk**2))
    mx, mn = max(vals), min(vals)
    ok2 = sp.simplify(mx + mn - sp.sympify(md_answer(1, "2"))) == 0
    report(1, ok1 and ok2, f"p1={p1} k={sorted(ks)} sum={s}; (1-2) max={mx} min={mn} sum={mx+mn}")


def p02():
    import mpmath as mp
    mp.mp.dps = 30
    E = mp.e
    g = lambda s: (mp.e**s - E**2) * (mp.e**s - 1 / E)
    gp = lambda s: 2 * mp.e**(2 * s) - (E**2 + 1 / E) * mp.e**s
    # (가) x=2, -1 에서 f(|g|) 의 좌우 미분계수 = ∓f'(0)g'(.) → b=f'(0)=0 이어야 함 (g' ≠ 0 확인)
    okA = abs(gp(2)) > 1 and abs(gp(-1)) > 1e-3
    # 기호로: f(x)=(ax^2+bx)e^x 의 f'(0) = b
    a, b, X = sp.symbols("a b X", real=True)
    okA = okA and sp.diff((a * X**2 + b * X) * sp.exp(X), X).subs(X, 0) == b

    # (나) a 에 대해 |g(f(x))| 가 미분불가능한 점 개수: f(x) ∈ {-1, 2} 이고 f'(x) ≠ 0 인 x 의 개수
    def nondiff_count(av):
        f = lambda x: av * x**2 * mp.e**x
        fp = lambda x: av * (x**2 + 2 * x) * mp.e**x
        cnt = 0
        xs = [-60 + i * 0.01 for i in range(6601)]
        for c in (-1, 2):
            prev = f(xs[0]) - c
            for x0, x1 in zip(xs, xs[1:]):
                cur = f(x1) - c
                if prev == 0 or prev * cur < 0:
                    r = mp.findroot(lambda z: f(z) - c, (x0, x1), solver="bisect")
                    if abs(fp(r)) > 1e-8:
                        cnt += 1
                prev = cur
        return cnt
    lo, hi = -float(E**2 / 4), float(E**2 / 2)
    tests = [(lo * 1.02, False), (lo * 0.98, True), (lo * 0.5, True), (hi * 0.5, True), (hi * 0.98, True), (hi * 1.02, False)]
    okB = all((nondiff_count(av) == 1) == want for av, want in tests)
    # 경계 a=-e^2/4, e^2/2 는 접하는 점(f'=0)이라 미분가능 → 포함
    okB = okB and nondiff_count(lo) == 1 and nondiff_count(hi) == 1
    ans1 = md_answer(2, "1").replace(" ", "")
    okB = okB and "[-exp(2)/4,0)U(0,exp(2)/2]" in ans1
    # p = max |g(s)|, s=4a e^{-2} ∈ [-1,0)∪(0,2]  (격자 + 정밀화)
    grid = [-1 + i * 3e-4 for i in range(10001)]
    grid = [s for s in grid if abs(s) > 1e-9]
    s0 = max(grid, key=lambda s: abs(g(s)))
    smax = mp.findroot(gp, s0)
    pnum = abs(g(smax))
    psym = sp.sympify(ans1.split(";")[1].split("=", 1)[1])
    okP = abs(pnum - mp.mpf(str(sp.N(psym, 30)))) < 1e-20 and 0 < smax <= 2
    # (2-2) a = s* e^2/4 로 h(t) 직접 계수
    av = smax * E**2 / 4
    F = lambda x: abs(g(av * x**2 * mp.e**x))
    xs = [-80 + i * 0.0005 for i in range(int(84 / 0.0005) + 1)]
    Fv = [F(x0) for x0 in xs]

    def h(tv):
        if tv < 0:
            return 0
        c = 0
        for v0, v1 in zip(Fv, Fv[1:]):
            if (v0 - tv) == 0 or (v0 - tv) * (v1 - tv) < 0:
                c += 1
        return c
    eps = pnum * mp.mpf("1e-4")
    g0 = g(0)
    val = h(pnum + eps) + h(pnum - eps) + h(g0)
    ok22 = val == int(md_answer(2, "2")) and g0 < 0 and abs(g0) < pnum
    report(2, okA and okB and okP and ok22,
           f"b=0; a-range checks ok={okB}; s*={mp.nstr(smax, 12)} p={mp.nstr(pnum, 15)}; h(p+)+h(p-)+h(g(0))={h(pnum+eps)}+{h(pnum-eps)}+{h(g0)}={val}")


def p03():
    # 좌표: A(0,0,4) B(4,0,4) C(4,4,4) D(0,4,4) E(0,0,0) F(4,0,0) G(4,4,0) H(0,4,0)
    V = sp.Matrix
    A, B, C, D = V([0, 0, 4]), V([4, 0, 4]), V([4, 4, 4]), V([0, 4, 4])
    E_, F, G, H = V([0, 0, 0]), V([4, 0, 0]), V([4, 4, 0]), V([0, 4, 0])
    div = lambda P1, P2: P1 + (P2 - P1) / 4          # P1P2 를 1:3 내분
    I, J, K, L, M, N = div(C, D), div(H, D), div(H, E_), div(F, E_), div(F, B), div(C, B)
    nh = (J - I).cross(K - I)
    okplane = all(nh.dot(Pt - I) == 0 for Pt in (J, K, L, M, N))
    k = sp.symbols("k", positive=True)
    P, Q, R = A + V([0, k, 0]), A + V([0, 0, -3 * k]), A + V([3 * k, 0, 0])
    npqr = (Q - P).cross(R - P)
    cos = sp.simplify(sp.Abs(npqr.dot(nh)) / (npqr.norm() * nh.norm()))
    ok1 = sp.simplify(cos - sp.sympify(md_answer(3, "1"))) == 0
    # (3-2) 정사영이 사각형 IKLN(볼록) 안에 있는 최대 k (정사영 넓이 ∝ k^2 이므로 최대 k 를 찾으면 됨)
    u = nh / nh.norm()
    proj = lambda X_: X_ - u * (u.dot(X_ - I))
    quad = [I, K, L, N]
    # 볼록사각형 순서 확인 후 각 변에 대한 부등식
    cen = sum(quad, V([0, 0, 0])) / 4

    def inside_conds(Pt):
        conds = []
        for a1, a2 in zip(quad, quad[1:] + quad[:1]):
            side = u.dot((a2 - a1).cross(Pt - a1))
            ref = u.dot((a2 - a1).cross(cen - a1))
            conds.append(sp.expand(side * sp.sign(ref)))
        return conds
    bounds = []
    for X_ in (P, Q, R):
        for cnd in inside_conds(proj(X_)):
            sol = sp.solve(sp.Eq(cnd, 0), k)
            for s_ in sol:
                if s_.is_positive:
                    bounds.append(s_)
    # 후보 중 모든 조건을 만족하는 최대 k (Q 가 선분 AE 위: 3k ≤ 4)
    def feasible(kv):
        return 3 * kv <= 4 and all(c.subs(k, kv) >= 0 for X_ in (P, Q, R) for c in inside_conds(proj(X_)))
    cands = sorted(set(bounds + [sp.Rational(4, 3)]))
    kmax = max(c for c in cands if feasible(c))
    okmono = feasible(kmax / 2) and not feasible(kmax * sp.Rational(101, 100))
    ok2 = sp.simplify(kmax - sp.sympify(md_answer(3, "2"))) == 0 and okmono
    report(3, okplane and ok1 and ok2, f"hexagon coplanar={okplane}; cos={cos}; k_max={kmax} (후보 {cands})")


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
