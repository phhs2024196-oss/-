#!/usr/bin/env python3
"""HYU-G25C (한양대 2025학년도 논술 자연계열 오후2) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건에서 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "HYU-G25C"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HYU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def md_parts(no: int, sub: str, loc=None) -> dict:
    """'키 = 식; 키 = 식' 형식 answer.value 를 사전으로(값은 sympy 식)."""
    return {k.strip(): sp.sympify(v.strip(), locals=loc or {})
            for k, v in (seg.split("=", 1) for seg in md_answer(no, sub).split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    m, k = sp.symbols("m k", real=True)
    # (1) g(x)=f(x-m)+k 가 기함수 → 모든 x 에서 g(-x)+g(x)=0 이 되도록 계수 비교
    f = x**3 + 3 * x**2 - x - 6
    g = f.subs(x, x - m) + k
    sol = sp.solve(sp.Poly(sp.expand(g.subs(x, -x) + g), x).coeffs(), [m, k], dict=True)
    assert len(sol) == 1
    G = sp.expand(g.subs(sol[0]))
    pts = sorted(sp.real_roots(sp.Poly(G - f, x)))
    pts = sorted(set(pts))
    area = sum(sp.Abs(sp.integrate(G - f, (x, pts[i], pts[i + 1]))) for i in range(len(pts) - 1))
    md1 = md_parts(1, "1")
    ok1 = md1["m"] == sol[0][m] and md1["n"] == sol[0][k] and sp.simplify(area - md1["area"]) == 0
    # (2) B(0,0), A(0,3), C(4,0). P(0,x), Q(q,0), R = Q + (P-Q) 을 90° 회전 → R 이 AC 위
    q = sp.symbols("q", real=True)
    X = sp.symbols("x", real=True)
    P = sp.Matrix([0, X]); Q = sp.Matrix([q, 0])
    v = P - Q
    best = None
    for rot in (sp.Matrix([[0, 1], [-1, 0]]), sp.Matrix([[0, -1], [1, 0]])):
        R = Q + rot * v
        qs = sp.solve(sp.Eq(3 * R[0] + 4 * R[1], 12), q)
        for qq in qs:
            Rv = R.subs(q, qq)
            ok_pos = all(0 <= float(qq.subs(X, xv)) <= 4 and 0 <= float(Rv[0].subs(X, xv)) <= 4
                         for xv in (0, 0.5, 1.5, 2.5, 3))
            if ok_pos:
                best = (qq, Rv)
    qq, Rv = best
    S = sp.expand(sp.Rational(1, 2) * v.subs(q, qq).dot(v.subs(q, qq)))
    md2 = md_parts(1, "2", {"x": X})
    crit = sp.solve(sp.diff(S, X), X)[0]
    ok2 = sp.simplify(S - md2["S(x)"]) == 0 and crit == md2["x_min"] and 0 <= crit <= 3 and sp.diff(S, X, 2) > 0
    # (3) 정삼각형 한 변 1 좌표로 A2, B2 교점 → 넓이비
    T = sp.symbols("t", positive=True)
    A1 = sp.Matrix([sp.Rational(1, 2), sp.sqrt(3) / 2]); B1 = sp.Matrix([0, 0]); C1 = sp.Matrix([1, 0])
    A1p = A1 + T * (B1 - A1); B1p = B1 + T * (C1 - B1); C1p = C1 + T * (A1 - C1)
    u, w = sp.symbols("u w", real=True)

    def meet(P1, P2, P3, P4):
        s_ = sp.solve(list(P1 + u * (P2 - P1) - (P3 + w * (P4 - P3))), [u, w], dict=True)[0]
        return sp.simplify(P1 + s_[u] * (P2 - P1))
    A2 = meet(A1, B1p, C1, A1p)
    B2 = meet(B1, C1p, A1, B1p)
    C2 = meet(C1, A1p, B1, C1p)
    d = lambda U, V: sp.simplify((U - V).dot(U - V))
    equil = sp.simplify(d(A2, B2) - d(B2, C2)) == 0 and sp.simplify(d(B2, C2) - d(C2, A2)) == 0
    r = sp.simplify(d(A2, B2) / d(A1, B1))
    md3 = md_parts(1, "3", {"t": T})
    ok3a = equil and sp.simplify(r - md3["r"]) == 0
    tsol = [z for z in sp.solve(sp.Eq(1 / (1 - r), 3), T) if 0 < z < sp.Rational(1, 2)]
    rv = [float(r.subs(T, tv)) for tv in (0.01, 0.2, 0.49)]
    ok3 = ok3a and len(tsol) == 1 and sp.simplify(tsol[0] - md3["t"]) == 0 and all(0 < z < 1 for z in rv)
    report(1, ok1 and ok2 and ok3,
           f"m,n={sol[0][m]},{sol[0][k]} 교점{pts} 넓이={area}; S(x)={S} 최소 x={crit}; r={sp.factor(r)} t={tsol}")


def p02():
    a, b = sp.symbols("a b", real=True)
    # (1) 양변을 두 번 미분: f''=h''+2f'' ⇒ f''=-h'' ⇒ f=-h+ax+b, 원식에 대입해 a,b 결정
    h = sp.exp(x) * (3 * sp.sin(x) - sp.cos(x))
    f = -h + a * x + b
    tt = sp.symbols("tt", real=True)
    f2 = sp.diff(f, x, 2).subs(x, tt)
    rhs = h + 2 * sp.integrate(f2 * (x - tt), (tt, 0, x)) + sp.integrate(sp.exp(tt) * f2, (tt, 0, sp.pi / 2))
    resid = sp.simplify(sp.expand(rhs - f))
    sol = sp.solve([resid.subs(x, 0), sp.diff(resid, x).subs(x, 0)], [a, b], dict=True)[0]
    F = f.subs(sol)
    chk = all(abs(sp.N(resid.subs(sol).subs(x, xv), 30)) < 1e-20 for xv in (-1, 0.3, 2))
    ok1 = chk and sp.simplify(F.subs(x, 0) - sp.sympify(md_answer(2, "1"))) == 0
    # (2) 일반 사차식 대입 → 항등식 계수비교
    c4, c3, c2, c1, c0 = sp.symbols("c4 c3 c2 c1 c0", real=True)
    p = c4 * x**4 + c3 * x**3 + c2 * x**2 + c1 * x + c0
    pt = p.subs(x, tt)
    rhs = 4 * x**4 + sp.integrate(sp.diff(pt, tt, 2) * (3 * x - 5 * tt), (tt, 0, x)) \
        + sp.Rational(3, 2) * sp.integrate(tt * pt, (tt, 0, 2))
    eqs = sp.Poly(sp.expand(rhs - p), x).coeffs()
    sl = sp.solve(eqs, [c4, c3, c1, c0], dict=True)
    sl = [z for z in sl if z[c4] != 0]
    assert len(sl) == 1
    pp = p.subs(sl[0])                                       # c2 는 자유
    I3 = sp.simplify(sp.integrate(pp, (x, 0, 3)))
    ok2 = sp.simplify(I3 - sp.sympify(md_answer(2, "2"))) == 0 and not I3.has(c2)
    # (3) [0,1] 에서 증가, 역함수 r 과 x축·y축·y=1 로 둘러싸인 넓이 159/5
    #     y∈[0,1] 에서 곡선 x=q(y), 둘레 x=0 → 넓이 = ∫_0^1 |q(y)| dy (q 의 부호가 일정할 때)
    cands = []
    for s in (1, -1):
        for cv in sp.solve(sp.Eq(s * sp.integrate(pp, (x, 0, 1)), sp.Rational(159, 5)), c2):
            q = pp.subs(c2, cv)
            ys = [i / 200 for i in range(201)]
            vals = [float(q.subs(x, y)) for y in ys]
            incr = all(float(sp.diff(q, x).subs(x, y)) >= 0 for y in ys)
            const_sign = all(v * s > 0 for v in vals)
            if incr and const_sign:
                cands.append((cv, q))
    ok3 = len(cands) == 1 and cands[0][1].subs(x, 1) == sp.sympify(md_answer(2, "3"))
    report(2, ok1 and ok2 and ok3,
           f"f(0)={sp.simplify(F.subs(x, 0))}, p={sp.expand(pp)}, ∫0^3 p={I3}, 조건 만족 c2={[c for c, _ in cands]}, p(1)={[q.subs(x, 1) for _, q in cands]}")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    m = re.search(r"-s(\d+)\.py$", __file__)
    label = PREFIX + (f"-s{m.group(1)}" if m else "")
    print(f"SUMMARY {label}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
