#!/usr/bin/env python3
"""SKKU-G21C (성균관대 2021학년도 수시 논술우수전형 자연계 3교시 수학 1·2) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import re
import sys

import sympy as sp

PREFIX = "SKKU-G21C"

WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SKKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def md_parts(no: int, sub: str, loc=None) -> dict:
    """'key = expr; key2 = expr2' 형식의 정답 문자열을 dict 로."""
    out = {}
    for seg in md_answer(no, sub).split(";"):
        k, v = seg.split("=", 1)
        out[k.strip()] = sp.sympify(v.strip(), locals=loc or {})
    return out


def eq(a, b) -> bool:
    return sp.simplify(sp.nsimplify(a) - sp.nsimplify(b)) == 0 if not isinstance(a, tuple) else all(eq(u, w) for u, w in zip(a, b))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


x, y = sp.symbols("x y", real=True)


def p01():
    a = sp.symbols("a", positive=True)
    SD = sp.integrate(-x**2 + 4, (x, 0, 2))
    b = 2 * (SD / 2) / a                                        # 삼각형 넓이 ab/2 = SD/2
    alo = sp.solve(sp.Eq(b, 4), a)[0]; ahi = 2                  # 0<b<=4, 0<a<=2
    # 원점-직선 거리 d(a) = (SD) / sqrt(a^2+b^2) — 끝점과 임계점 비교
    d = (a * b) / sp.sqrt(a**2 + b**2)
    cands = [alo, sp.Integer(ahi)] + [c for c in sp.solve(sp.diff(d, a), a) if alo <= c <= ahi]
    a1 = min(cands, key=lambda c: d.subs(a, c)); a2 = max(cands, key=lambda c: d.subs(a, c))
    L1 = b.subs(a, a1) * x + a1 * y - a1 * b.subs(a, a1)
    L2 = b.subs(a, a2) * x + a2 * y - a2 * b.subs(a, a2)
    P = md_parts(1, "1", {"x": x, "y": y})
    prop = lambda u, w: sp.simplify(sp.Poly(u, x, y).as_expr() * 1 - w * (sp.Poly(u, x, y).coeffs()[0] / sp.Poly(w, x, y).coeffs()[0])) == 0
    # 현이 D 안에 있는지(오목): 선분 위 점이 포물선 아래
    inside = all(float(b.subs(a, av) * (1 - s)) <= 4 - float(av * s) ** 2 + 1e-12 for av in (a1, a2) for s in [i / 50 for i in range(51)])
    ok1 = prop(L1, P["l1"]) and prop(L2, P["l2"]) and inside
    def foot(L):
        A_, B_, C_ = sp.Poly(L, x, y).coeff_monomial(x), sp.Poly(L, x, y).coeff_monomial(y), L.subs({x: 0, y: 0})
        s = -C_ / (A_**2 + B_**2)
        return (A_ * s, B_ * s)
    C1, C2 = foot(L1), foot(L2)
    area = sp.Abs(C1[0] * C2[1] - C1[1] * C2[0]) / 2
    ok2 = eq(area, sp.sympify(md_answer(1, "2")))
    m, n = sp.symbols("m n", real=True)
    F = -(x - m) ** 2 + 4 + n
    conds = []
    for L in (L1, L2):
        ly = sp.solve(L, y)[0]
        conds.append(sp.discriminant(sp.expand(F - ly), x))
    sol = sp.solve(conds, [m, n], dict=True)[0]
    P3 = md_parts(1, "3")
    ok3 = eq(sol[m], P3["m"]) and eq(sol[n], P3["n"])
    Fs = F.subs(sol)
    l1y, l2y = sp.solve(L1, y)[0], sp.solve(L2, y)[0]
    xt1 = sp.solve(sp.diff(Fs - l1y, x), x)[0]; xt2 = sp.solve(sp.diff(Fs - l2y, x), x)[0]
    xi = sp.solve(sp.Eq(l1y, l2y), x)[0]
    lo, hi = sorted([xt1, xt2])
    upper_lo = l2y if lo == xt2 else l1y; upper_hi = l1y if upper_lo == l2y else l2y
    SF = sp.integrate(upper_lo - Fs, (x, lo, xi)) + sp.integrate(upper_hi - Fs, (x, xi, hi))
    ok4 = eq(SF, sp.sympify(md_answer(1, "4")))
    report(1, ok1 and ok2 and ok3 and ok4, f"l1:{L1} l2:{L2} area={area} m,n={sol} SF={SF}")


def hpieces():
    """h = min(f, g) 를 구간별로 — f<g 인 구간을 직접 풀어 만든다."""
    f1, f2, g = 2 * x**2 + 4 * x + 3, 1 / x + 2, x + 3
    r1 = sorted(sp.solve(sp.Eq(f1, g), x)); r2 = [r for r in sp.solve(sp.Eq(f2, g), x) if r > 0]
    return [(g, -sp.oo, r1[0]), (f1, r1[0], 0), (g, 0, r2[0]), (f2, r2[0], sp.oo)], r1, r2


def count_roots(t):
    pcs, _, _ = hpieces()
    pts = set()
    for e, lo, hi in pcs:
        num = sp.numer(sp.together(e - t * (x + 3)))
        for r in sp.solve(num, x):
            if r.is_real and lo <= r <= hi:
                pts.add(sp.nsimplify(sp.simplify(r)) if r.is_rational else sp.N(r, 30))
    # 같은 점 중복 제거(구간 경계)
    uniq = []
    for p_ in pts:
        if all(abs(sp.N(p_ - q, 30)) > 1e-20 for q in uniq):
            uniq.append(p_)
    return len(uniq)


def p02():
    pcs, r1, r2 = hpieces()
    Qs = [(r, r + 3) for r in r1] + [(r2[0], r2[0] + 3)]
    P1 = md_parts(2, "1")
    ok1 = all(eq(tuple(P1[f"Q{i+1}"]), Qs[i]) for i in range(3))
    t = sp.symbols("t", positive=True)
    t0 = [s for s in sp.solve(sp.discriminant(2 * x**2 + 4 * x + 3 - t * (x + 3), x), t) if 0 < s < 1][0]
    P2 = md_parts(2, "2")
    below = {count_roots(sp.Rational(k, 100)) for k in range(1, int(float(t0) * 100) + 1) if sp.Rational(k, 100) < t0}
    above = {count_roots(sp.Rational(k, 100)) for k in range(int(float(t0) * 100) + 1, 100)}
    at = count_roots(t0)
    ok2 = eq(P2["t0"], t0) and below == {int(P2["k_below"])} and above == {int(P2["k_above"])} and at == int(P2["k_at"])
    # (iii) 포물선 부분 접선이 (n,0) 을 지날 조건 — 독립 유도
    n = sp.symbols("n", positive=True)
    al = sp.symbols("alpha", real=True)
    par = 2 * x**2 + 4 * x + 3
    tang0 = sp.diff(par, x).subs(x, al) * (n - al) + par.subs(x, al)          # (n,0)에서 접선 값 = 0
    roots = sp.solve(sp.Eq(tang0, 0), al)
    al_n = [r for r in roots if r.subs(n, 1) < 0][0]
    be_n = sp.expand(par.subs(x, al_n))
    loc = {"n": n}
    P3 = md_parts(2, "3", loc)
    okf = all(abs(float((P3["alpha_n"] - al_n).subs(n, k))) < 1e-9 and abs(float((P3["beta_n"] - be_n).subs(n, k))) < 1e-7
              and -1.5 < float(al_n.subs(n, k)) < 0 and float(be_n.subs(n, k)) > 0 for k in range(1, 30))
    la, lb = sp.limit(al_n, n, sp.oo), sp.limit(be_n, n, sp.oo)
    ok3 = okf and eq(la, P3["lim_alpha"]) and eq(lb, P3["lim_beta"])
    # (iv) n 별로 T, R, S 를 직접 적분으로
    A_, B_ = sp.symbols("alpha beta", real=True)
    P4 = md_parts(2, "4", {"alpha": A_, "beta": B_})
    import mpmath as mp
    mp.mp.dps = 40
    ok4 = True
    for k in (1, 2, 3, 7, 20):
        av, bv = al_n.subs(n, k), be_n.subs(n, k)
        line = sp.diff(par, x).subs(x, av) * (x - av) + bv
        Ty = line.subs(x, 0)
        Rx = [r for r in sp.solve(sp.Eq(line, x + 3), x)][0]
        S = sp.integrate(x + 3 - line, (x, Rx, r1[0])) + sp.integrate(par - line, (x, r1[0], 0))
        sub = {A_: av, B_: bv}
        ok4 &= abs(sp.N(Ty - P4["T_y"].subs(sub), 30)) < 1e-20 and abs(sp.N(Rx - P4["R_x"].subs(sub), 30)) < 1e-20 \
            and abs(sp.N(S - P4["S"].subs(sub), 30)) < 1e-20 and float(Rx) < float(r1[0])
    big = 10**8
    av, bv = mp.mpf(sp.N(al_n.subs(n, big), 40)), mp.mpf(sp.N(be_n.subs(n, big), 40))
    Slim = -(bv - 4 * av - 3) ** 2 / (2 * (4 * av + 3)) - mp.mpf(9) / 8
    ok4 &= abs(Slim - mp.mpf(sp.N(P4["lim_S"], 40))) < 1e-6
    # 공식 예시답안의 인쇄식 (beta-4alpha+3)^2 은 다른 값으로 감 — 오식 재현
    typo = -(bv - 4 * av + 3) ** 2 / (2 * (4 * av + 3)) - mp.mpf(9) / 8
    report(2, ok1 and ok2 and ok3 and ok4, f"t0={t0} k:{below}/{at}/{above} lim a,b={la},{lb} S_inf~{float(Slim):.8f} (인쇄식 {float(typo):.4f})")


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
