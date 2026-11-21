#!/usr/bin/env python3
"""SKKU-G25C (성균관대 2025학년도 수시 논술우수 수리형 3교시) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import sys
from fractions import Fraction

import numpy as np
import sympy as sp

PREFIX = "SKKU-G25C"
WIDTH = 2
RESULTS = []
x, th = sp.symbols("x theta", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SKKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def md_set(no, sub, loc=None):
    v = md_answer(no, sub).strip().strip("{}")
    return [sp.sympify(p, locals=loc or {}) for p in v.split(",")]


def md_parts(no, sub, loc=None):
    return {k.strip(): sp.sympify(v, locals=loc or {}) for k, v in
            (seg.split("=", 1) for seg in md_answer(no, sub).split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def geom(tv):
    """좌표로 직접: D=3(-cosθ, sinθ), 원 중심 (0,r) (C에서 x축에 접함), A=직선 BD 와 원의 다른 교점."""
    c, s = sp.cos(tv), sp.sin(tv)
    D = sp.Matrix([-3 * c, 3 * s]); B = sp.Matrix([-4, 0])
    r = sp.symbols("r", positive=True)
    rr = sp.solve(sp.Eq(D[0]**2 + (D[1] - r)**2, r**2), r)[0]
    u = sp.symbols("u", real=True)
    P = B + u * (D - B)                                   # u=1 → D
    us = sp.solve(sp.Eq(P[0]**2 + (P[1] - rr)**2, rr**2), u)
    A = [B + v * (D - B) for v in us]
    A = [a for a in A if sp.simplify((a - D).norm()) != 0] or [D]
    A = A[0]
    return sp.simplify(rr), sp.simplify((A - D).norm()), sp.simplify(A.norm()), sp.simplify((D - B).norm())


def geomf(tv):
    """geom 의 빠른 수치판: 원 x^2+y^2-2ry=0 과 직선 B+u(D-B) 의 두 근 중 u≠1 (근의 곱=16/|v|^2)."""
    D = np.array([-3 * np.cos(tv), 3 * np.sin(tv)]); B = np.array([-4.0, 0.0]); v = D - B
    r = (D @ D) / (2 * D[1])
    a2, a1, a0 = v @ v, 2 * (B @ v) - 2 * r * v[1], B @ B - 2 * r * B[1]
    us = np.roots([a2, a1, a0]).real
    u2 = us[np.argmax(np.abs(us - 1))] if abs(us[0] - us[1]) > 1e-15 else 1.0
    A = B + u2 * v
    return r, np.linalg.norm(A - D), np.linalg.norm(A), np.linalg.norm(D - B)


def p01():
    R1 = sp.sympify(md_answer(1, "1"), locals={"theta": th})
    ok1 = all(abs(float(geom(sp.Rational(k, 10))[0] - R1.subs(th, sp.Rational(k, 10)))) < 1e-12 for k in (3, 7, 12, 20, 28))
    # (ii) α: A=D(접함) ⇔ BD 가 원에 접함 ⇔ 이차식 중근 — 수치로 f(θ)=0 인 θ 탐색
    fnum = lambda v: geomf(v)[1]
    ts = np.linspace(0.05, 3.09, 400)
    vals = [fnum(v) for v in ts]
    i0 = int(np.argmin(vals))
    lo, hi = ts[max(i0 - 2, 0)], ts[min(i0 + 2, 399)]
    for _ in range(200):                                   # 삼분 탐색으로 f 의 최솟점(=0) α
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if fnum(m1) < fnum(m2):
            hi = m2
        else:
            lo = m1
    al = (lo + hi) / 2
    P2 = md_parts(1, "2", {"alpha": sp.Symbol("alpha")})
    cos_md = [v for k, v in P2.items() if "cos" in k][0]; r_md = P2["r"]
    ok2 = abs(np.cos(al) - float(cos_md)) < 1e-6 and abs(float(geom(sp.acos(cos_md))[0]) - float(r_md)) < 1e-12 \
        and float(geom(sp.acos(cos_md))[1]) < 1e-12
    P3 = md_parts(1, "3")
    ok3 = abs(float(geom(sp.pi / 3)[1]) - float(P3["f(pi/3)"])) < 1e-12 and abs(float(geom(sp.pi / 2)[1]) - float(P3["f(pi/2)"])) < 1e-12
    # (iv) f(θ)=AC 인 θ: 격자 부호변화 + 근 정밀화
    g = lambda v: (lambda q: q[1] - q[2])(geomf(v))
    grid = np.linspace(0.02, 3.12, 600)
    gv = [g(v) for v in grid]
    roots = []
    for i in range(len(grid) - 1):
        if gv[i] * gv[i + 1] < 0:
            lo, hi = grid[i], grid[i + 1]
            for _ in range(80):                            # 이분법
                mid = (lo + hi) / 2
                if g(lo) * g(mid) <= 0:
                    hi = mid
                else:
                    lo = mid
            roots.append(np.cos((lo + hi) / 2))
    md4 = sorted(float(v) for v in md_set(1, "4"))
    ok4 = len(roots) == len(md4) and all(abs(a - b) < 1e-8 for a, b in zip(sorted(roots), md4))
    report(1, ok1 and ok2 and ok3 and ok4, f"cos(alpha)={np.cos(al):.9f} f(pi/3)={geom(sp.pi / 3)[1]} f(pi/2)={geom(sp.pi / 2)[1]} cos-roots={sorted(roots)}")


def p02():
    par = 1 - x**2; L = x / 2 + sp.Rational(1, 2)
    xs = sp.solve(sp.Eq(par, L), x)
    SP = sp.integrate(par - L, (x, min(xs), max(xs)))
    SQ = sp.integrate(par, (x, -1, 1)) - SP
    ok1 = sp.simplify(SP / SQ - sp.sympify(md_answer(2, "1"))) == 0
    # (ii) 넓이 최대: C=(q,0), 높이 1-q^2, A 의 x 좌표는 L 위 같은 높이
    q = sp.symbols("q", real=True)
    pA = sp.solve(sp.Eq(L.subs(x, sp.Symbol("p")), par.subs(x, q)), sp.Symbol("p"))[0]
    area = sp.expand((q - pA) * par.subs(x, q))
    crit = [c for c in sp.Poly(sp.diff(area, q), q).nroots(n=40) if abs(sp.im(c)) < 1e-30 and 0.5 < sp.re(c) < 1]
    qs = max(crit, key=lambda c: area.subs(q, sp.re(c)))
    qq = np.linspace(0.5, 1, 200001); av = np.polyval([float(c) for c in sp.Poly(area, q).all_coeffs()], qq)
    w = sp.re(qs) - sp.Rational(1, 2)
    ok2 = abs(qq[np.argmax(av)] - float(sp.re(qs))) < 1e-5 and abs(float(8 * w**3 + 15 * w**2 + 3 * w) - float(sp.sympify(md_answer(2, "2")))) < 1e-25
    # (iii) 직사각형: B,C 는 M 위, |BC|=√17/4, A 는 L 위, D 는 곡선 위 (방향벡터 사용, 독립 연립)
    a, s = sp.symbols("a s", real=True)
    M = lambda t: (t + 1) / 4
    B = sp.Matrix([a, M(a)]); d = sp.Matrix([4, 1]) / sp.sqrt(17); nrm = sp.Matrix([-1, 4]) / sp.sqrt(17)
    C = B + sp.sqrt(17) / 4 * d
    A = B + s * nrm; D = C + s * nrm
    sol = sp.solve([A[1] - L.subs(x, A[0]), D[1] - par.subs(x, D[0])], [a, s], dict=True)
    good = []
    for so in sol:
        av_, sv = so[a], so[s]
        if sv.is_real and sv > 0 and A[0].subs(so) > -1:
            good.append(sp.simplify(B.subs(so)))
    Bmd = sp.Matrix(list(sp.sympify(md_answer(2, "3"))))
    ok3 = len(good) == 1 and sp.simplify(good[0] - Bmd) == sp.zeros(2, 1)
    report(2, ok1 and ok2 and ok3, f"ratio={SP / SQ} q*={sp.N(sp.re(qs), 12)} 8w^3+15w^2+3w={sp.N(8 * w**3 + 15 * w**2 + 3 * w, 15)} B={list(good[0]) if good else None}")


def bseq(r, N=150):
    """b_n 을 정확한 유리수 비교로 계산: b=M ⇔ (M-1/2)^3 ≤ X < (M+1/2)^3, X=n/4+(-1)^n a_n."""
    out = []
    for n in range(1, N + 1):
        X = Fraction(n, 4) + (-1)**n * r * Fraction(1, 2**(n - 1))
        M = round(float(np.cbrt(float(X))))
        for cand in range(M - 2, M + 3):
            lo, hi = (Fraction(2 * cand - 1, 2))**3, (Fraction(2 * cand + 1, 2))**3
            if lo <= X < hi:
                out.append(cand)
                break
    return out


def p03():
    b = bseq(Fraction(4, 3))
    ok1 = b[0] == int(md_answer(3, "1"))
    ok2 = sum(b) == int(md_answer(3, "2"))
    sums = set()
    for k in range(1, 4000):                      # r ∈ (1,2) ∪ (-2,-1) 조밀 격자(유리수)
        for sgn in (1, -1):
            r = sgn * (1 + Fraction(k, 4000))
            sums.add(sum(bseq(r)))
    ok3 = sums == {int(v) for v in md_set(3, "3")}
    report(3, ok1 and ok2 and ok3, f"b1={b[0]} sum(4/3)={sum(b)} possible sums={sorted(sums)}")


if __name__ == "__main__":
    for name in sorted(n for n in dir() if n.startswith("p") and n[1:].isdigit()):
        globals()[name]()
    n_ok = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {n_ok}/{len(RESULTS)} PASS")
    sys.exit(0 if n_ok == len(RESULTS) else 1)
