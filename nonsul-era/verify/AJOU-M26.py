#!/usr/bin/env python3
"""AJOU-M26 (아주대 2026학년도 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import math
import re
import sys

import sympy as sp

PREFIX = "AJOU-M26"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/AJOU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def md_pairs(no: int, sub: str) -> dict:
    """'k=3; limit=1/2' 또는 'radius: ..; circle: ..' 꼴을 사전으로."""
    out = {}
    for seg in md_answer(no, sub).split(";"):
        k, v = re.split(r"[:=]", seg, maxsplit=1)
        out[k.strip()] = v.strip()
    return out


def md_set(no: int, sub: str) -> list:
    return [sp.sympify(s) for s in md_answer(no, sub).strip().strip("{}").split(",")]


def same(a, b, tol=1e-10) -> bool:
    return abs(complex(sp.N(a, 30)) - complex(sp.N(b, 30))) < tol


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def _mock_P(tv):
    """시각 t 에서 A(cos t, sin t), B(α,0), P(C2 위, 제1사분면)."""
    import mpmath as mp
    A = (mp.cos(tv), mp.sin(tv))
    al = A[0] + mp.sqrt(9 - A[1]**2)                       # |AB|=3, B 는 A 의 오른쪽 x축 위
    # P: (x-4)^2+y^2=4 와 (x-al)^2+y^2=4 의 교점 중 y>0
    px = (al + 4) / 2
    py = mp.sqrt(4 - (px - 4)**2)
    return A, al, (px, py)


def p01():
    import mpmath as mp
    mp.mp.dps = 40
    th = sp.symbols("theta", real=True)
    # (1-1) S1, S2 를 좌표로: B 원점, A(-3,0), P(2cos(π-θ)... ) → P(-2cosθ... ) 로 두지 않고 직접 좌표
    A = sp.Matrix([-3, 0]); P = sp.Matrix([-2 * sp.cos(th), 2 * sp.sin(th)])   # ∠ABP=θ (BA 방향이 -x)
    H = sp.Matrix([P[0], 0])
    S1 = sp.Abs((A[0] * P[1] - A[1] * P[0])) / 2
    S2 = sp.Abs((H[0] - A[0]) * P[1]) / 2
    lim = sp.limit(sp.simplify(S1 / S2).subs(sp.Abs(sp.sin(th)), sp.sin(th)), th, sp.pi, "-")
    ok1 = same(lim, sp.sympify(md_answer(1, "1-1")))
    # (1-2) R = AP / (2 sin θ), [π/6,5π/6] 최소
    R = sp.sqrt((A - P).dot(A - P)) / (2 * sp.sin(th))
    dR = sp.diff(R, th)
    cands = [sp.pi / 6, 5 * sp.pi / 6]
    for g0 in (0.5, 0.8, 1.2, 2.0, 2.5):
        try:
            r = sp.nsolve(dR, th, g0, prec=30)
            if math.pi / 6 <= float(r) <= 5 * math.pi / 6:
                cands.append(r)
        except Exception:  # noqa: BLE001
            pass
    best = min(cands, key=lambda c: float(R.subs(th, c)))
    ok2 = same(sp.sin(best), sp.sympify(md_answer(1, "1-2")), 1e-20)
    # (2-1) 직선 AB 가 처음으로 C1 에 접하는 t0: OA ⊥ AB
    def perp(tv):
        A_, al, _ = _mock_P(tv)
        return A_[0] * (al - A_[0]) + A_[1] * (0 - A_[1])
    t0 = mp.findroot(perp, 1.2)
    first = all(perp(v) > 0 for v in mp.linspace(1e-6, t0 - mp.mpf("1e-6"), 400))
    ok3 = first and abs(mp.cos(t0) - mp.mpf(str(sp.N(sp.sympify(md_answer(1, "2-1")), 40)))) < mp.mpf("1e-25")
    # (2-2) f(t)=|OP|^2, f'(t0) 수치 미분
    f = lambda tv: _mock_P(tv)[2][0]**2 + _mock_P(tv)[2][1]**2
    fp = mp.diff(f, t0)
    ok4 = abs(fp - mp.mpf(str(sp.N(sp.sympify(md_answer(1, "2-2")), 40)))) < mp.mpf("1e-20")
    # (2-3) 자취: C2 위 각도 범위 × 반지름 2
    angs = []
    for v in mp.linspace(0, 2 * mp.pi, 4001):
        _, _, (px, py) = _mock_P(v)
        assert abs((px - 4)**2 + py**2 - 4) < mp.mpf("1e-30")
        angs.append(mp.atan2(py, px - 4))
    arc = 2 * (max(angs) - min(angs))
    ok5 = abs(arc - mp.mpf(str(sp.N(sp.sympify(md_answer(1, "2-3")), 40)))) < mp.mpf("1e-6")
    report(1, ok1 and ok2 and ok3 and ok4 and ok5,
           f"lim={lim} sinθ*={sp.N(sp.sin(best), 12)} cos t0={mp.nstr(mp.cos(t0), 12)} f'(t0)={mp.nstr(fp, 12)} "
           f"arc≈{mp.nstr(arc, 10)} [{ok1},{ok2},{ok3},{ok4},{ok5}]")


def p02():
    import mpmath as mp
    from fractions import Fraction
    mp.mp.dps = 30
    # (1) 제시문 정의 그대로: c=2, g(x)=x²/2, a_n = g(0) + Σ (c/n) g'(kc/n)
    def a_n(nv):
        return Fraction(0) + sum(Fraction(2, nv) * Fraction(2 * k, nv) for k in range(1, nv + 1))
    ok1 = sp.Rational(a_n(7).numerator, a_n(7).denominator) == sp.sympify(md_answer(2, "1-1"))
    k_, N = sp.symbols("k N", positive=True, integer=True)
    an_sym = sp.summation(sp.Rational(2) / N * 2 * k_ / N, (k_, 1, N))
    ok2 = sp.limit(an_sym, N, sp.oo) == sp.sympify(md_answer(2, "1-2"))
    # (3) Σ_{n=2}^{2026} a_n/(n²-1) 정확 계산
    tot = sum(a_n(nv) / (nv * nv - 1) for nv in range(2, 2027))
    tot = sp.Rational(tot.numerator, tot.denominator)
    md3 = sp.sympify(md_answer(2, "1-3"))
    ok3 = tot == md3
    # (2-1) 수치 적분
    f = lambda v: mp.cos(v) / (1 + mp.sec(v) + mp.tan(v))
    I = mp.quad(f, [-mp.pi / 3, 0, mp.pi / 3])
    ok4 = abs(I - mp.mpf(str(sp.N(sp.sympify(md_answer(2, "2-1")), 30)))) < mp.mpf("1e-20")
    # (2-2) g(t) 를 큰 t 에서 수치 계산
    g = lambda tv: mp.quad(lambda v: f(v / mp.power(3, tv - 1)), [-mp.pi / 3, 0, mp.pi / 3])
    target = mp.mpf(str(sp.N(sp.sympify(md_answer(2, "2-2")), 30)))
    ok5 = abs(g(15) - target) < mp.mpf("1e-12") and abs(g(25) - target) < mp.mpf("1e-20")
    msg3 = "" if ok3 else f" ※(1-3) 독립 계산 {tot} ≠ YAML(공식 채점기준) {md3} — 공식답 산술 오류 의심, 원문 보존"
    report(2, ok1 and ok2 and ok3 and ok4 and ok5,
           f"a7={a_n(7)} lim a_n={sp.limit(an_sym, N, sp.oo)} Σ={tot} ∫={mp.nstr(I, 12)} g(25)={mp.nstr(g(25), 12)} "
           f"[{ok1},{ok2},{ok3},{ok4},{ok5}]{msg3}")


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
