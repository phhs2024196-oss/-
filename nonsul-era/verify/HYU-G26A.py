#!/usr/bin/env python3
"""HYU-G26A (한양대 2026학년도 논술 자연계열 오전) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건에서 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "HYU-G26A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HYU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # 1. 좌표: A=(0,0), B=(5,0), C=(3cosA, 3sinA). 넓이 조건으로 sinA, 예각이므로 cosA>0
    sA = sp.Rational(2) * sp.sqrt(26) / (3 * 5)
    cA = sp.sqrt(1 - sA**2)
    B, C = sp.Matrix([5, 0]), sp.Matrix([3 * cA, 3 * sA])
    area = sp.Rational(1, 2) * abs(B[0] * C[1] - B[1] * C[0])
    a = sp.sqrt((B - C).dot(B - C))
    r = sp.nsimplify(sp.simplify(2 * area / (a + 3 + 5)))
    ok1 = sp.simplify(area - sp.sqrt(26)) == 0 and cA > 0 and \
        sp.simplify(r - sp.sympify(md_answer(1, "1"))) == 0
    # 2. 전수 열거: f:{a,b,c}→{1..n}, 조건 f(a) 홀수 또는 f(b) 짝수
    an = {}
    for m in range(1, 13):
        an[m] = sum(1 for fa, fb, fc in itertools.product(range(1, m + 1), repeat=3) if fa % 2 == 1 or fb % 2 == 0)
    S = lambda m: sum(an[k] for k in range(1, m + 1))
    v2 = sp.Rational(S(9) - S(1), S(12))
    ok2 = v2 == sp.sympify(md_answer(1, "2"))
    # 3. n=1..2025 에 대해 정적분을 직접 계산(구간을 근에서 나눠 정확 적분)해 4 미만인 n 개수
    cache = {}

    def integral(c):
        if c not in cache:
            p = (x + 2) * (x - 1)**2 - c
            pts = sorted({sp.Rational(1, 2), sp.Integer(2)} | {rt for rt in sp.real_roots(sp.Poly(p, x))
                                                                 if sp.Rational(1, 2) < rt < 2})
            cache[c] = sum(abs(sp.integrate(p, (x, pts[i], pts[i + 1]))) for i in range(len(pts) - 1))
        return cache[c]
    cnt = 0
    for m in range(1, 2026):
        c = sp.nsimplify(4 * sp.sin((2 * m - 1) * sp.pi / 6))
        if integral(c) < 4:
            cnt += 1
    ok3 = cnt == int(md_answer(1, "3"))
    report(1, ok1 and ok2 and ok3,
           f"r={r} (S9-S1)/S12={v2} count={cnt} I(c)={ {k: str(sp.nsimplify(v)) for k, v in cache.items()} }")


def p02():
    # 1. F(x)=∫_0^x f. 0≤x≤2π 에서 f=-(e^x-cos x)^2026 sin x, x>2π 에서 x-2π, f 는 기함수 → F 는 우함수
    mp.mp.dps = 40
    f = lambda u: -(mp.e**u - mp.cos(u))**2026 * mp.sin(u)
    # 부호 확인(격자): (0,π) 에서 f<0, (π,2π) 에서 f>0, (2π,∞) 에서 f=x-2π>0
    sgn_ok = all(f(mp.pi * k / 200) < 0 for k in range(1, 200)) and \
        all(f(mp.pi + mp.pi * k / 200) > 0 for k in range(1, 200))
    Fpi = mp.quad(f, [0, mp.pi / 2, mp.pi])
    F2pi = Fpi + mp.quad(f, [mp.pi, 1.25 * mp.pi, 1.5 * mp.pi, 1.75 * mp.pi, 2 * mp.pi])
    # 단조 구간(열린구간)과 양 끝 F 값. F(∞)=∞. 음수 쪽은 우함수 대칭.
    pieces = [(mp.mpf(0), Fpi), (Fpi, F2pi), (F2pi, mp.inf)]
    zeros = {0: mp.mpf(0), 1: Fpi, 2: F2pi}            # f 의 영점 0, ±π, ±2π 에서의 F 값(±는 대칭)
    # h(x)=F(x)-F(a), h(a)=0. f(a)≠0 이면 |h| 는 a 에서 미분 불가 → a 는 f 의 영점이어야 함.
    # a 가 영점이면 y=F(a) 와 y=F(x) 의 교점이 모두 f 의 영점이어야 함: 열린 단조구간 내부 교점이 없어야 함.
    good = []
    for key, L in zeros.items():
        inside = any(min(u, v) < L < max(u, v) for u, v in pieces)
        if not inside:
            good.append(key)
    got = sorted(sp.nsimplify(s * k * sp.pi) for k in good for s in ((1, -1) if k else (1,)))
    md = sorted(sp.sympify(md_answer(2, "1").replace("{", "[").replace("}", "]")))
    ok1 = sgn_ok and F2pi > 0 > Fpi and got == md
    # 2. 두 평행 현 사이(원판 내부) 넓이 = seg(c(k)) - seg(c0), seg(c)=arccos c - c√(1-c²): 법선방향 거리 c 이상인 부분
    a, k = sp.symbols("a k", positive=True)
    seg = lambda c: sp.acos(c) - c * sp.sqrt(1 - c**2)
    c0 = a / sp.sqrt(1 + a**2)
    ck = (a - k) / sp.sqrt(1 + a**2)
    Sk = seg(ck) - seg(c0)
    lim = sp.simplify(sp.diff(Sk, k).subs(k, 0))          # S(0)=0 이므로 S(k)/k → S'(0)
    md2 = sp.sympify(md_answer(2, "2"), locals={"a": a})
    ok2 = sp.simplify(lim - md2) == 0
    for av in (sp.Rational(3, 2), sp.Integer(2), sp.Rational(37, 10)):                                 # 작은 k 에서 수치 확인
        Sn = sp.lambdify(k, Sk.subs(a, av), "mpmath")
        ok2 = ok2 and abs(Sn(mp.mpf("1e-12")) / mp.mpf("1e-12") - float(md2.subs(a, av))) < 1e-9
    # 3. f(x)=x^3+p x^2+q x+r, g'(1)=±1 경우를 모두 풀어 조건 확인
    p, q, r, b, g1, g2 = sp.symbols("p q r b g1 g2", real=True)
    F = x**3 + p * x**2 + q * x + r
    Fp, Fpp = sp.diff(F, x), sp.diff(F, x, 2)
    cands = []
    for gp1 in (1, -1):                                       # 속력 √(1+g'(1)^2)=√2
        gpp1 = 0                                              # 속력 최대 ⇒ g'(1)g''(1)=0, g'(1)≠0
        # t=1/4: g=0, g'=0(g≥0 의 최솟점), g''=1(가속도 (0,1))
        eqs = [F.subs(x, 0) - 0,                                         # h(1/4)=f(0)=0
               Fpp.subs(x, 0) * 0**2 + Fp.subs(x, 0) * 1 - 4,             # h''(1/4)=4
               Fp.subs(x, b) * gp1 - sp.Rational(9, 4),                   # h'(1)=9/4
               Fpp.subs(x, b) * gp1**2 + Fp.subs(x, b) * gpp1 + 2]        # h''(1)=-2
        for sol in sp.solve(eqs, [p, q, r, b], dict=True):
            if sol[b] < 0:
                continue
            fp = Fp.subs(sol)
            vx = sp.solve(sp.diff(fp, x), x)[0]                 # f' 은 아래로 볼록한 이차식
            fmin = fp.subs(x, vx) if vx > 0 else fp.subs(x, 0)    # [0,∞) 에서 f' 의 최솟값
            if sp.N(fmin) > 0:
                cands.append(F.subs(sol).subs(x, 3))
    ok3 = len(cands) == 1 and sp.simplify(cands[0] - sp.sympify(md_answer(2, "3"))) == 0
    report(2, ok1 and ok2 and ok3,
           f"F(pi)={mp.nstr(Fpi, 5)} F(2pi)={mp.nstr(F2pi, 5)} a={got} lim={lim} f(3)={cands}")


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
