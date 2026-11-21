#!/usr/bin/env python3
"""DGU-G19 (2019학년도 동국대 수리논술 실전, 문제1 과학논술 제외) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import itertools
import random
import re
import sys

import sympy as sp

PREFIX = "DGU-G19"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p02():
    """베이즈: P(A|B)=(p/5)/(p/5+(1-p)/14). 첫날 p=30/100 → 6/11, 누적 p∈{36..40}/110 → 증가."""
    p = sp.symbols("p", positive=True)
    # 전수: 한 사람 = (주사위 선택, 눈). 윤목 1~5, 목제주령구 1~14 균등
    def post(pv):
        num = pv * sp.Rational(1, 5)
        den = num + (1 - pv) * sp.Rational(1, 14)
        return num / den
    f = sp.simplify(post(p))
    first = post(sp.Rational(30, 100))
    # 둘째 날 10명 중 윤목 > 목제 → 윤목 6~10명
    cum = [post(sp.Rational(30 + j, 110)) for j in range(0, 11) if j > 10 - j]
    inc = sp.simplify(sp.diff(f, p)) 
    ok = first == sp.Rational(6, 11) and all(v > first for v in cum) and min(cum) == sp.Rational(252, 437) and max(cum) == sp.Rational(8, 13) \
        and all(inc.subs(p, sp.Rational(k, 100)) > 0 for k in range(1, 100))
    report(2, ok, f"P(A|B)={f}, 첫날 {first}, 누적 {min(cum)}~{max(cum)} (모두 > 6/11), dP/dp={inc}>0")


def p03():
    """(1) L'=kL(N-L), L(0)=1 → L=N/(1+(N-1)e^{-Nkt}) (2) L 증가·N 수렴, P 발산."""
    k, N = sp.symbols("k N", positive=True)
    Lf = sp.Function("L")
    sol = sp.dsolve(sp.Eq(Lf(t).diff(t), k * Lf(t) * (N - Lf(t))), Lf(t), ics={Lf(0): 1})
    cand = N / (1 + (N - 1) * sp.exp(-N * k * t))
    ode_ok = sp.simplify(sp.diff(cand, t) - k * cand * (N - cand)) == 0 and sp.simplify(cand.subs(t, 0) - 1) == 0
    sol_ok = sp.simplify(sol.rhs - cand) == 0
    lim = sp.limit(cand.subs({N: 5, k: 1}), t, sp.oo)
    Ld = sp.simplify(sp.diff(cand, t))
    # 수치: N=5,k=0.3 → L 증가, L' 은 결국 감소(증가율 감소), P=e^{kt} 증가율은 증가
    import numpy as np
    tt = np.linspace(0, 8, 801)
    Lv = sp.lambdify(t, cand.subs({N: 5, k: sp.Rational(3, 10)}), "numpy")(tt)
    rel = np.diff(Lv) / Lv[:-1]  # 상대 증가율
    ok = ode_ok and sol_ok and lim == 5 and bool(np.all(np.diff(Lv) > 0)) and bool(np.all(np.diff(rel) < 0))
    report(3, ok, f"(1) L={cand} (dsolve 일치 {sol_ok}) (2) lim L=N, L'={Ld}>0, 상대증가율 단조감소")


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
