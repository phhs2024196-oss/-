#!/usr/bin/env python3
"""숭실대 SSU-G13-01 — sympy 검산. 규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수는 자동 실행된다. 소문항 정답은 yaml 의 subparts[].answer.value 에서 읽는다."""
import re
import sys

import sympy as sp

PREFIX = "SSU-G13"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answers(seq):
    """소문항 id → answer.value(문자열). 증명 소문항은 None."""
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(f"items/SSU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    return {str(s["id"]): s["answer"]["value"] for s in d["subparts"]}


def p01():
    """A: S(x) 를 적분으로 재구성·좌우 미분계수·최솟값, B: 역함수 수치 미분, 시간 적분."""
    import mpmath as mp
    ans = load_answers(1)
    xp = sp.Symbol("xp", positive=True)
    ft = sp.Piecewise((6*t, t <= 2), (3*t**2, True))
    S1 = (20 + sp.integrate(6*t, (t, 0, xp))) / xp
    S2 = (20 + sp.integrate(6*t, (t, 0, 2)) + sp.integrate(3*t**2, (t, 2, xp))) / xp
    Smd = sp.sympify(ans["A1"], locals={"x": xp})
    okA1 = all(sp.simplify(Smd.subs(xp, v) - (S1 if v <= 2 else S2).subs(xp, v)) == 0 for v in [sp.Rational(k, 4) for k in range(1, 40)])
    h = sp.Symbol("h", positive=True)
    L = sp.limit((S1.subs(xp, 2 - h) - S1.subs(xp, 2)) / (-h), h, 0)
    R = sp.limit((S2.subs(xp, 2 + h) - S2.subs(xp, 2)) / h, h, 0)
    okA2 = L == R == -2                                   # [증명-수치확인] 좌우 미분계수
    Sf = lambda v: float((S1 if v <= 2 else S2).subs(xp, v))
    grid = [0.05 + k*0.0005 for k in range(20000)]
    xbest = min(grid, key=Sf)
    okA3 = abs(xbest - float(sp.sympify(ans["A3"]))) < 1e-3
    # B1: f(x)=∫_2^x sqrt(3+t^2)dt, 역함수를 수치로 구해 0 근처 차분
    f = lambda X: mp.quad(lambda u: mp.sqrt(3 + u*u), [2, X])
    g = lambda Y: mp.findroot(lambda X: f(X) - Y, 2)
    gp = (g(mp.mpf("1e-8")) - g(mp.mpf("-1e-8"))) / mp.mpf("2e-8")
    okB1 = abs(gp - float(sp.sympify(ans["B1"]))) < 1e-7
    # B2: ds/dt = s^2-1 를 수치 ODE 로 풀어 s=3 도달 시각
    sol = mp.odefun(lambda tt, s: s*s - 1, 0, 2)
    T = mp.findroot(lambda tt: sol(tt) - 3, 0.2)
    okB2 = abs(T - float(sp.sympify(ans["B2"]))) < 1e-10
    report(1, okA1 and okA2 and okA3 and okB1 and okB2,
           f"A1 적분 재구성 {okA1} | A2 좌={L} 우={R} [증명-수치확인] | A3 격자 최소 x={xbest:.4f} md={ans['A3']} | B1 g'(0)≈{mp.nstr(gp,10)} md={ans['B1']} | B2 ODE T={mp.nstr(T,12)} md={ans['B2']}")


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
