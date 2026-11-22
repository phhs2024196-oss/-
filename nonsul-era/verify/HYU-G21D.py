#!/usr/bin/env python3
"""HYU-G21D (한양대 2021학년도 논술 의예과, 수학 문제 2만) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "HYU-G21D"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HYU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(str(s["answer"]["value"]) for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p02():
    import mpmath as mp
    mp.mp.dps = 30
    # (1) 좌표기하로 직선 y=0 에 접하는 원 (중심 (p, r)). 두 원 (p1,r1),(p2,r2) 사이 틈에 넣을 수 있는
    #     가장 큰 원은 두 원과 외접하고 직선에 접하는 원 → 연립방정식을 수치로 풀어 반지름을 구한다.
    def gap_circle(c1, c2):
        (p1, r1), (p2, r2) = c1, c2
        f = lambda p, r: [(p - p1)**2 + (r - r1)**2 - (r + r1)**2, (p - p2)**2 + (r - r2)**2 - (r + r2)**2]
        r0 = (r1 * r2) / (mp.sqrt(r1) + mp.sqrt(r2))**2 * 1.01
        sol = mp.findroot(f, (p1 + (p2 - p1) * mp.sqrt(r1) / (mp.sqrt(r1) + mp.sqrt(r2)), r0))
        return (sol[0], sol[1])
    circles = [(mp.mpf(-1), mp.mpf(1)), (mp.mpf(1), mp.mpf(1))]      # C, D (x 좌표 순)
    chosen = []
    while len(chosen) < 12:
        cands = [(gap_circle(circles[i], circles[i + 1]), i) for i in range(len(circles) - 1)]
        (c, i) = max(cands, key=lambda z: z[0][1])                    # 가장 큰 원
        circles.insert(i + 1, c)
        chosen.append(c)
    r12 = chosen[11][1]
    ok1 = abs(r12 - mp.mpf(sp.Rational(sp.sympify(md_answer(2, "1"))))) < 1e-20
    # (2) f 를 미분방정식에서 구함: f=2-g^2 이면 g'=(2-g^2)/2, g(0)=0 → g=√2 tanh(x/√2)
    X = sp.symbols("X", positive=True)
    G = sp.symbols("G")
    # 변수분리: ∫ 2/(2-G^2) dG = X, G(0)=0 → √2·artanh(G/√2) = X
    gs = sp.sqrt(2) * sp.tanh(X / sp.sqrt(2))
    okg = sp.simplify(sp.diff(gs, X) - (2 - gs**2) / 2) == 0 and gs.subs(X, 0) == 0 and \
        abs(mp.quad(lambda q: 2 / (2 - q**2), [0, float(gs.subs(X, 0.7))]) - mp.mpf("0.7")) < 1e-12
    f = sp.simplify(2 - gs**2)
    okode = all(abs(float((sp.diff(f, X) + f * sp.sqrt(2 - f)).subs(X, v))) < 1e-12 for v in (0.3, 1, 4)) and \
        sp.limit(f, X, 0, "+") == 2 and all(0 < float(f.subs(X, v)) < 2 for v in (0.01, 1, 10))
    an = lambda N: mp.quad(sp.lambdify(X, f, "mpmath"), [mp.mpf(1) / N, 1, 10, 100, 2021 * N])
    vals = [an(N) for N in (10, 10**6)]   # a_N 은 2√2 로 단조 수렴(오차 약 2/N)
    ok2 = okg and okode and abs(vals[-1] - mp.mpf(float(sp.sympify(md_answer(2, "2"))))) < 1e-5 and \
        abs(sp.limit(sp.integrate(f, (X, 0, sp.Symbol("M", positive=True))), sp.Symbol("M", positive=True), sp.oo)
            - sp.sympify(md_answer(2, "2"))) == 0
    # (3) b_n 을 정확히 적분해 n b_n / 2^n 의 극한
    N = sp.symbols("N", integer=True, positive=True)
    s = sp.symbols("s", positive=True)
    bN = sp.integrate(s * (s + 1)**N, (s, sp.exp(-1), 1))          # t=e^{-x} 치환식
    okb = all(abs(float(bN.subs(N, k)) - float(mp.quad(lambda xx: mp.e**(-(k + 2) * xx) * (1 + mp.e**xx)**k, [0, 1])))
              < 1e-9 * float(bN.subs(N, k)) for k in (1, 5, 12))
    L3 = sp.limit(sp.simplify(N * bN / 2**N), N, sp.oo)
    ok3 = okb and sp.simplify(L3 - sp.sympify(md_answer(2, "3"))) == 0
    report(2, ok1 and ok2 and ok3, f"r12={mp.nstr(r12, 12)} a_1e6={mp.nstr(vals[-1], 10)} lim3={L3}")


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
