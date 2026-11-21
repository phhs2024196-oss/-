#!/usr/bin/env python3
"""SJU-G15B (세종대 2015학년도 수시 논술[B] 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SJU-G15B"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    return dict(seg.split("=", 1) for seg in s.replace(" ", "").split(";"))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # 상자: 높이 = 축 방향 길이 3r, 바깥지름 2(d+r)
    r, d = sp.symbols("r d", positive=True)
    sol = sp.solve([3 * r - 18, 2 * (d + r) - 62], [r, d], dict=True)[0]
    P1 = parts(md_answer(1, "1"))
    ok1 = sol[r] == sp.sympify(P1["center_distance"]) and sol[d] == sp.sympify(P1["axis_distance"])
    # (2) 구체적 f 로 부피 비교
    x, a, b, c = sp.symbols("x a b c", positive=True)
    f = sp.Function("f")
    ok2 = True
    for fx, cv, av, bv in ((sp.sin(x)**2, 3, 0, 2), (x / 4, 2, 1, 3)):
        Vdirect = sp.pi * sp.integrate((cv + fx)**2 - (cv - fx)**2, (x, av, bv))
        expr = sp.sympify(md_answer(1, "2"), locals={"f": f, "x": x, "a": a, "b": b, "c": c})
        Vmd = expr.subs({a: av, b: bv, c: cv}).replace(f, sp.Lambda(x, fx)).doit()
        ok2 &= sp.simplify(Vdirect - Vmd) == 0
    # (3) 겹친 두 원 단면을 축 방향 x 로 적분: f(x)=위쪽 절반 두께, 수치 적분 후 4cπ∫f
    import mpmath as mp
    R = int(sol[r]); C = int(sol[d])
    def half(xx):                       # 원 중심 x=0, x=R, 반지름 R 의 합집합의 y-반두께
        h1 = mp.sqrt(max(R**2 - xx**2, 0)); h2 = mp.sqrt(max(R**2 - (xx - R)**2, 0))
        return max(h1, h2)
    I = mp.quad(half, [-R, R / 2, 2 * R])
    V = 4 * C * mp.pi * I
    ok3 = abs(V - mp.mpf(float(sp.N(sp.sympify(md_answer(1, "3")), 30)))) < 1e-8
    report(1, ok1 and ok2 and ok3, f"r={sol[r]} d={sol[d]} ∫f={mp.nstr(I, 12)} V={mp.nstr(V, 14)}")


def p02():
    import mpmath as mp
    import random
    f = lambda t: mp.sin(t) / mp.sqrt(3)
    # (2-1) h(x)=f(x)-x/√3-c 의 근 개수: 단조 감소 확인(h' ≤ 0) + 부호 변화 1회
    ok1 = all(mp.cos(t) - 1 <= 0 for t in [random.uniform(-50, 50) for _ in range(1000)])
    for cv in (-3.7, 0.0, 2.2):
        xs = [i / 50 for i in range(-1000, 1001)]
        vals = [f(t) - t / mp.sqrt(3) - cv for t in xs]
        sc = sum(1 for i in range(len(vals) - 1) if vals[i] * vals[i + 1] < 0 or vals[i] == 0)
        ok1 &= sc == 1
    # (2-2) 무작위 할선 기울기 상한 < 1/√3
    sl = []
    for _ in range(20000):
        x1 = random.uniform(-20, 20); x2 = x1 + random.uniform(1e-4, 20)
        sl.append((f(x2) - f(x1)) / (x2 - x1))
    ok2 = max(sl) < 1 / mp.sqrt(3)
    # (2-3) m 몇 개에 대해 (0,π) 에서 L(0,x)=m 근 존재
    ok3 = True
    for m in (0.01, 0.2, 0.4, 0.57):
        g = lambda t: f(t) / t - m
        ok3 &= g(mp.mpf("1e-9")) > 0 and g(mp.pi) < 0
    # (2-4) θ 를 격자로 훑어 회전 곡선이 함수 그래프인지(회전 좌표 X 가 단조인지) 수치 판정
    def is_graph(th):
        ts = [i / 400 for i in range(-4000, 4001)]
        X = [mp.cos(th) * t - mp.sin(th) * f(t) for t in ts]
        return all(X[i + 1] > X[i] for i in range(len(X) - 1)) or all(X[i + 1] < X[i] for i in range(len(X) - 1))
    lo, hi = 0.0, float(mp.pi / 2)            # is_graph(lo)=True, is_graph(hi)=False
    for _ in range(40):
        mid = (lo + hi) / 2
        if is_graph(mid):
            lo = mid
        else:
            hi = mid
    ans = float(sp.sympify(md_answer(2, "4")))
    ok4 = is_graph(ans) and not is_graph(ans + 1e-3) and abs(lo - ans) < 1e-3
    report(2, ok1 and ok2 and ok3 and ok4, f"[증명-수치확인] 유일근·max 할선={mp.nstr(max(sl), 10)}<{mp.nstr(1/mp.sqrt(3), 10)} θmax≈{lo:.6f}")


def p03():
    from fractions import Fraction as Fr
    # (3-1) 0.0(1) 반복 = Σ_{k≥2} 2^-k
    k = sp.symbols("k", integer=True, positive=True)
    a = sp.summation(sp.Rational(1, 2)**k, (k, 2, sp.oo))
    ok1 = a == sp.sympify(md_answer(3, "1"))
    # (3-3) 탐욕 전개로 1/3 의 이진 자리 40개
    rem = Fr(1, 3); digs = []
    for i in range(1, 41):
        u = Fr(1, 2**i)
        if u <= rem:
            digs.append("1"); rem -= u
        else:
            digs.append("0")
    P = parts(md_answer(3, "3"))
    four = P["four_digits"].split(".")[1]
    blk = P["repeating_block"]
    ok3 = "".join(digs[:4]) == four and "".join(digs) == (blk * 40)[:40] and \
        sp.summation(sp.Rational(1, 4)**k, (k, 1, sp.oo)) == sp.Rational(1, 3)
    # (3-2) 탐욕법이 (가)(나)에서 최소 개수인지 DP 로 확인
    def greedy(M, units):
        cnt = 0
        for u in sorted(units, reverse=True):
            cnt += M // u; M %= u
        return cnt
    def dp(M, units):
        INF = 10**9; best = [0] + [INF] * M
        for v in range(1, M + 1):
            best[v] = min([best[v - u] + 1 for u in units if u <= v] + [INF])
        return best[M]
    U1 = [500, 100, 50, 10]
    U2 = [2**i for i in range(7)]
    ok2 = greedy(1370, U1) == dp(1370, U1) == 8 and greedy(73, U2) == dp(73, U2) == 3
    report(3, ok1 and ok2 and ok3, f"a={a} [증명-수치확인] 탐욕=최소(8개,3개) 1/3=0.{''.join(digs[:12])}..(2)")


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
