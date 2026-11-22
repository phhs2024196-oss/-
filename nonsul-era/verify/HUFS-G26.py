#!/usr/bin/env python3
"""한국외대 2026학년도 논술전형(자연계) 문제 1~7 (HUFS-G26-01~07) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산해 YAML answer.value 와 비교.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다."""
import sys

import sympy as sp

PREFIX = "HUFS-G26"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
a, b, k = sp.symbols("a b k", real=True)
SYM = {"x": x, "t": t, "n": n, "a": a, "b": b, "k": k, "pi": sp.pi}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0, raw=False):
    """yaml 의 subparts[sub].answer.value 를 읽는다(raw=True 면 문자열 그대로)."""
    import pathlib
    import yaml
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/HUFS/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    v = d["subparts"][sub]["answer"]["value"]
    return v if raw else sp.sympify(v, locals=SYM)


def p01():
    """log2(x+a)+b = x 의 두 근이 -1, 1 → a, b 를 풀고 log2(x+a)+b = 3 의 해."""
    A, B = sp.symbols("A B", real=True)
    f = lambda X: sp.log(X + A, 2) + B
    sols = sp.solve([sp.Eq(f(-1), -1), sp.Eq(f(1), 1)], [A, B], dict=True)
    sols = [s for s in sols if s[A] > 1]                       # 정의역: -1+a>0
    assert len(sols) == 1
    s = sols[0]
    g = f(x).subs(s)
    # 증가함수이므로 역함수와의 교점 = y=x 와의 교점: 두 근이 -1, 1 인지 재확인
    chk = all(sp.simplify(g.subs(x, v) - v) == 0 for v in (-1, 1))
    X = sp.solve(sp.Eq(g, 3), x)
    ans = load_answer(1)
    ok = chk and len(X) == 1 and sp.simplify(X[0] - ans) == 0
    report(1, ok, f"a={s[A]}, b={sp.nsimplify(s[B])} (≈{sp.N(s[B], 8)}), y=3 교점 x={X}, yaml={ans}")


def p02():
    """(가)(나)로 f 를 적분해 결정 → 정수 C ∈ [-30,30] 전수: |f| 의 미분불가능점 = f'(r)≠0 인 실근 r 의 개수가 3 인 C."""
    K, C = sp.symbols("K C", real=True)
    F = sp.integrate(K * (x - 1) * (x - 3), x) + C
    Kv = sp.solve(sp.Eq(sp.Poly(F, x).LC(), 1), K)[0]
    F = sp.expand(F.subs(K, Kv))
    good = []
    for c in range(-30, 31):
        fc = F.subs(C, c)
        roots = sp.Poly(fc, x).real_roots()
        kinks = {r for r in roots if sp.diff(fc, x).subs(x, r) != 0}
        if len(kinks) == 3:
            good.append(c)
    total = sum(F.subs({C: c, x: 1}) for c in good)
    ans = load_answer(2)
    report(2, total == ans and Kv == 3, f"f={F}, 조건(다) 만족 정수 C={good}, Σf(1)={total}, yaml={ans}")


def p03():
    """세 꼭짓점 좌표로 외심을 직접 구해(두 수직이등분선 교점) R^2 을 얻고 n=1..10 합. 사인법칙 값과도 대조."""
    total = 0
    sine_ok = True
    for nv in range(1, 11):
        O, B, A = sp.Point(0, 0), sp.Point(0, 1), sp.Point(nv, nv + 1)
        cc = sp.Triangle(O, B, A).circumcenter
        R2 = sp.simplify(cc.distance(O) ** 2)
        total += sp.pi * R2
        ang = sp.Triangle(B, O, A).angles[B]
        R2_sine = sp.simplify((O.distance(A) / (2 * sp.sin(ang))) ** 2)
        sine_ok &= sp.simplify(R2 - R2_sine) == 0 and sp.simplify(ang - 3 * sp.pi / 4) == 0
    ans = load_answer(3)
    report(3, sine_ok and sp.simplify(total - ans) == 0, f"∠OBA_n=3π/4 전 n 확인 {sine_ok}, Σa_n={sp.simplify(total)}, yaml={ans}")


def p04():
    """f(n)=실수인 n제곱근 개수(값 v=log_a n - 1 의 부호와 n 의 홀짝)를 정의대로 계산.
    a>1 후보: 경계값 a=2..9 와 구간 대표점(1.5, 2.5, ..., 9.5, 10, 20) 전수 → 합이 56 인 a."""
    def nroots(v, m):
        if m % 2 == 1:
            return 1
        return 0 if v < 0 else (1 if v == 0 else 2)
    cands = [sp.Integer(c) for c in range(2, 10)] + [sp.Rational(2 * c + 1, 2) for c in range(1, 10)] + [sp.Integer(10), sp.Integer(20)]
    sols = []
    for av in cands:
        S = sum(m * nroots(sp.simplify(sp.log(m, av) - 1), m) for m in range(2, 10))
        if S == 56:
            sols.append(av)
    ans = load_answer(4)
    report(4, sols == [ans], f"Σnf(n)=56 인 a 후보(경계·구간 대표점 {len(cands)}개 전수)={sols}, yaml={ans}")


def p05():
    """f 는 (다)에서 사차, 최고차 ±1. 실근이 모두 정수 → 가능한 꼴을 전수:
    (A) 실근 4개: f=s(x-r1)(x-r2)(x-r3)(x-r4), r_i ∈ [-15,15];
    (B) 실근 2개 이하 + 허근 이차인수: f=s(x-r1)(x-r2)(x^2+px+q), p^2<4q, p,q ∈ [-6,6] 정수(실근은 허근 인수와 무관).
    (가)(나) 극한값을 cancel 로 직접 계산해 존재·양수 판정, 서로 다른 실근 합=4 → 실근 곱 집합."""
    import itertools
    prods = set()
    hits = 0

    def check(f):
        nonlocal hits
        L1 = sp.cancel(f / (x + 1))
        L2 = sp.cancel(f / (x - 2))
        if sp.denom(L1).has(x) or sp.denom(L2).has(x):
            return                                   # 극한값이 존재하지 않음(분자→0 아님)
        if not (L1.subs(x, -1) > 0 and L2.subs(x, 2) > 0):
            return
        if sp.limit(sp.Abs(f) / x**4, x, sp.oo) != 1:
            return
        rr = set(sp.Poly(f, x).real_roots())
        if not all(r.is_integer for r in rr) or sum(rr) != 4:
            return
        hits += 1
        prods.add(sp.prod(rr))

    R = range(-15, 16)
    for sgn in (1, -1):
        for r in itertools.combinations_with_replacement(R, 4):
            if -1 not in r or 2 not in r:
                continue                             # (가)(나)에서 f(-1)=f(2)=0 이 필요(아니면 극한 발산) — 아래 check 로도 확인됨
            check(sp.expand(sgn * sp.prod([x - ri for ri in r])))
        for p_, q_ in itertools.product(range(-6, 7), repeat=2):
            if p_**2 >= 4 * q_:
                continue
            check(sp.expand(sgn * (x + 1) * (x - 2) * (x**2 + p_ * x + q_)))
    ans = load_answer(5)
    report(5, prods == {ans}, f"조건 만족 f {hits}개, 실근 곱 집합={prods}, yaml={ans}")


def p06():
    """정수 공차 d ∈ [-100,100] 전수: T10=T11 인 d 를 찾고, n=1..2000 에서 T_n>T_{n+1} 인 n 의 최소+최대."""
    S = lambda m, d: sp.Rational(m * (60 + (m - 1) * d), 2)
    T = lambda m, d: abs(S(m, d))
    ds = [d for d in range(-100, 101) if T(10, d) == T(11, d)]
    # 비정수 경우 재현: |S10+a11|=|S10| 의 두 번째 해
    D = sp.symbols("D")
    other = sp.solve(sp.Eq(30 + 10 * D, -2 * (10 * (60 + 9 * D) / 2)), D)
    res = []
    for d in ds:
        ns = [m for m in range(1, 2001) if T(m, d) > T(m + 1, d)]
        res.append((d, min(ns), max(ns), min(ns) + max(ns)))
    ans = load_answer(6)
    ok = len(res) == 1 and res[0][3] == ans
    report(6, ok, f"정수 d={ds} (비정수 해 {other} 배제), (d, n최소, n최대, 합)={res}, yaml={ans}")


def p07():
    """g(t)=max_{[-1,1]} f 를 정의대로 계산: 후보점(끝점 ±1, 꺾인 점 t∈[-1,1], 각 조각의 꼭짓점 ±1)에서 f 의 최대.
    t 구간(-3,-1),(-1,1),(1,2) 의 내부 대표점 각 5개에서 후보 최대를 기호식으로 비교해 g 를 조각별 다항식으로 확정 → 적분.
    추가로 t 격자 1001점에서 x 격자 2001점 수치 최대와 대조."""
    f = lambda X, T: X**2 - 2 * sp.Abs(X - T) + 1
    def g_num(T):
        cands = [-1, 1] + ([T] if -1 <= T <= 1 else [])
        return max(f(sp.nsimplify(c), T) for c in cands)
    pieces = []
    for lo, hi, cand in ((-3, -1, -1), (-1, 1, t), (1, 2, 1)):
        # 위로 볼록한 포물선 조각이므로 최대는 조각의 끝점(-1, 1, t) 중 하나. 구간 내부에서 절댓값 부호 확정
        mid = sp.Rational(lo + hi, 2)
        sgn = sp.sign((cand - t).subs(t, mid)) if cand != t else 0
        poly = sp.expand(cand**2 - 2 * sgn * (cand - t) + 1)
        reps = [lo + sp.Rational(i, 6) * (hi - lo) for i in range(1, 6)]
        okp = all(sp.simplify(g_num(r) - poly.subs(t, r)) == 0 for r in reps)
        pieces.append((poly, lo, hi, okp))
    I = sum(sp.integrate(pc, (t, lo, hi)) for pc, lo, hi, _ in pieces)
    import numpy as np
    xs = np.linspace(-1, 1, 2001)
    ts = np.linspace(-3, 2, 1001)
    gv = np.array([np.max(xs**2 - 2 * np.abs(xs - T) + 1) for T in ts])
    I_num = np.trapz(gv, ts) if hasattr(np, "trapz") else np.trapezoid(gv, ts)
    ans = load_answer(7)
    ok = all(pc[3] for pc in pieces) and sp.simplify(I - ans) == 0 and abs(I_num - float(ans)) < 1e-3
    report(7, ok, f"g 조각={[str(pc[0]) for pc in pieces]}, ∫g={I}, 수치≈{I_num:.6f}, yaml={ans}")


if __name__ == "__main__":
    for name in sorted(k for k in dict(globals()) if k.startswith("p") and k[1:].isdigit()):
        globals()[name]()
    passed = sum(RESULTS)
    print(f"SUMMARY G26: {passed}/{len(RESULTS)} PASS")
    sys.exit(0 if passed == len(RESULTS) else 1)
