#!/usr/bin/env python3
"""KKU-G26 (건국대 2026학년도 KU논술우수자 자연계 문제 1~4) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "KKU-G26"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
LOC = {"e": sp.E, "E": sp.E}


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/KKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """A(a,e^a), B 는 ln x 위에서 AB 기울기 -1, C=-A. ∠ACB 를 a 에 대해 직접 수치 최소화."""
    import mpmath as mp
    mp.mp.dps = 30

    def angle(a):
        A = (a, mp.e**a)
        # B=(s, ln s), (ln s - e^a)/(s - a) = -1  →  ln s + s - a - e^a = 0 (좌변 증가함수, 근 유일)
        s = mp.findroot(lambda s: mp.log(s) + s - a - mp.e**a, A[1])
        B = (s, mp.log(s))
        C = (-A[0], -A[1])
        u = (A[0] - C[0], A[1] - C[1]); v = (B[0] - C[0], B[1] - C[1])
        return mp.acos((u[0]*v[0] + u[1]*v[1]) / (mp.sqrt(u[0]**2 + u[1]**2) * mp.sqrt(v[0]**2 + v[1]**2)))

    grid = [i / 200 for i in range(1, 801)]          # 제1사분면: a>0
    a0 = min(grid, key=angle)
    amin = mp.findroot(lambda a: mp.diff(angle, a), a0)
    ans = sp.sympify(md_answer(1, "1"), locals=LOC)
    ok = abs(amin - float(ans[0])) < 1e-12 and abs(mp.e**amin - float(ans[1])) < 1e-10 \
        and angle(amin) < angle(amin + 0.05) and angle(amin) < angle(amin - 0.05)
    report(1, ok, f"argmin a={mp.nstr(amin, 15)} A=({mp.nstr(amin, 6)}, {mp.nstr(mp.e**amin, 10)}) "
                  f"min∠ACB={mp.nstr(angle(amin), 10)}")


def p02():
    """세 장 조합을 전수 열거해 X 분포·기댓값을 직접 계산."""
    def dist(N):
        cnt = {}
        tot = 0
        for c in itertools.combinations(range(1, N + 1), 3):
            k = c[2] - c[0]
            cnt[k] = cnt.get(k, 0) + 1
            tot += 1
        return cnt, tot

    cnt, tot = dist(10)
    p9 = sp.Rational(cnt.get(9, 0), tot)
    ok1 = p9 == sp.sympify(md_answer(2, "1"))
    # 기호: E(X) 일반식(전수 열거로 n=3..30 확인한 뒤 sympy 합으로 닫힌 꼴)
    k = sp.symbols("k", integer=True)
    EX = sp.simplify(sp.summation(k * (n - k) * (k - 1), (k, 2, n - 1)) / sp.binomial(n, 3))
    ok_formula = all(
        sp.Rational(sum(kk * c for kk, c in dist(N)[0].items()), dist(N)[1]) == EX.subs(n, N)
        for N in range(3, 31))
    sols = [int(N) for N in range(3, 1000) if EX.subs(n, N) == 30]
    ok2 = sols == [int(sp.sympify(md_answer(2, "2")))] and ok_formula
    report(2, ok1 and ok2, f"P(X=9)={p9} E(X)={sp.factor(EX)} (전수 n=3..30 일치={ok_formula}) E=30 → n={sols}")


def p03():
    """P(p,p^2+1), Q(q,-(q-a)^2): l1∥l2, PQ⊥l1 을 연립해 a=8 에서 P 를 구하고, dL/da 는 a 근방 수치미분으로 확인."""
    import mpmath as mp
    mp.mp.dps = 40
    p, q, a = sp.symbols("p q a", real=True)
    eqs = [sp.Eq(2 * p, -2 * (q - a)),                                          # 접선 평행
           sp.Eq(2 * p * (-(q - a)**2 - (p**2 + 1)), -(q - p))]                 # PQ ⊥ l1 (분모 정리)
    sol8 = [s for s in sp.solve([e.subs(a, 8) for e in eqs], [p, q], dict=True)
            if s[p].is_real and s[q].is_real]
    P8 = [(s[p], s[p]**2 + 1) for s in sol8]
    ans1 = sp.sympify(md_answer(3, "1"))
    ok1 = len(P8) == 1 and sp.simplify(P8[0][0] - ans1[0]) == 0 and sp.simplify(P8[0][1] - ans1[1]) == 0

    def L(av):
        # a 를 주면 p 는 실근 유일(4p^3+4p-a 증가) — 연립에서 q 소거한 식을 직접 쓰지 않고 수치로 풂
        f = sp.lambdify(p, eqs[1].lhs.subs({q: a - p}).subs(a, av) - eqs[1].rhs.subs({q: a - p}).subs(a, av), "mpmath")
        pv = mp.findroot(f, 1)
        qv = av - pv
        return mp.sqrt((qv - pv)**2 + (-(qv - av)**2 - (pv**2 + 1))**2)

    dL = mp.diff(L, 8)
    ans2 = sp.sympify(md_answer(3, "2"))
    ok2 = abs(dL - mp.mpf(str(sp.N(ans2, 40)))) < mp.mpf("1e-25")
    report(3, ok1 and ok2, f"a=8: P={P8} dL/da={mp.nstr(dL, 20)}")


def p04():
    """A(a,b) 에 대해 (t-a, t^2/2-b)·(1,t)=0 의 서로 다른 실근 수를 직접 세어 3 인 격자점 개수."""
    count = 0
    pts = []
    for av in range(4, 21):
        for bv in range(5, 8):
            if sp.Rational(av**2, 2) == bv:          # A 가 곡선 위면 제외
                continue
            poly = sp.Poly(sp.expand((t - av) * 1 + (t**2 / 2 - bv) * t), t)
            nroots = len(set(sp.real_roots(poly)))
            if nroots == 3:
                count += 1
                pts.append((av, bv))
    ok = count == int(sp.sympify(md_answer(4, "1")))
    report(4, ok, f"count={count} points={pts}")


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
