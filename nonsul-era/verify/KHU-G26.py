#!/usr/bin/env python3
"""경희대 2026학년도 논술(자연계) KHU-G26-01~03 — sympy 검산. 정답 하드코딩 비교 금지, 문제 설정에서 독립 재계산."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "KHU-G26"
WIDTH = 2
RESULTS = []


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """논제 I: x=4/(2n+1)(t^((2n+1)/2)-1), y=t^(2n+1)/(2n+1)-ln t-1/(2n+1), t>=1."""
    t, n = sp.symbols("t n", positive=True)
    ok = True
    vals = []
    for N in range(1, 40):
        x = sp.Rational(4, 2*N+1)*t**sp.Rational(2*N+1, 2) - sp.Rational(4, 2*N+1)
        y = t**(2*N+1)/(2*N+1) - sp.log(t) - sp.Rational(1, 2*N+1)
        slope = sp.diff(y, t)/sp.diff(x, t)
        root = sp.nsolve(slope - N, t, (1, 3), solver="bisect", prec=40)
        a_n = x.subs(t, root)
        closed = sp.Rational(4, 2*N+1)*(N - 1 + sp.sqrt(N**2+1))
        ok &= abs(sp.N(a_n - closed, 30)) < 1e-25 and root >= 1
        vals.append(a_n)
    # 극한: 닫힌 식에서 n→∞
    nn = sp.symbols("nn", positive=True)
    lim = sp.limit(4*(nn - 1 + sp.sqrt(nn**2+1))/(2*nn+1), nn, sp.oo)
    ok &= lim == 4
    # (2) l/m
    for N in range(1, 8):
        x = sp.Rational(4, 2*N+1)*t**sp.Rational(2*N+1, 2) - sp.Rational(4, 2*N+1)
        y = t**(2*N+1)/(2*N+1) - sp.log(t) - sp.Rational(1, 2*N+1)
        b = sp.exp(sp.Rational(1, 2*N+1))
        speed = sp.sqrt(sp.diff(x, t)**2 + sp.diff(y, t)**2)
        l = sp.Integral(speed, (t, 1, b)).evalf(30)
        m = y.subs(t, b)
        ok &= abs(sp.N(l/m - sp.E/(sp.E-2), 25)) < 1e-18
    report(1, ok, f"(1) a_n 수치해=닫힌식(n=1..39), lim a_n={lim}; (2) l/m=e/(e-2) (n=1..7 수치적분)")


def p02():
    """논제 II: 말판 점 4개(이동 3칸). 전수 열거로 확률 계산."""
    p = sp.symbols("p", positive=True)

    def reach(steps_needed, tails_limit):
        # 앞면 steps_needed 번이 나오기 전에 뒷면이 tails_limit 번 나오지 않을 확률(유한 열거)
        tot = 0
        L = steps_needed + tails_limit - 1
        for seq in itertools.product("HT", repeat=L):
            h = tl = 0
            for c in seq:
                if c == "H":
                    h += 1
                else:
                    tl += 1
                if h == steps_needed or tl == tails_limit:
                    break
            if h == steps_needed:
                # 같은 접두 중복 제거: 도달 시점까지의 접두만 의미 → 전체 길이 L 수열 확률 합으로 계산
                tot += p**seq.count("H") * (1-p)**seq.count("T")
        return sp.expand(tot)

    f = reach(2, 3)                      # 첫 앞면 후 남은 2칸, 뒷면 3번 전에
    g = sp.Rational(4, 5)*reach(1, 2) + sp.Rational(1, 5)*reach(2, 2)
    ok1 = sp.expand(f - p**2*(3*p**2 - 8*p + 6)) == 0
    sols = [s for s in sp.solve(sp.Eq(f, g), p) if s.is_real and 0 < s < 1]
    ok2 = sols == [sp.Rational(8, 15)]
    report(2, ok1 and ok2, f"(1) f(p)={sp.factor(f)}; (2) 0<p<1 의 근 {sols}")


def p03():
    """논제 III: d=x, AP=x+4 → 자취 y^2=16x."""
    x, y, a = sp.symbols("x y a", positive=True)
    # 자취 방정식 확인: 거리 조건
    curve = sp.solve(sp.Eq(x, sp.sqrt((x-4)**2 + y**2) - 4), y)
    ok = any(sp.simplify(c - 4*sp.sqrt(x)) == 0 for c in curve)
    f = 4*sp.sqrt(x)
    # (1) sinθ 로부터 P 의 x좌표: cosθ=(4-x)/(4+x)
    xs = []
    for s in (sp.Rational(4, 5), 2*sp.sqrt(2)/3):
        c = sp.sqrt(1 - s**2)
        xs.append(sp.solve(sp.Eq((4-x)/(4+x), c), x)[0])
    x1, x2 = xs
    # 다각형 A→P1→(곡선)→P2→A 넓이: 그린 정리
    tt = sp.symbols("tt", positive=True)
    seg = lambda P, Q: (P[0]*Q[1] - Q[0]*P[1])/2
    A = (4, 0); P1 = (x1, f.subs(x, x1)); P2 = (x2, f.subs(x, x2))
    curve_part = sp.integrate((x*sp.diff(f, x) - f)/2, (x, x1, x2))
    area = sp.simplify(abs(seg(A, P1) + curve_part + seg(P2, A)))
    ok1 = sp.simplify(area - sp.Rational(2, 3)*(14*sp.sqrt(2) - 13)) == 0
    # (2) S(a)=∫0^4 (접선 - 곡선) dx 최소
    h = f.subs(x, a) + sp.diff(f, x).subs(x, a)*(x - a)
    S = sp.integrate(h - f, (x, 0, 4))
    crit = [r for r in sp.solve(sp.diff(S, a), a) if 0 < r < 4]
    Smin = sp.simplify(S.subs(a, crit[0]))
    grid_min = min(float(S.subs(a, sp.Rational(k, 1000))) for k in range(1, 4000))
    ok2 = crit == [2] and sp.simplify(Smin - 16*(sp.sqrt(2) - sp.Rational(4, 3))) == 0 and grid_min >= float(Smin) - 1e-12
    # 접선이 곡선 위에 있음(오목) [증명-수치확인]
    ok3 = all(float((h - f).subs({a: sp.Rational(i, 10), x: sp.Rational(j, 10)})) >= -1e-12 for i in range(1, 40, 3) for j in range(0, 41, 2))
    report(3, ok and ok1 and ok2 and ok3, f"자취 y=4√x; (1) x1={x1},x2={x2}, 넓이={area}; (2) a={crit}, 최솟값={Smin}")


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
