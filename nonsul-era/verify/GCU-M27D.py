#!/usr/bin/env python3
"""가천대 2027학년도 의약학 논술 모의고사 의예 수학(문제 1~4) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산."""
import re
import sys

import sympy as sp

PREFIX = "GCU-M27D"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)




def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # 좌표: C=(0,0), A=(2,0), B=√3(cosθ, sinθ) (θ=∠ACB). P=(1,0). Q: A 를 지나 BC 에 평행한 직선과 직선 BP 의 교점
    th = sp.symbols("theta", positive=True)
    C, A = sp.Matrix([0, 0]), sp.Matrix([2, 0])
    B = sp.sqrt(3) * sp.Matrix([sp.cos(th), sp.sin(th)])
    P = (A + C) / 2
    s, u = sp.symbols("s u", real=True)
    sol = sp.solve(list(A + s * (B - C) - (B + u * (P - B))), [s, u], dict=True)[0]
    Q = A + sol[s] * (B - C)
    D = (P + Q) / 2
    # 네 점 공원: A,B,C 의 외접원 위에 D. 원: x^2+y^2+Ex+Fy=0 (C 지남)
    E, F = sp.symbols("E F")
    circ = lambda Pt: Pt[0]**2 + Pt[1]**2 + E * Pt[0] + F * Pt[1]
    ef = sp.solve([circ(A), circ(B)], [E, F], dict=True)[0]
    cond = sp.simplify(circ(D).subs(ef))
    import mpmath as mpm
    fn = sp.lambdify(th, cond, "mpmath")
    roots = set()
    for i in range(1, 400):
        t0, t1 = mpm.pi * i / 400, mpm.pi * (i + 1) / 400
        if i + 1 < 400 and fn(t0) * fn(t1) < 0:
            roots.add(float(mpm.findroot(fn, (t0, t1), solver="bisect")))
    vals = []
    for r in roots:
        Bn = [float(v) for v in B.subs(th, r)]
        AB = mpm.sqrt((Bn[0] - 2)**2 + Bn[1]**2)
        R = AB / (2 * mpm.sin(r))
        # Q 가 직선 AC(x축) 에 대해 B 와 반대쪽인지
        if float(Q[1].subs(th, r)) * Bn[1] < 0:
            vals.append(float(3 * mpm.tan(r) / R))
    report(1, len(vals) == 1 and abs(vals[0] - 4) < 1e-9, f"θ 후보={roots}, 3tanθ/R={vals} (YAML 4)")


def L_k(k):
    t = sp.symbols("t", positive=True)
    A = sp.integrate((2 + t) / (k + 2 * t), (t, 0, 1))
    B = sp.integrate((k + 2 * t) / (2 + t), (t, 0, 1))
    return sp.simplify(A), sp.simplify(B)


def p02():
    import mpmath as mpm
    # 리만합 수치 확인(n=4000) + 정적분 닫힌식, L(k)=1 인 자연수 k 탐색
    k = sp.symbols("k", positive=True)
    A, B = L_k(k)
    okA = sp.simplify(A - (sp.Rational(1, 2) + (1 - k / 4) * sp.log((k + 2) / k))) == 0
    okB = sp.simplify(B - (2 + (k - 4) * sp.log(sp.Rational(3, 2)))) == 0
    n = 4000
    riem = lambda kk: (sum((2 * n + i) / (kk * n + 2 * i) for i in range(1, n + 1)) / n) * (sum((kk * n + 2 * i) / (2 * n + i) for i in range(1, n + 1)) / n)
    okR = abs(riem(3) - float((A * B).subs(k, 3))) < 1e-3
    Lf = sp.lambdify(k, A * B, "mpmath")
    mp_vals = {kk: Lf(mpm.mpf(kk)) for kk in range(1, 2001)}
    hits = [kk for kk, v in mp_vals.items() if abs(v - 1) < mpm.mpf(10)**-25]
    above = all(v > 1 for kk, v in mp_vals.items() if kk != 4)
    report(2, okA and okB and okR and hits == [4], f"적분식 확인={okA and okB}, 리만합 확인={okR}, L(k)=1 인 k={hits} (k≤2000), k≠4 에서 L(k)>1: {above} (YAML k=4)")


def p03():
    c, k = sp.symbols("c k", real=True)
    g = lambda u: 2 * u + sp.sin(sp.pi * u / 2)
    total = 0
    found = []
    # f(a)^2=9, f'(a)=0. f(a)=-3 이면 g^{-1}(a)=3, a=g(3)=5 > k(음의 정수) 모순 → f(a)=3, a=g(-3)
    a = g(-3)
    for kk in range(-20, 0):
        if not (a < kk):
            continue
        for cc in [sp.Rational(-5, 3) / (kk + 5)]:
            if cc >= 0:
                continue
            f = sp.integrate(cc * (x - a) * (x - kk), (x, a, x)) + 3
            # x=a 에서 미분가능: 좌미분 = 우미분 (수치 극한으로 독립 확인)
            import mpmath as mpm
            mpm.mp.dps = 50
            fl = sp.lambdify(x, f, "mpmath")
            ginv = lambda y: mpm.findroot(lambda u: 2 * u + mpm.sin(mpm.pi * u / 2) - y, -3)
            hL = lambda u: (fl(u) + ginv(u)) / mpm.log(10)
            hR = lambda u: (mpm.log10(1 + fl(u)**2) - 1) / (u - a)
            e = mpm.mpf(10)**-12
            dl = (hL(a) - hL(a - e)) / e
            dr = (hR(a + e) - hL(a)) / e
            if abs(dl - dr) < 1e-8 and abs(hR(a + e) - hL(a)) < 1e-9:
                found.append((kk, cc, f.subs(x, kk)))
                total += f.subs(x, kk)
    report(3, a == -5 and [t[0] for t in found] == [-4, -3, -2, -1] and sp.nsimplify(total) == sp.Rational(61, 3),
           f"a={a}, (k, c, 극댓값)={found}, 합={sp.nsimplify(total)} (YAML 61/3)")


def p04():
    # f = x^3+px^2+qx+r. 조건 (나): f(x)-x = (x-a)^2(x-b) 꼴(정수 a≠b) 전수, (가): f 의 실근 집합이 x→2x-2 로 닫힘(극한 존재) 수치 확인
    res = set()
    for a in range(-20, 21):
        for b in range(-20, 21):
            if a == b:
                continue
            f = sp.expand((x - a)**2 * (x - b) + x)
            rts = [r for r in sp.real_roots(sp.Poly(f, x))]
            ok = True
            for r in set(rts):
                # 극한 lim_{x→r} f(2x-2)/f(x) 존재 ⇔ f(2x-2) 가 (x-r)^m 로 나누어짐(m=r 의 중복도)
                m = rts.count(r)
                num = sp.Poly(f.subs(x, 2 * x - 2), x)
                if sp.rem(num, sp.Poly((x - r)**m, x)).as_expr() != 0:
                    ok = False
            if ok:
                area = sp.integrate(sp.Abs(f - x), (x, min(a, b), max(a, b)))
                res.add((a, b, sp.nsimplify(area)))
    areas = {t[2] for t in res}
    report(4, areas == {sp.Rational(1, 12)}, f"가능한 (중근, 단근, 넓이)={res} (YAML 1/12)")


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
