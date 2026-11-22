#!/usr/bin/env python3
"""서울과기대 2027학년도 모의논술고사 문제 1~3 (SNUT-M27-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산해 YAML answer.value 와 비교.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SNUT-M27"
WIDTH = 2
RESULTS = []
x, t, n, k = sp.symbols("x t n k", real=True)
SYM = {"x": x, "t": t, "n": n, "k": k, "pi": sp.pi, "e": sp.E}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0, raw=False):
    """yaml 의 subparts[sub].answer.value 를 읽는다(raw=True 면 문자열 그대로)."""
    import pathlib
    import yaml
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/SNUT/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    v = d["subparts"][sub]["answer"]["value"]
    return v if raw else sp.sympify(v, locals=SYM)


def same(a, b, tol=1e-9):
    """기호 동일 또는 수치 근사 동일."""
    try:
        if sp.simplify(a - b) == 0:
            return True
    except Exception:  # noqa: BLE001
        pass
    return abs(complex(sp.N(a - b, 30))) < tol


import itertools
import math


def p01():
    import numpy as np
    # [1.1] 격자 탐색 후 국소 정밀화
    F = lambda a_, b_: (a_ + 2*b_ + 1)**2 - 8*a_*b_ + 18/(a_ + 2*b_)
    A, B = np.meshgrid(np.linspace(0.01, 8, 1600), np.linspace(0.01, 4, 800))
    V = F(A, B)
    i = np.unravel_index(np.argmin(V), V.shape)
    a0, b0 = A[i], B[i]
    # 좌표별 삼분탐색(국소 볼록)
    def tern(fn, lo, hi):
        for _ in range(120):
            m1 = lo + (hi - lo)/3; m2 = hi - (hi - lo)/3
            if fn(m1) < fn(m2): hi = m2
            else: lo = m1
        return (lo + hi)/2
    a1, b1 = a0, b0
    for _ in range(60):
        a1 = tern(lambda z: F(z, b1), max(1e-6, a1 - 0.1), a1 + 0.1)
        b1 = tern(lambda z: F(a1, z), max(1e-6, b1 - 0.1), b1 + 0.1)
    raw = load_answer(1, 0, raw=True)
    ya, yb = [float(sp.sympify(z.split("=")[1])) for z in raw.split(";")]
    # 기호: 임계점도 확인
    aa, bb = sp.symbols("aa bb", positive=True)
    crit = sp.solve([sp.diff(F(aa, bb), aa), sp.diff(F(aa, bb), bb)], [aa, bb], dict=True)
    ok1 = abs(a1 - ya) < 1e-5 and abs(b1 - yb) < 1e-5 and abs(F(ya, yb) - 13) < 1e-12 and any(
        same(c[aa], sp.sympify(raw.split(";")[0].split("=")[1])) and same(c[bb], sp.sympify(raw.split(";")[1].split("=")[1])) for c in crit)
    report(1, ok1, f"[1.1] 수치 최소 (a,b)=({a1:.6f},{b1:.6f}) 값 {F(a1,b1):.9f}, 임계점={crit}, yaml=({ya},{yb})")
    # [1.2]
    def a_n(nn):
        p = sp.Poly(sp.prod([x + k_**2 for k_ in range(1, nn + 1)]), x)
        return p.coeff_monomial(x**(nn - 1))
    form = lambda nn: sp.Rational(nn*(nn + 1)*(2*nn + 1), 6)
    chk_a = all(a_n(nn) == form(nn) for nn in range(1, 10))
    def b_n(nn):
        m = 2*nn + 1
        return sp.integrate(x*sp.cos(x/m), (x, 0, m**2*sp.pi))
    chk_b = all(sp.simplify(b_n(nn) + 2*(2*nn + 1)**2) == 0 for nn in range(1, 6))
    nn_ = sp.symbols("nn_", positive=True)
    L = sp.limit(nn_*(-2*(2*nn_ + 1)**2)/(nn_*(nn_ + 1)*(2*nn_ + 1)/6), nn_, sp.oo)
    ratio_num = float(sp.N(10**7*(-2*(2*10**7 + 1)**2)/(sp.Rational(10**7*(10**7 + 1)*(2*10**7 + 1), 6))))
    ok2 = chk_a and chk_b and same(L, load_answer(1, 1)) and abs(ratio_num - float(load_answer(1, 1))) < 1e-5
    report(1, ok2, f"[1.2] a_n 전개 계수=공식(n≤9)={chk_a}, b_n 직접 적분=-2(2n+1)²(n≤5)={chk_b}, 극한={L}, n=1e7 비={ratio_num:.6f}, yaml={load_answer(1,1)}")
    # [1.3] 거리 S(a)=∫0^{10-2a}|v| 를 격자로 최대화
    def S(a_):
        ts = np.linspace(0, 10 - 2*a_, 200001)
        v = -a_/(5 - a_)**2*ts**2 + a_
        return float(np.trapz(np.abs(v), ts)) if hasattr(np, "trapz") else float(np.trapezoid(np.abs(v), ts))
    As = np.linspace(0.05, 4.95, 491)
    Sv = np.array([S(a_) for a_ in As])
    amax = As[int(np.argmax(Sv))]
    closed = max(2*a_*(5 - a_) for a_ in As)
    ok3 = abs(amax - float(load_answer(1, 2))) < 1e-9 and abs(Sv.max() - closed) < 1e-6
    report(1, ok3, f"[1.3] 격자(Δa=0.01)에서 거리 최대 a={amax:.3f}, 최대값≈{Sv.max():.6f}, yaml={load_answer(1,2)}")
    # [1.4] 전수 열거: 남 B0..B4(합창단원 B0,B1), 여 G0..G3, 9명 중 6명 선택 × 일렬 배열
    people = [("B", i) for i in range(5)] + [("G", i) for i in range(4)]
    cnt = 0
    for sel in itertools.combinations(people, 6):
        nb = sum(1 for p in sel if p[0] == "B")
        if nb != 3: continue
        if sum(1 for p in sel if p == ("B", 0) or p == ("B", 1)) > 1: continue
        for perm in itertools.permutations(sel):
            if all(not (perm[k][0] == "G" and perm[k + 1][0] == "G") for k in range(5)):
                cnt += 1
    ok4 = cnt == int(load_answer(1, 3))
    report(1, ok4, f"[1.4] 9명 중 6명 선택·배열 전수 열거={cnt}, yaml={load_answer(1,3)}")


def p02():
    import numpy as np
    # [2.1] 기하 직접: P=(2cosφ,2sinφ), A=(Px,0), B=(0,Py), ∠ABP 를 벡터로 계산 → φ 와 비교
    maxerr = 0.0
    raw = load_answer(2, 0, raw=True)
    ex, ey = [sp.sympify(z.split("=")[1], locals={"theta": sp.Symbol("theta")}) for z in raw.split(";")]
    th = sp.Symbol("theta")
    for ph in np.linspace(0.01, math.pi/2 - 0.01, 300):
        P = np.array([2*math.cos(ph), 2*math.sin(ph)]); A = np.array([P[0], 0.0]); B = np.array([0.0, P[1]])
        u = A - B; w = P - B
        ang = math.acos(np.dot(u, w)/(np.linalg.norm(u)*np.linalg.norm(w)))
        px = float(ex.subs(th, ang)); py = float(ey.subs(th, ang))
        maxerr = max(maxerr, abs(px - P[0]), abs(py - P[1]))
    report(2, maxerr < 1e-9, f"[2.1] 기하 구성(300점)으로 ∠ABP=θ 로 되돌린 P 와 yaml 식 최대오차={maxerr:.2e}, yaml=({ex}, {ey})")
    # [2.2] r(θ) = 2*넓이/둘레 를 세 점에서 풀어 a,b,c 결정 후 다른 점 검사
    def geom_r(ph):
        P = np.array([2*math.cos(ph), 2*math.sin(ph)]); A = np.array([P[0], 0.0]); B = np.array([0.0, P[1]])
        s1, s2, s3 = np.linalg.norm(A - B), np.linalg.norm(P - B), np.linalg.norm(P - A)
        area = 0.5*abs((A[0]-P[0])*(B[1]-P[1]) - (A[1]-P[1])*(B[0]-P[0]))
        return 2*area/(s1 + s2 + s3)
    ths = [0.3, 0.7, 1.2]
    M = np.array([[math.sin(t_), math.cos(t_), 1.0] for t_ in ths]); rhs = np.array([geom_r(t_) for t_ in ths])
    sol = np.linalg.solve(M, rhs)
    raw2 = load_answer(2, 1, raw=True)
    ya, yb, yc = [float(sp.sympify(z.split("=")[1])) for z in raw2.split(";")]
    resid = max(abs(geom_r(t_) - (ya*math.sin(t_) + yb*math.cos(t_) + yc)) for t_ in np.linspace(0.05, 1.5, 200))
    ok2 = np.allclose(sol, [ya, yb, yc], atol=1e-9) and resid < 1e-9
    report(2, ok2, f"[2.2] 기하 r 로 푼 (a,b,c)={np.round(sol,10).tolist()}, 200점 잔차={resid:.2e}, yaml=({ya},{yb},{yc})")
    # [2.3] 내심을 변 길이 가중평균으로 구해 OC=√70/5 가 되는 θ∈(0,π/4) 탐색
    def OC(ph):
        P = np.array([2*math.cos(ph), 2*math.sin(ph)]); A = np.array([P[0], 0.0]); B = np.array([0.0, P[1]])
        la, lb, lp = np.linalg.norm(P - B), np.linalg.norm(P - A), np.linalg.norm(A - B)   # A,B,P 의 대변
        C = (la*A + lb*B + lp*P)/(la + lb + lp)
        return float(np.linalg.norm(C))
    target = math.sqrt(70)/5
    phs = np.linspace(1e-6, math.pi/4 - 1e-6, 400001)
    vals = np.array([OC(p_) for p_ in phs[::400]])
    idx = [i for i in range(len(vals) - 1) if (vals[i] - target)*(vals[i + 1] - target) < 0]
    roots = []
    for i in idx:
        lo, hi = phs[::400][i], phs[::400][i + 1]
        for _ in range(100):
            mid = (lo + hi)/2
            if (OC(lo) - target)*(OC(mid) - target) <= 0: hi = mid
            else: lo = mid
        roots.append((lo + hi)/2)
    tans = [math.tan(r_) for r_ in roots]
    ok3 = len(tans) == 1 and abs(tans[0] - float(load_answer(2, 2))) < 1e-8
    report(2, ok3, f"[2.3] OC=√70/5 인 θ∈(0,π/4)={roots}, tanθ={tans}, yaml={load_answer(2,2)}")


def p03():
    import numpy as np
    # [3.1] 진수 조건 영역에서 수치 근 탐색
    g = lambda r: math.log2(1 + r) + math.log2(3 - r) - 4*math.log(1 - r, 4) - math.log2(15)
    rs = np.linspace(-0.999999, 0.999999, 2000001)
    gv = np.array([g(r) for r in rs[::1000]])
    sub = rs[::1000]
    roots = []
    for i in range(len(gv) - 1):
        if gv[i]*gv[i + 1] < 0:
            lo, hi = sub[i], sub[i + 1]
            for _ in range(200):
                mid = (lo + hi)/2
                if g(lo)*g(mid) <= 0: hi = mid
                else: lo = mid
            roots.append((lo + hi)/2)
    # 영역 밖 후보 r=3/2 는 진수 1-r<0 으로 불가
    ok1 = len(roots) == 1 and abs(roots[0] - float(load_answer(3, 0))) < 1e-9
    report(3, ok1, f"[3.1] (-1,1)에서 수치 근={roots}, yaml={load_answer(3,0)} (r=3/2 는 1-r<0)")
    # [3.2]
    a, b = sp.symbols("a b")
    r = load_answer(3, 0)
    f = a*x**2 + b*x
    sol = sp.solve([sp.Eq(f.subs(x, r), 1), sp.Eq(sp.diff(f, x).subs(x, r), 0)], [a, b], dict=True)[0]
    yf = sp.sympify(load_answer(3, 1, raw=True).split("=")[1], locals=SYM)
    ok2 = same(f.subs(sol) - yf, 0)
    report(3, ok2, f"[3.2] f={sp.expand(f.subs(sol))}, yaml={yf}")
    # [3.3]
    nn = sp.symbols("nn", integer=True, positive=True)
    fx = f.subs(sol)
    Sn_nums = [sp.integrate(sp.Abs(x**k_*fx), (x, 0, 1)) for k_ in range(1, 9)]
    Sn_sym = sp.simplify(sp.integrate(x**nn*fx, (x, 0, 1)))
    form_ok = all(same(Sn_sym.subs(nn, k_), Sn_nums[k_ - 1]) for k_ in range(1, 9))
    tot = sp.summation(sp.apart(Sn_sym, nn), (nn, 1, sp.oo))
    ok3 = form_ok and same(tot, load_answer(3, 2))
    report(3, ok3, f"[3.3] S_n 기호식={Sn_sym}, n≤8 |적분| 일치={form_ok}, Σ S_n={tot}, yaml={load_answer(3,2)}")


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
