#!/usr/bin/env python3
"""서울과기대 2025학년도 논술전형고사(오후) 문제 1~3 (SNUT-G25B-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산해 YAML answer.value 와 비교.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SNUT-G25B"
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
from sympy.calculus.util import continuous_domain
import math


def p01():
    # [1.1] x(t)=∫0^t |s-5| ds 를 수치 적분(구분구적)과 조각 적분으로 계산해 부등식 영역을 격자로 판정
    s = sp.symbols("s", real=True)
    def xpos(tv):
        tv = sp.nsimplify(tv)
        return sp.integrate(sp.Abs(s - 5), (s, 0, tv))
    expr = load_answer(1, 0, raw=True)
    pred = lambda tv: eval(expr.replace(" or ", " or ").replace("sqrt", "math.sqrt"), {"math": math, "t": tv})
    grid = [sp.Rational(i, 20) for i in range(0, 241)]
    bad = [g for g in grid if ((xpos(g) <= 8) or (xpos(g) >= 14)) != pred(float(g))]
    # 경계점 정밀 확인
    b1 = sp.solve(sp.Eq(-t**2/2 + 5*t, 8), t); b2 = [r for r in sp.solve(sp.Eq(t**2/2 - 5*t + 25, 14), t) if r > 5]
    ok1 = not bad and 2 in b1 and same(b2[0], 5 + sp.sqrt(3))
    report(1, ok1, f"[1.1] 격자(0~12, 241점) 불일치={len(bad)}, 경계 t={sorted(b1)}, {b2}, yaml={expr}")
    # [1.2] 두 직선의 방향벡터 (1,1),(n,1): tan = |cross|/dot
    prod = sp.Integer(1)
    for nn in range(2, 21):
        tanv = sp.Abs(1*1 - 1*nn) / sp.Integer(1*nn + 1*1)
        prod *= tanv
    ok2 = same(prod, load_answer(1, 1))
    report(1, ok2, f"[1.2] 곱={prod}, yaml={load_answer(1,1)}")
    # [1.3] 16개 점 전수: 어느 세 점도 한 직선 위에 있지 않은 4점 조합(볼록 사각형이 됨)
    pts = [(i, 0) for i in range(5)] + [(4, j) for j in range(1, 5)] + [(i, 4) for i in range(3, -1, -1)] + [(0, j) for j in range(3, 0, -1)]
    assert len(set(pts)) == 16
    def col(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]) == 0
    cnt = sum(1 for q in itertools.combinations(pts, 4) if not any(col(*tr) for tr in itertools.combinations(q, 3)))
    ok3 = cnt == int(load_answer(1, 2))
    report(1, ok3, f"[1.3] 전수 열거 사각형 수={cnt}, yaml={load_answer(1,2)}")


def p02():
    import numpy as np
    # 합의 거리 함수는 (a,b)에서 볼록 → 중첩 삼분탐색
    P = np.array([0.0, 4.0]); Q = np.array([4.0, 2.0])
    def f(av, bv):
        return np.linalg.norm(np.array([av, 5.0]) - P) + math.hypot(av - bv, 5.0) + np.linalg.norm(np.array([bv, 0.0]) - Q)
    def argmin1(fn, lo, hi):
        for _ in range(200):
            m1 = lo + (hi - lo)/3; m2 = hi - (hi - lo)/3
            if fn(m1) < fn(m2): hi = m2
            else: lo = m1
        return (lo + hi)/2
    def inner(av):
        bb = argmin1(lambda bv: f(av, bv), 0.0, 4.0)
        return f(av, bb)
    a_opt = argmin1(inner, 0.0, 4.0)
    b_opt = argmin1(lambda bv: f(a_opt, bv), 0.0, 4.0)
    class best:  # 결과 묶음
        fun = f(a_opt, b_opt); x = (a_opt, b_opt)
    ans = float(sp.N(load_answer(2, 0)))
    ok1 = abs(best.fun - ans) < 1e-6
    report(2, ok1, f"[2.1] 수치 최솟값={best.fun:.9f} at X=({best.x[0]:.5f},5), Y=({best.x[1]:.5f},0), yaml≈{ans:.9f}")
    # [2.2] UD ⟂ XY, D=(4,5), U=(0,u), M = XY ∩ UD, 외접원 반지름 = abc/(4 area)
    X = sp.Point(sp.Rational(1, 2), 5); Y = sp.Point(3, 0); D = sp.Point(4, 5)
    assert abs(best.x[0]-0.5) < 1e-4 and abs(best.x[1]-3) < 1e-4
    u = sp.symbols("u", real=True)
    uu = sp.solve(sp.Eq(sp.Matrix([0 - 4, u - 5]).dot(sp.Matrix([Y.x - X.x, Y.y - X.y])), 0), u)[0]
    U = sp.Point(0, uu)
    M = sp.Line(X, Y).intersection(sp.Line(U, D))[0]
    tri = sp.Triangle(M, U, Y)
    a_ = Y.distance(M); b_ = M.distance(U); c_ = U.distance(Y)
    sinM = sp.sin(sp.Triangle(M, U, Y).angles[M]); sinU = sp.sin(tri.angles[U]); sinY = sp.sin(tri.angles[Y])
    ident = sp.simplify(c_*sinM - (a_*sinU + b_*sinY)) == 0
    R = sp.simplify(tri.circumradius)
    ok2 = ident and same(R, load_answer(2, 1))
    report(2, ok2, f"[2.2] U=(0,{uu}), M={M}, 등식 성립={ident}, R={R}, yaml={load_answer(2,1)}")
    # [2.3] 외접원(중심 O=UY 중점, 반지름 R) 위 호 UY 를 n등분, S_k=넓이, (1/n)ΣS_k → 극한
    Uc = np.array([0.0, float(uu)]); Yc = np.array([3.0, 0.0]); O = (Uc + Yc)/2; Rf = float(R)
    def mean_S(nn):
        th0 = math.atan2(*(Uc - O)[::-1])
        tot = 0.0
        for kk in range(1, nn):
            ang = th0 - kk*math.pi/nn   # 반원을 따라 U 에서 Y 로
            V = O + Rf*np.array([math.cos(ang), math.sin(ang)])
            tot += 0.5*abs((Yc[0]-Uc[0])*(V[1]-Uc[1]) - (Yc[1]-Uc[1])*(V[0]-Uc[0]))
        return tot/nn
    v1, v2 = mean_S(20000), mean_S(40000)
    lim_rich = (4*v2 - v1)/3
    ans3 = float(sp.N(load_answer(2, 2)))
    nn_ = sp.symbols("nn_", positive=True)
    lim_sym = sp.limit(sp.Rational(9, 2)/nn_*sp.cot(sp.pi/(2*nn_)), nn_, sp.oo)
    ok3 = abs(lim_rich - ans3) < 1e-8 and same(lim_sym, load_answer(2, 2))
    report(2, ok3, f"[2.3] n=40000 평균 S_k={v2:.9f}, 외삽≈{lim_rich:.9f}, 기호극한={lim_sym}, yaml≈{ans3:.9f}")


def p03():
    import numpy as np
    f = lambda z: z - sp.log(z + 3) + 1
    g = lambda z: sp.log(z + 3) - 1
    # [3.1] 연속, 부호, 보기 선택
    c0 = sp.N(f(0)); c1 = sp.N(f(1))
    cont = continuous_domain(f(x), x, sp.Interval(0, 1)) == sp.Interval(0, 1)
    c0_09 = (-1.099 + 1) < 0; c1_09 = (2 - 2*0.693) > 0   # 제시문 (라)의 근삿값으로도 같은 부호
    raw = load_answer(3, 0, raw=True)
    expected = {"①": "연속" if cont else "미분가능", "②": "<0" if c0 < 0 else (">0" if c0 > 0 else "=0"),
                "③": ">0" if c1 > 0 else ("<0" if c1 < 0 else "=0"), "④": "사잇값 정리"}
    parsed = {p.strip()[0]: p.strip()[1:].strip() for p in raw.split(",")}
    ok1 = cont and c0_09 and c1_09 and all(parsed[kk].replace(" ", "") == vv.replace(" ", "") for kk, vv in expected.items())
    report(3, ok1, f"[3.1] 연속={cont}, f(0)={c0:.4f}, f(1)={c1:.4f}, 기대={expected}, yaml={parsed}")
    # [3.2] [증명-수치확인] (0,1)에서 f'>0 (격자 1000점), 부호변화 1회
    xs = np.linspace(0.0005, 0.9995, 1000)
    fp = sp.lambdify(x, sp.diff(f(x), x), "numpy")
    fv = sp.lambdify(x, f(x), "numpy")
    sgn = np.sign(fv(np.linspace(0, 1, 100001)))
    changes = int(np.sum(sgn[1:] != sgn[:-1]))
    ok2 = bool(np.all(fp(xs) > 0)) and changes == 1
    report(3, ok2, f"[3.2] [증명-수치확인] f'>0 격자 1000점={bool(np.all(fp(xs)>0))}, (0,1) 부호변화 횟수={changes}")
    # [3.3] α 와 가장 큰 자연수 p
    alpha = sp.nsolve(f(x), x, 0.5, prec=30)
    ts = float(alpha) + np.logspace(-9, 3, 4000)
    gv = sp.lambdify(x, g(x), "numpy"); a_f = float(alpha)
    ratio = (gv(ts) - a_f) / (ts - a_f)
    good = [p for p in range(1, 12) if bool(np.all((gv(ts) - a_f > 0) & (ratio < 1.0/p)))]
    ok3 = max(good) == int(load_answer(3, 2))
    report(3, ok3, f"[3.3] α={sp.N(alpha,12)}, 조건 만족 p={good}, 최대={max(good)}, yaml={load_answer(3,2)}")
    # [3.4] a_{n+1}=g(a_n), M 식
    raw4 = load_answer(3, 3, raw=True)
    Mexpr = sp.sympify(raw4.split(";")[0].split("=")[1], locals=SYM)
    lim_name = raw4.split(";")[1].strip()
    alpha = sp.nsolve(f(x), x, 0.5, prec=120)
    okk = True
    av = sp.Float(1, 120)
    for nn in range(1, 61):
        d = av - alpha
        Mv = Mexpr.subs(n, nn)
        # n=1 은 a_1-α=(1-α)=M(1-α) 로 등호(공식 해설은 엄밀 부등호로 적음 → 보고 사항)
        if not (0 < d <= Mv*(1 - alpha) and (nn == 1 or d < Mv*(1 - alpha))):
            okk = False
        av = g(av)
    ok4 = okk and abs(av - alpha) < 1e-25 and lim_name == "alpha"
    report(3, ok4, f"[3.4] n=1..60에서 0<a_n-α<M(1-α) 성립(n=1은 등호)={okk}, a_61-α={sp.N(av-alpha,5)}, M={Mexpr}, 극한={lim_name}")


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
