#!/usr/bin/env python3
"""숭실대 2022학년도 모의논술고사(자연) (SSU-M22-01~04) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
yaml 의 answer.value(sympy 문법, 'a=.., b=..' 꼴은 이름별 dict, 'L=R' 꼴은 L-R 식)를 읽어 독립 재계산 값과 비교한다.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "SSU-M22"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
mp.mp.dps = 30


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def _split_top(s):
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def parse_answer(s):
    """'1/4' → 식, '[..]' → 리스트, 'a=1, b=2' → dict, 'L=R' → L-R."""
    s = str(s).strip()
    loc = {"E": sp.E, "pi": sp.pi}
    if s.startswith("["):
        return list(sp.sympify(s, locals=loc))
    parts = _split_top(s)
    if all(re.fullmatch(r"[A-Za-z]\w*\s*=[^=].*", p) for p in parts):
        if len(parts) > 1 or re.fullmatch(r"[A-Za-z]\w*", parts[0].split("=")[0].strip()):
            return {p.split("=", 1)[0].strip(): sp.sympify(p.split("=", 1)[1], locals=loc) for p in parts}
    if "=" in s:
        l, r = s.split("=", 1)
        return sp.sympify(l, locals=loc) - sp.sympify(r, locals=loc)
    return sp.sympify(s, locals=loc)


def load_answer(seq, sub="1"):
    import pathlib
    import yaml
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/SSU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    sp_ = next(s for s in d["subparts"] if str(s["id"]) == str(sub))
    return parse_answer(sp_["answer"]["value"])


def close(a, b, tol=1e-12):
    return abs(complex(sp.N(a, 40)) - complex(sp.N(b, 40))) < tol


def p01():
    """수열의 극한으로 정의된 f 를 큰 n 에서 수치로 확인해 구간별 식을 얻고, 연속·미분가능·근 개수 조건을 sympy 로 풂."""
    a, b, c = sp.symbols("a b c", real=True)
    N = sp.Symbol("N", positive=True)
    expr = lambda X: (-X**(N + 2) + a * X**(N + 1) + b * X**N + c * sp.cos(4 * sp.pi * X / 3)) / (X**N + 1)
    fL = c * sp.cos(4 * sp.pi * x / 3)
    fR = -x**2 + a * x + b
    f1 = sp.limit(expr(sp.Integer(1)), N, sp.oo)
    # 구간별 식 확인: 임의의 a,b,c 와 x=0.4, 1.7 에서 n=400 값과 비교
    vals = {a: sp.Rational(3, 7), b: -sp.Rational(5, 3), c: sp.Rational(2, 5)}
    for X0, piece in ((sp.Rational(2, 5), fL), (sp.Rational(17, 10), fR)):
        approx = sp.N(expr(X0).subs(vals).subs(N, 400), 30)
        assert abs(approx - sp.N(piece.subs(vals).subs(x, X0), 30)) < 1e-20
    # (1) 연속: 좌극한 = f(1) = 우극한
    a_sol = sp.solve([fL.subs(x, 1) - f1, fR.subs(x, 1) - f1], [a], dict=True)
    R_ = {sp.Symbol("a"): a, sp.Symbol("b"): b, sp.Symbol("c"): c}
    ans1 = {k: v.subs(R_) for k, v in load_answer(1, "1").items()}
    ok1 = len(a_sol) == 1 and sp.simplify(a_sol[0][a] - ans1["a"]) == 0
    # (2) 미분가능: 좌우 미분계수 일치
    fR1 = fR.subs(a, a_sol[0][a])
    b_sol = sp.solve(sp.diff(fL, x).subs(x, 1) - sp.diff(fR1, x).subs(x, 1), b)
    ans2 = {k: v.subs(R_) for k, v in load_answer(1, "2").items()}
    ok2 = len(b_sol) == 1 and sp.simplify(b_sol[0] - ans2["b"]) == 0
    # (3) [0,∞) 에서 f=0 의 서로 다른 해가 2개인 c>0 : 해 개수 함수를 직접 세어 판정
    fR2 = sp.expand(fR1.subs(b, b_sol[0]))

    def count(cv):
        roots = set()
        # [0,1): c cos(4πx/3)=0 ⇔ 4πx/3 = π/2 + kπ
        for k in range(0, 3):
            xr = sp.Rational(3, 8) + sp.Rational(3, 4) * k
            if 0 <= xr < 1:
                roots.add(xr)
        if sp.simplify(f1.subs({a: a_sol[0][a].subs(b, b_sol[0]), b: b_sol[0]}).subs(c, cv)) == 0:
            roots.add(sp.Integer(1))
        q = fR2.subs(c, cv)
        for r in sp.solve(q, x):
            if r.is_real and r > 1:
                roots.add(sp.nsimplify(r))
        return len(roots)
    disc = sp.discriminant(fR2, x)
    cands = [cv for cv in sp.solve(disc, c) if cv.is_real and cv > 0]
    good = [cv for cv in cands if count(cv) == 2]
    ans3 = load_answer(1, "3")
    side = [count(sp.nsimplify(good[0] * k)) for k in (sp.Rational(1, 2), sp.Rational(3, 2))] if good else []
    ok3 = len(good) == 1 and sp.simplify(good[0] - ans3) == 0 and 2 not in side
    report(1, ok1 and ok2 and ok3, f"a={a_sol[0][a]}, b={b_sol[0]}, c 후보={good} (c/2,3c/2 에서 해 개수 {side}), md=({ans1}; {ans2}; {ans3})")


def p02():
    """(1) 정적분을 sympy 로 직접 계산. (2) 조건을 만족하는 서로 다른 f 두 개로 f_n(1/2) 를 실제 반복 적분해 급수를 계산."""
    b, c = sp.symbols("b c", real=True)
    s = sp.Symbol("s", real=True)
    p = s**2 + b * s + c
    I1 = sp.integrate((p + sp.diff(p, s)) / 3 * sp.exp(s), (s, 0, x))
    ans1 = load_answer(2, "1")
    ok1 = sp.simplify(I1 - ans1.subs({sp.Symbol("x"): x, sp.Symbol("b"): b, sp.Symbol("c"): c})) == 0
    ans2 = load_answer(2, "2")
    sums = []
    for f0 in (6 * s, 12 * s**2, 3 * sp.sin(sp.pi * s) + (s - s)):
        fn, terms = f0, []
        for _ in range(4):   # f_1..f_4 를 실제로 적분
            fn = sp.integrate((fn + sp.diff(fn, s)) / 3 * sp.exp(s), (s, 0, x)).subs(x, s)
            terms.append(sp.simplify(fn.subs(s, sp.Rational(1, 2))))
        ratio = sp.simplify(terms[1] / terms[0])
        assert all(sp.simplify(terms[i + 1] / terms[i] - ratio) == 0 for i in range(3))
        assert sp.N(sp.Abs(ratio)) < 1
        sums.append(sp.simplify(terms[0] / (1 - ratio)))
    ok2 = all(sp.simplify(sm - ans2) == 0 for sm in sums)
    report(2, ok1 and ok2, f"(1) {sp.simplify(I1)}  (2) f 세 가지의 급수 합={sums}, md=({ans1}; {ans2})")


def p03():
    """(1) 원의 중심 (0,k) 가 P,Q,R,S 에서 같은 거리 ⇒ r 에 대한 두 식을 소거(종결식)해 a,c 관계를 얻음.
    (2) 사각형 PRQS 넓이를 c 에 대해 수치 최대화."""
    a, c, r = sp.symbols("a c r", real=True)
    g = -a * x**2 + c
    Ry = -1                                                    # P(1,1) 에서의 접선 y=2x-1 의 y절편
    Sy = sp.expand((g + sp.diff(g, x) * (0 - x)).subs(x, r))   # Q 에서의 접선의 y절편
    k = (Ry + Sy) / 2
    rho2 = ((Sy - Ry) / 2) ** 2
    e1 = sp.expand(1 + (1 - k) ** 2 - rho2)
    e2 = sp.expand(r**2 + (g.subs(x, r) - k) ** 2 - rho2)
    res = sp.factor(sp.resultant(sp.numer(sp.together(e1)), sp.numer(sp.together(e2)), r))
    rel = load_answer(3, "1").subs({sp.Symbol("a"): a, sp.Symbol("c"): c})
    # md 관계식이 종결식의 인수인지(나머지 0) 확인
    q, rem = sp.div(sp.Poly(res, a, c), sp.Poly(sp.expand(rel), a, c))
    ok1 = rem.is_zero

    def best(cv):
        av = sp.solve(rel.subs(c, cv), a)[0]
        rs = [z for z in sp.solve([e1.subs({a: av, c: cv}), e2.subs({a: av, c: cv})], r) if z.is_real and z < 0] \
            if False else [z for z in sp.solve(e1.subs({a: av, c: cv}), r) if z.is_real and z < 0 and abs(sp.N(e2.subs({a: av, c: cv, r: z}))) < 1e-20]
        if not rs:
            return None
        rv = rs[0]
        Q = (rv, g.subs({a: av, c: cv, x: rv}))
        pts = [(1, 1), (0, -1), Q, (0, Sy.subs({a: av, c: cv, r: rv}))]
        area = abs(sum(sp.N(pts[i][0] * pts[(i + 1) % 4][1] - pts[(i + 1) % 4][0] * pts[i][1]) for i in range(4))) / 2
        return float(area), av
    grid = [(best(sp.Rational(1, 4) + sp.Rational(5, 4) * sp.Rational(j, 200)), sp.Rational(1, 4) + sp.Rational(5, 4) * sp.Rational(j, 200)) for j in range(1, 200)]
    grid = [(v[0], v[1], cv) for v, cv in grid if v]
    m = max(grid, key=lambda z: z[0])
    ans2 = load_answer(3, "2")
    ok2 = abs(float(m[2]) - float(ans2["c"])) < 1e-9 and sp.simplify(m[1] - ans2["a"]) == 0
    report(3, ok1 and ok2, f"종결식={res}, 넓이 최대 c={m[2]}, a={m[1]}, 넓이={m[0]:.10f}, md=({load_answer(3,'1')}; {ans2})")


def p04():
    """(1) [증명-수치확인] 무작위 삼각형 50개에서 AB:AC = BP:PC.
    (2) AB=4, BC=5 로 좌표를 잡고 AC=b 에 대해 P,Q,R,S 를 작도해 넓이비 27 이 되는 b 를 수치로 찾음."""
    import random
    random.seed(1)

    def bis_point(A, B, C):   # ∠BAC 의 이등분선과 BC 의 교점(각을 직접 이등분하여 직선 교차)
        ab = mp.atan2(B[1] - A[1], B[0] - A[0])
        ac = mp.atan2(C[1] - A[1], C[0] - A[0])
        d = ac - ab
        while d > mp.pi:
            d -= 2 * mp.pi
        while d < -mp.pi:
            d += 2 * mp.pi
        m = ab + d / 2
        u = (mp.cos(m), mp.sin(m))
        # A + s u = B + w (C-B)
        det = u[0] * (-(C[1] - B[1])) - u[1] * (-(C[0] - B[0]))
        s_ = ((B[0] - A[0]) * (-(C[1] - B[1])) - (B[1] - A[1]) * (-(C[0] - B[0]))) / det
        return (A[0] + s_ * u[0], A[1] + s_ * u[1])
    dist = lambda P, Q: mp.sqrt((P[0] - Q[0]) ** 2 + (P[1] - Q[1]) ** 2)
    ok1 = True
    for _ in range(50):
        A, B, C = [(mp.mpf(random.uniform(-5, 5)), mp.mpf(random.uniform(-5, 5))) for _ in range(3)]
        P = bis_point(A, B, C)
        ok1 = ok1 and abs(dist(A, B) / dist(A, C) - dist(B, P) / dist(P, C)) < 1e-20

    def inter(P0, d0, P1, P2):   # P0 + s d0 와 선분 P1P2 의 교점
        e = (P2[0] - P1[0], P2[1] - P1[1])
        det = d0[0] * (-e[1]) - d0[1] * (-e[0])
        s_ = ((P1[0] - P0[0]) * (-e[1]) - (P1[1] - P0[1]) * (-e[0])) / det
        return (P0[0] + s_ * d0[0], P0[1] + s_ * d0[1])
    area = lambda P, Q, R: abs((Q[0] - P[0]) * (R[1] - P[1]) - (R[0] - P[0]) * (Q[1] - P[1])) / 2

    def ratio(bv):
        B, C = (mp.mpf(0), mp.mpf(0)), (mp.mpf(5), mp.mpf(0))
        xa = (16 - bv**2 + 25) / 10
        A = (xa, mp.sqrt(16 - xa**2))
        P = bis_point(A, B, C)
        Q = inter(P, (A[0] - B[0], A[1] - B[1]), A, C)
        R = bis_point(Q, P, C)
        S = inter(R, (Q[0] - P[0], Q[1] - P[1]), A, C)
        return area(A, B, P) / area(Q, R, S), A
    bsol = mp.findroot(lambda bv: ratio(bv)[0] - 27, (mp.mpf("1.2"), mp.mpf("3.5")), solver="bisect")
    A = ratio(bsol)[1]
    S_ABC = 5 * A[1] / 2
    ans2 = load_answer(4, "2")
    ok2 = abs(S_ABC - mp.mpf(str(sp.N(ans2, 30)))) < 1e-15
    report(4, ok1 and ok2, f"[증명-수치확인] (1) 50개 삼각형 통과={ok1}; (2) AC={mp.nstr(bsol, 15)}, 넓이={mp.nstr(S_ABC, 15)}, md={ans2}")


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
