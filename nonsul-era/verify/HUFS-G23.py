#!/usr/bin/env python3
"""한국외대 2023학년도 논술전형(자연계) 1~10 (HUFS-G23-01~10) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산해 YAML answer.value 와 비교.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "HUFS-G23"
WIDTH = 2
RESULTS = []
x, t, n, r = sp.symbols("x t n r", real=True)
a, b, c, k = sp.symbols("a b c k", real=True)
SYM = {"x": x, "t": t, "n": n, "a": a, "b": b, "c": c, "k": k, "pi": sp.pi}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0, raw=False):
    """yaml 의 subparts[sub].answer.value 를 읽는다(raw=True 면 문자열 그대로)."""
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/HUFS/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    v = d["subparts"][sub]["answer"]["value"]
    return v if raw else sp.sympify(v, locals=SYM)


# ── 문항별 검산 함수(p01, p02, ...)는 아래에 덧붙인다 ──


def p01():
    """판별식 ≤ 0 인 θ∈[0,π] 집합을 sympy 로 풀고, 격자로 원 부등식(모든 x)도 재확인."""
    th = sp.symbols("theta", real=True)
    D = (4 * sp.cos(th))**2 - 4
    S = sp.solveset(D <= 0, th, sp.Interval(0, sp.pi))
    al, be = S.inf, S.sup
    import math
    def holds(tv):  # 이차식 최솟값 1-4cos^2 ≥ 0
        return 1 - 4 * math.cos(tv)**2 >= -1e-12
    grid_ok = all(holds(math.pi * i / 3000) == (float(al) - 1e-9 <= math.pi * i / 3000 <= float(be) + 1e-9)
                  for i in range(3001))
    ans = load_answer(1)
    report(1, sp.simplify(al + be - ans) == 0 and grid_ok, f"θ범위={S}, α+β={sp.simplify(al+be)}, 격자일치 {grid_ok}, md={ans}")


def p02():
    """f=2x^3+ax^2+bx+c 일반형에서 (가)·(나) 극한을 sympy limit 로 세워 a,b 를 풀고 f'(2) 계산."""
    A, B, C = sp.symbols("A B C")
    f = 2 * x**3 + A * x**2 + B * x + C
    l1 = sp.limit(f / (2 * x**3 + 5 * x + 1), x, sp.oo)
    fp = sp.diff(f, x)
    s1 = sp.solve(fp.subs(x, 0), B)              # (나): 극한 존재 ⇒ f'(0)=0
    fp1 = fp.subs(B, s1[0])
    s2 = sp.solve(sp.Eq(sp.limit(fp1 / x, x, 0), 2), A)
    val = fp1.subs(A, s2[0]).subs(x, 2)
    ans = load_answer(2)
    report(2, l1 == 1 and val == ans, f"(가) 극한={l1}, b={s1}, a={s2}, f'(2)={val}, md={ans}")


def p03():
    """PQ^2 의 [-1,3] 최대·최소: 임계점+끝점(sympy) 과 격자 4001점 탐색을 함께 비교."""
    f = x**2 + (x**2 / 4 - 3)**2
    crit = [cc for cc in sp.solve(sp.diff(f, x), x) if -1 <= cc <= 3]
    vals = [f.subs(x, v) for v in crit + [-1, 3]]
    M, m = max(vals), min(vals)
    import math
    g = [float(f.subs(x, -1 + 4 * sp.Rational(i, 4000))) for i in range(4001)]
    ok_grid = abs(max(g) - float(M)) < 1e-9 and abs(min(g) - float(m)) < 1e-6
    ans = load_answer(3)
    report(3, sp.simplify(M * m - ans) == 0 and ok_grid, f"임계점 {crit}, a^2={M}, b^2={m}, 곱={M*m}, 격자확인 {ok_grid}, md={ans}")


def p04():
    """f=x^3+ax^2+bx+8 에서 (다) 를 f(1)=3, f'(1)=0 으로 바꾸고, 실제로 g 가 존재함을 확인(g 를 구성)."""
    A, B = sp.symbols("A B")
    f = x**3 + A * x**2 + B * x + 8
    sol = sp.solve([sp.Eq(f.subs(x, 1), 3), sp.Eq(sp.diff(f, x).subs(x, 1), 0)], [A, B], dict=True)[0]
    F = f.subs(sol)
    # 존재성: G(x)=∫_1^x g 라 두면 F'(x) = G + 2(x-1)G'  → G 를 다항식으로 풀어 g=G' 이 연속인지 확인
    G = sp.Function("G")
    ode = sp.Eq(sp.diff(F, x), G(x) + 2 * (x - 1) * G(x).diff(x))
    cs = sp.symbols("c0:4")
    Gp = sum(cs[i] * (x - 1)**i for i in range(4))
    eqs = sp.Poly(sp.expand(sp.diff(F, x) - Gp - 2 * (x - 1) * sp.diff(Gp, x)), x).all_coeffs()
    gs = sp.solve(eqs + [Gp.subs(x, 1)], cs, dict=True)
    exist = bool(gs)
    if exist:
        g = sp.diff(Gp.subs(gs[0]), x).subs(t, t)
        chk = sp.simplify(3 + sp.integrate((x - 2 + t) * g.subs(x, t), (t, 1, x)) - F) == 0
    else:
        chk = False
    ans = load_answer(4)
    val = F.subs(x, 2)
    report(4, val == ans and chk, f"a,b={sol}, f={F}, g 구성 후 (다) 성립 {chk}, f(2)={val}, md={ans}")


def p05():
    """n=1..200 전수: (i) X^2-2^{10/n}X+1=0 의 양의 실근 X=2^r 존재, (ii) 4^r+4^{-r}=X^2+X^{-2} 가 자연수 — 정확 계산."""
    good = []
    for nn in range(1, 201):
        s = sp.Integer(2)**sp.Rational(10, nn)
        disc = s**2 - 4
        if disc < 0:
            continue
        roots = [(s + sp.sqrt(disc)) / 2, (s - sp.sqrt(disc)) / 2]
        ok = False
        for X in roots:
            if X > 0:
                v = sp.nsimplify(sp.simplify(X**2 + X**-2))
                if v.is_integer and v > 0:
                    ok = True
        if ok:
            good.append(nn)
    ans = load_answer(5)
    report(5, sum(good) == ans, f"n=1..200 전수: 조건 만족 n={good}, 합={sum(good)}, md={ans} (오답 재현: 20 포함 시 {sum(good)+20})")


def p06():
    """정수 m,n∈[-30,30], α 격자(1/100)에서 (다)(라)(마)를 만족하는 경우를 만들고 log(ab) 의 최대·최소를 직접 계산."""
    logs = []
    for m in range(-30, 31):
        for nn in range(-30, 31):
            if m <= 0 or nn <= 0 or m * nn != 18:      # (다) 제1사분면 P, (라) n=18/m
                continue
            for i in range(1, 100):
                al = sp.Rational(i, 100)
                be = 1 - al                            # (마)
                if not (0 < be < 1):                   # (다) 제1사분면 Q
                    continue
                logs.append((m + al) + (nn + be))       # log a + log b = log ab
    kk = max(logs) + min(logs)
    ans = load_answer(6)
    report(6, kk == ans, f"log(ab) 최대={max(logs)}, 최소={min(logs)}, k={kk}, md={ans}")


def p07():
    """교점을 sympy 로 풀고 ∫|f-g| 를 교점·꺾인 점으로 나눠 정확 계산 + 위아래 판정 없이 수치 적분으로 재확인."""
    f = -4 * x * (x + 3) * (x - 3)
    gl, gr = 16 * x + 48, -16 * x + 48
    roots = sorted(set([rr for rr in sp.solve(f - gl, x) if rr <= 0] + [rr for rr in sp.solve(f - gr, x) if rr >= 0]))
    lo, hi = roots[0], roots[-1]
    pts = sorted(set(roots + [sp.Integer(0)]))
    S = 0
    for p, q in zip(pts, pts[1:]):
        g = gl if q <= 0 else gr
        S += sp.Abs(sp.integrate(f - g, (x, p, q)))   # 각 구간에서 부호가 일정(교점 사이)
    import mpmath
    num = float(mpmath.quad(lambda xx: abs(float(-4 * xx * (xx + 3) * (xx - 3)) - (-16 * abs(xx) + 48)), [float(v) for v in pts]))
    ans = load_answer(7)
    report(7, S == ans and abs(num - float(ans)) < 1e-9, f"교점 x={roots}, 넓이={S}, 수치={num:.10f}, md={ans}")


def p08():
    """a_n=a1 r^(n-1): (나)÷(다) 로 a1 을 소거한 방정식을 풀고(r=0,±1 은 (다)에서 90≠0 이라 제외),
    각 r 에 대해 a1 을 (나)에서 정확히 구해 (다)를 재확인하고 (가) a1<0 인 것만 남긴다."""
    R = sp.symbols("R", real=True)
    cand = [rv for rv in sp.solve(sp.Eq(9 * (R**4 - 1), 13 * R**2 * (R**2 - 1)), R) if rv.is_real]
    good, info = [], []
    for rv in cand:
        if rv**2 * (rv**2 - 1) == 0:
            continue
        a1 = sp.Rational(130) / (rv**2019 * (rv**4 - 1))
        chk = sp.simplify(a1 * rv**2021 * (rv**2 - 1) - 90) == 0
        info.append((rv, "a1<0" if a1 < 0 else "a1>0", chk))
        if chk and a1 < 0:
            good.append(rv)
    ans = load_answer(8)
    report(8, good == [ans], f"후보 {info}, (가) 만족 r={good}, md={ans}")


def p09():
    """연산 ⊙ 을 구현해, 대소가 바뀌는 점(sin=cos, cos=1-cos)을 sympy 로 구해 [π/4,π/2] 를 나누고
    각 구간에서 고른 조각 방정식을 solveset 으로 풀어 합침. 격자 20001점에서 좌변-우변 영점도 재확인."""
    th = sp.symbols("theta", real=True)
    s, c = sp.sin(th), sp.cos(th)
    J = sp.Interval(sp.pi / 4, sp.pi / 2)
    cuts = set(sp.solveset(s - c, th, J)) | set(sp.solveset(c - (1 - c), th, J)) | {J.inf, J.sup}
    cuts = sorted(cuts, key=lambda v: float(v))
    sols = set()
    for p, q in zip(cuts, cuts[1:]):
        mid = (p + q) / 2
        def br(A, B):
            return (A - B) if float((A - B).subs(th, mid)) >= 0 else sp.Integer(0)
        expr = br(s, c) + br(c, 1 - c) - s
        for cand in sp.solveset(expr, th, sp.Interval(p, q)):
            # 경계점이면 실제 정의로 다시 확인
            A1, B1 = s.subs(th, cand), c.subs(th, cand)
            val = (A1 - B1 if A1 >= B1 else 0) + (B1 - (1 - B1) if B1 >= 1 - B1 else 0) - A1
            if sp.simplify(val) == 0:
                sols.add(cand)
    import math
    def Fn(v):
        a_, b_ = math.sin(v), math.cos(v)
        o = lambda A, B: A - B if A >= B else 0.0
        return o(a_, b_) + o(b_, 1 - b_) - a_
    N = 20000
    near = [math.pi / 4 + (math.pi / 4) * i / N for i in range(N + 1)]
    near = [v for v in near if abs(Fn(v)) < 1e-12]
    ans = load_answer(9)
    ansset = set(ans) if isinstance(ans, sp.FiniteSet) else {ans}
    ok = sols == ansset and near and all(abs(v - math.pi / 2) < 1e-9 for v in near)
    report(9, ok, f"분할점 {cuts}, 해={sols}, 격자 영점 {len(near)}개(π/2), md={ans}")


def p10():
    """[증명-수치확인] 정수 a_n 을 n=1..12 까지 정확 계산해 3^n | a_n^2+2 확인 + 귀납 단계 항등식
    (3^k b-2)(3^k b+1)^2+2 = 3^{k+1}(3^{2k-1}b^3-b) 를 sympy 로 확인(k=1..30 대입 및 기호 전개)."""
    assert load_answer(10, raw=True) is None          # 증명 문항: answer.value 는 null
    seq, av = [], 1
    for nn in range(1, 13):
        seq.append((av * av + 2) % 3**nn == 0)
        av = av**3 + 3 * av
    kk, bb = sp.symbols("k b", positive=True, integer=True)
    lhs = (3**kk * bb - 2) * (3**kk * bb + 1)**2 + 2
    rhs = 3**(kk + 1) * (3**(2 * kk - 1) * bb**3 - bb)
    ident = sp.simplify(sp.expand(lhs - rhs, power_exp=True)) == 0
    ident_k = all(sp.expand(lhs.subs(kk, kv) - rhs.subs(kk, kv)) == 0 for kv in range(1, 31))
    step = sp.expand((x**3 + 3 * x)**2 - x**2 * (x**2 + 3)**2) == 0
    report(10, all(seq) and ident_k and step, f"[증명-수치확인] n=1..12 모두 배수 {all(seq)}, 귀납 항등식 기호 {ident}/k=1..30 {ident_k}, a_(k+1)^2 변형 {step}")


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
