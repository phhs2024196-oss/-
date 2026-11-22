#!/usr/bin/env python3
"""한국외대 2026학년도 논술전형 모의논술(자연계) 문제 1~7 (HUFS-M26-01~07) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "HUFS-M26"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
a, b, k = sp.symbols("a b k", real=True)
SYM = {"x": x, "t": t, "n": n, "a": a, "b": b, "k": k}


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


def p01():
    """등차수열 첫째항 A, 공차 D 를 자연수 범위 전수 탐색: 모든 항 자연수, (b2-b1)(b3-b2)=70 → b5."""
    sols = set()
    for A in range(1, 80):
        for D in range(0, 80):
            an = lambda m: A + (m - 1) * D
            S = lambda m: sum(an(i) for i in range(1, m + 1))
            bn = lambda m: S(m + 2) - an(m + 1)
            if (bn(2) - bn(1)) * (bn(3) - bn(2)) == 70:
                sols.add((A, D, bn(5)))
    ans = load_answer(1)
    # 일반항 기호 확인: b_{n+1}-b_n = a_{n+2}
    A_, D_ = sp.symbols("A D")
    anS = lambda m: A_ + (m - 1) * D_
    SS = lambda m: m * (2 * A_ + (m - 1) * D_) / 2
    diff_ok = sp.simplify((SS(n + 3) - anS(n + 2)) - (SS(n + 2) - anS(n + 1)) - anS(n + 2)) == 0
    ok = diff_ok and len(sols) == 1 and list(sols)[0][2] == ans
    report(1, ok, f"b_(n+1)-b_n=a_(n+2) 항등식 {diff_ok}; (a,d,b5) 해 {sorted(sols)}, md={ans} (오답 재현: 공차 무시 a3=1,a4=70 등은 a2<0)")


def p02():
    """u=ln a, v=ln b (>0) 로 두고 근과 계수의 관계를 연립해 풀고, 원래 방정식의 두 근이 log_a b, log_b a 인지 대입 확인."""
    u, v = sp.symbols("u v", positive=True)
    L2 = sp.log(2)
    r1, r2 = v / u, u / v                                  # log_a b, log_b a
    eqs = [sp.Eq(r1 + r2, 5 * (v - L2) / u), sp.Eq(r1 * r2, 2 * u / v)]
    sols = sp.solve(eqs, [u, v], dict=True)
    vals = []
    for s in sols:
        A, B = sp.exp(s[u]), sp.exp(s[v])
        if A > 1 and B > 1:
            X = sp.Symbol("X")
            quad = X**2 - 5 * (sp.log(B) - L2) / sp.log(A) * X + 2 * sp.log(A) / sp.log(B)
            chk = all(sp.simplify(quad.subs(X, r)) == 0 for r in (sp.log(B) / sp.log(A), sp.log(A) / sp.log(B)))
            vals.append((sp.nsimplify(sp.simplify(A**3)), sp.nsimplify(sp.simplify(B**3)), chk))
    ans = load_answer(2)
    ok = len(vals) == 1 and vals[0][2] and sp.simplify(vals[0][0] + vals[0][1] - ans) == 0
    report(2, ok, f"(a^3,b^3,근 대입확인)={vals}, md={ans}")


def p03():
    """sin^2x=s, sin^2y=r 연립 → [0,π] 해 집합을 solveset 으로 구하고 모든 (x,y) 조합을 원 조건에 대입해 tan(x+y) 최댓값.
    추가로 y 를 소거한 잔차 함수를 x 격자에서 조사해 다른 해가 없음을 수치 확인."""
    X, Y = sp.symbols("X Y", real=True)
    s, r = sp.symbols("s r", nonnegative=True)
    sr = sp.solve([s + r - sp.Rational(1, 2), s * r - sp.Rational(1, 16)], [s, r], dict=True)
    xs = set()
    for d in sr:
        xs |= set(sp.solveset(sp.sin(X)**2 - d[s], X, sp.Interval(0, sp.pi)))
    pairs = [(p, q) for p in xs for q in xs
             if sp.simplify(sp.cos(p)**2 + sp.cos(q)**2 - sp.Rational(3, 2)) == 0
             and sp.simplify(sp.sin(p) * sp.sin(q) - sp.Rational(1, 4)) == 0]
    tans = [sp.tan(p + q) for p, q in pairs if sp.cos(p + q) != 0]
    mx = max(tans, key=lambda e: float(e))
    import math
    # 독립 확인: (다)에서 sin y = 1/(4 sin x) 로 y 를 소거하면 (나)의 잔차 h(x)=1/2-sin^2x-1/(16 sin^2x).
    # x 격자(0,π) 20000점에서 h ≤ 0 이고, h 가 0 에 가까운 점은 π/6, 5π/6 근처뿐인지 확인.
    N = 20000
    hs = [(math.pi * i / N, 0.5 - math.sin(math.pi * i / N)**2 - 1 / (16 * math.sin(math.pi * i / N)**2)) for i in range(1, N)]
    near = [xv for xv, h in hs if h > -1e-4]
    covered = max(h for _, h in hs) < 1e-12 and all(min(abs(xv - math.pi / 6), abs(xv - 5 * math.pi / 6)) < 0.05 for xv in near)
    ans = load_answer(3)
    ok = sp.simplify(mx - ans) == 0 and covered and len(pairs) == 4
    report(3, ok, f"(s,r)={sr}, 해 {len(pairs)}쌍 {sorted(pairs, key=lambda z: (float(z[0]), float(z[1])))}, tan 값 {sorted(set(tans), key=float)}, y 소거 잔차 h≤0·근방점 {len(near)}개 π/6·5π/6 근처 {covered}, md={ans}")


def p04():
    """f=(x-α)(x-1)(x-c)(극솟값 0 → f(α)=f(1)=0). c 를 미지수로 두고 f'(0)=15 를 c=α, c=1 각각에서 풀고,
    각 후보에서 |f| 의 극소점 집합을 직접 계산(근 + f'=0 인 점, 양옆 값 비교)해 정확히 {α,1} 인지 확인.
    c∉{α,1} 이면 c 도 |f| 의 극소점이 됨을 무작위 20개 표본으로 확인."""
    al, c = sp.symbols("alpha c", real=True)

    def local_min_points(f):
        cands = set(sp.real_roots(sp.Poly(f, x))) | set(sp.real_roots(sp.Poly(sp.diff(f, x), x)))
        mins = set()
        g = lambda v: abs(f.subs(x, v))
        for p0 in cands:
            e = sp.Rational(1, 10**6)
            if g(p0 - e) > g(p0) and g(p0 + e) > g(p0):
                mins.add(sp.nsimplify(p0))
        return mins, {sp.nsimplify(p0): g(p0) for p0 in mins}

    good = []
    for cc in (al, sp.Integer(1)):
        f = sp.expand((x - al) * (x - 1) * (x - cc))
        for av in sp.solve(sp.Eq(sp.diff(f, x).subs(x, 0), 15), al):
            if av == 1:
                continue
            fv = f.subs(al, av)
            mins, vals = local_min_points(fv)
            if mins == {av, 1} and all(v == 0 for v in vals.values()):
                good.append(av)
    import random
    random.seed(1)
    bad_ok = True
    for _ in range(20):
        av = sp.Rational(random.randint(-90, 90), 7)
        if av == 1:
            continue
        f = sp.expand((x - av) * (x - 1) * (x - c))
        cs = sp.solve(sp.Eq(sp.diff(f, x).subs(x, 0), 15), c)
        for cv in cs:
            if cv in (av, 1):
                continue
            mins, _ = local_min_points(f.subs(c, cv))
            bad_ok &= cv in mins
    ans = load_answer(4)
    ok = sp.simplify(sum(good) - ans) == 0 and bad_ok
    report(4, ok, f"α 후보 {sorted(good)} 합 {sum(good)}; c∉{{α,1}} 표본에서 c 도 극소 {bad_ok}, md={ans}")


def p05():
    """g(x)=f(x)f(x-a) 의 x=a 에서 좌·우극한과 함숫값을 계산. (sympy limit 는 Piecewise 경계 우극한을 틀리게 주므로
    조각 식을 직접 골라 한쪽 극한을 구한다: x→p+ 이면 p<0 일 때만 왼쪽 식, x→p- 이면 p≤0 일 때 왼쪽 식.)
    a<0, a>0 은 기호로 풀고 a=0 은 따로 판정, 격자 a∈[-20,20](0.05 간격)로 교차 확인."""
    left = lambda X: -2 * (X + 1)
    right = lambda X: sp.Rational(1, 2) * X - 5
    fval = lambda p_: left(p_) if p_ <= 0 else right(p_)
    flim = lambda p_, side: left(p_) if (p_ < 0 or (p_ == 0 and side == "-")) else right(p_)

    def cont_at(av):
        R = flim(av, "+") * flim(0, "+")
        L = flim(av, "-") * flim(0, "-")
        V = fval(av) * fval(sp.Integer(0))
        return R == V and L == V

    A = sp.Symbol("A", real=True)
    sols = set()
    for expr, dom in ((left(A), sp.Interval.open(-sp.oo, 0)), (right(A), sp.Interval.open(0, sp.oo))):
        R, L, V = expr * flim(0, "+"), expr * flim(0, "-"), expr * fval(sp.Integer(0))
        sols |= set(sp.solveset(R - V, A, dom).intersect(sp.solveset(L - V, A, dom)))
    zero_ok = cont_at(sp.Integer(0))
    if zero_ok:
        sols.add(sp.Integer(0))
    grid = [sp.Rational(i, 20) for i in range(-400, 401)]
    gsols = {gv for gv in grid if cont_at(gv)}
    ans = load_answer(5)
    ok = sp.simplify(sum(sols) - ans) == 0 and gsols == sols
    report(5, ok, f"a=0 연속 {zero_ok}(우극한 {flim(0,'+')**2}, 좌극한 {flim(0,'-')**2}); 해 {sorted(sols)} 합 {sum(sols)}, 격자 {len(grid)}점 연속인 a {sorted(gsols)}, md={ans}")

def p06():
    """f=x^2+Bx+C(lim f/x^2=1). 세 적분이 같다는 두 등식을 그대로 연립해 B,C 를 a 로 표현,
    f(-1) 을 a>0 에서 최대화(도함수 0 + 경계 극한)하고, a 격자 탐색으로 교차 확인."""
    B, C = sp.symbols("B C", real=True)
    A = sp.Symbol("A", positive=True)
    f = x**2 + B * x + C
    I = lambda lo, hi: sp.integrate(f, (x, lo, hi))
    sol = sp.solve([sp.Eq(I(-A, 2 * A), I(0, 2 * A)), sp.Eq(I(0, 2 * A), I(-A, 0))], [B, C], dict=True)
    assert len(sol) == 1
    F = sp.expand((1 - B + C).subs(sol[0]))
    crit = [c0 for c0 in sp.solve(sp.diff(F, A), A) if c0.is_positive]
    Fmax = max(F.subs(A, c0) for c0 in crit)
    sup_ok = sp.limit(F, A, 0, "+") <= Fmax and sp.limit(F, A, sp.oo) <= Fmax
    best = max(float(F.subs(A, sp.Rational(i, 1000))) for i in range(1, 5000))
    ans = load_answer(6)
    ok = sp.simplify(Fmax - ans) == 0 and sup_ok and abs(best - float(ans)) < 1e-6
    report(6, ok, f"B={sol[0][B]}, C={sol[0][C]}, f(-1)={F}, 임계 a={crit}, 최댓값={Fmax}, 격자 최대≈{best:.8f}, md={ans}")


def p07():
    """[증명-수치확인] a_n=21^n+5^n+14: n=1..200 에서 40 의 배수, 귀납 단계 항등식 a_{k+1}-a_k=20(21^k+5^{k-1}) 기호 확인,
    21^k+5^{k-1} 짝수(k=1..200) 확인. 증명 문항이므로 answer.kind 가 proof 인지도 확인."""
    kk = sp.Symbol("kk", integer=True, positive=True)
    an = lambda m: 21**m + 5**m + 14
    base = an(1) == 40
    allmult = all(an(m) % 40 == 0 for m in range(1, 201))
    ident = sp.simplify(sp.expand_power_base(an(kk + 1) - an(kk) - 20 * (21**kk + 5**(kk - 1)))) == 0 \
        or all(an(m + 1) - an(m) == 20 * (21**m + 5**(m - 1)) for m in range(1, 201))
    even = all((21**m + 5**(m - 1)) % 2 == 0 for m in range(1, 201))
    kind = load_answer(7, raw=True) is None
    ok = base and allmult and ident and even and kind
    report(7, ok, f"[증명-수치확인] a_1=40 {base}, n=1..200 모두 40의 배수 {allmult}, 귀납 항등식 {ident}, 21^k+5^(k-1) 짝수 {even}, answer.kind=proof(value null) {kind}")


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
