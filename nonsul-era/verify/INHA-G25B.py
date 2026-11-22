#!/usr/bin/env python3
"""INHA-G25B (2025학년도 인하대 자연계열 논술 오후) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "INHA-G25B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")




def p01():
    """정사각형 좌표로 P,Q 위치를 직접 잡아 넓이를 계산(구간식과 독립) → 구간식 대조, 극값·최대최소 수치 판정."""
    import mpmath as mp
    mp.mp.dps = 30
    pi = mp.pi

    def point_on_boundary(th):          # A=(0,0), B=(10,0), C=(10,10), D=(0,10); 각 th 방향 반직선과 BC 또는 CD 의 교점
        if th <= pi / 4:
            return (mp.mpf(10), 10 * mp.tan(th))
        return (10 / mp.tan(th), mp.mpf(10))

    def S_geo(xv):
        P, Q = point_on_boundary(xv), point_on_boundary(xv + pi / 6)
        return abs(P[0] * Q[1] - P[1] * Q[0]) / 2   # 신발끈 공식

    X = sp.Symbol("X", real=True)
    pieces = [(25 / (sp.cos(X) * sp.cos(X + sp.pi / 6)), 0, sp.pi / 12),
              (25 / (sp.cos(X) * sp.cos(sp.pi / 3 - X)), sp.pi / 12, sp.pi / 4),
              (25 / (sp.sin(X) * sp.cos(sp.pi / 3 - X)), sp.pi / 4, sp.pi / 3)]      # YAML (1-1) 정답
    ok1 = True
    for e, lo, hi in pieces:
        fe = sp.lambdify(X, e, "mpmath")
        for k in range(0, 51):
            xv = mp.mpf(str(sp.N(lo, 40))) + (mp.mpf(str(sp.N(hi, 40))) - mp.mpf(str(sp.N(lo, 40)))) * k / 50
            ok1 &= abs(fe(xv) - S_geo(xv)) < mp.mpf("1e-15")
    # (1-2) S_geo 격자에서 국소 극대·극소
    N = 6000
    xs = [pi / 3 * k / N for k in range(N + 1)]
    ys = [S_geo(v) for v in xs]
    mx = [xs[i] for i in range(1, N) if ys[i] > ys[i - 1] and ys[i] > ys[i + 1]]
    mn = [xs[i] for i in range(1, N) if ys[i] < ys[i - 1] and ys[i] < ys[i + 1]]
    near = lambda L, E: len(L) == len(E) and all(abs(a - b) < 1e-3 for a, b in zip(L, E))
    ok2 = near(mx, [pi / 12, pi / 4]) and near(mn, [pi / 6])
    # 도함수 부호 기호 확인(구간 2): sympy diff 분자 = 25 sin(2x-π/3) 꼴
    d2 = sp.simplify(sp.diff(pieces[1][0], X) * (sp.cos(X) * sp.cos(sp.pi / 3 - X)) ** 2 - 25 * sp.sin(2 * X - sp.pi / 3))
    ok2 &= d2 == 0
    imax = max(range(N + 1), key=lambda i: ys[i])
    vmin = min(ys)
    mins = [xs[i] for i in range(N + 1) if ys[i] - vmin < mp.mpf("1e-12")]
    ok3 = (abs(xs[imax] - pi / 12) < 1e-3 or abs(xs[imax] - pi / 4) < 1e-3) and abs(S_geo(pi / 12) - S_geo(pi / 4)) < 1e-20 \
        and near(mins, [0, pi / 3]) and abs(S_geo(pi / 6) - mp.mpf(100) / 3) < 1e-20 and abs(vmin - 50 / mp.sqrt(3)) < 1e-20
    report(1, ok1 and ok2 and ok3,
           f"(1-1) 구간식=좌표 넓이 153점 (1-2) 극대 {[mp.nstr(v / pi, 6) for v in mx]}π 극소 {[mp.nstr(v / pi, 6) for v in mn]}π (1-3) 최소 x={[mp.nstr(v / pi, 4) for v in mins]}π, S(π/6)=100/3 > 50/√3")


def p02():
    """(2-1) 접할 조건 연립으로 b(a) (2-2) 접선 y절편 (2-3) 두 넓이 기호 적분 → 비 1:1 [증명-수치확인]."""
    a, b, X = sp.symbols("a b X", positive=True)
    F = a / 3 * X**2 + 1
    G = b * sp.sqrt(X)
    sol = sp.solve([sp.Eq(F, G), sp.Eq(sp.diff(F, X), sp.diff(G, X))], [X, b], dict=True)
    sol = [d for d in sol if d[X].is_positive is not False]
    bx, xp = sp.simplify(sol[0][b]), sp.simplify(sol[0][X])
    ok1 = len(sol) == 1 and sp.simplify(bx - sp.Rational(4, 3) * a ** sp.Rational(1, 4)) == 0
    T = sp.expand(F.subs(X, xp) + sp.diff(F, X).subs(X, xp) * (X - xp))
    y0 = sp.simplify(T.subs(X, 0))
    ok2 = y0 == sp.Rational(2, 3) and sp.diff(y0, a) == 0
    up = sp.simplify(sp.integrate(F - T, (X, 0, xp)))
    lo = sp.simplify(sp.integrate(T - G.subs(b, bx), (X, 0, xp)))
    # 위·아래 위치 관계 수치 확인(구간 내부에서 F ≥ T ≥ G)
    okpos = all(float((F - T).subs({a: av, X: xv})) >= -1e-12 and float((T - G.subs(b, bx)).subs({a: av, X: xv})) >= -1e-12
                for av in (0.3, 1, 7) for xv in [k / 20 / av**0.5 for k in range(21)])
    ok3 = sp.simplify(up / lo) == 1 and sp.simplify(up - 1 / (9 * sp.sqrt(a))) == 0 and okpos
    report(2, ok1 and ok2 and ok3, f"(2-1) b={bx}, P_x={xp} (2-2) 접선 {T}, y절편 {y0} (2-3) 위 {up}, 아래 {lo}, 비 1:1 [증명-수치확인]")


def p03():
    """(3-1) 연속 조건 기호 풀이 (3-2) a 격자에서 f=1/2 해 개수 직접 계산 (3-3) a 격자에서 β 존재 여부 직접 판정."""
    import math
    A, C = sp.symbols("A C", positive=True)
    Bs = sp.Symbol("Bs", positive=True)               # Bs = b^2
    eqs = [sp.Eq(sp.log(2) / sp.log(A), Bs), sp.Eq(sp.log(4) / sp.log(C), Bs**2)]
    Csol = sp.solve(eqs, [C, Bs], dict=True)[0][C]
    v1 = sp.simplify(sp.log(A) ** 2 / sp.log(Csol))
    ok1 = sp.simplify(v1 - sp.log(2) / 2) == 0

    def params(a):
        b = math.sqrt(math.log(2) / math.log(a))
        c = math.exp(2 * math.log(a) ** 2 / math.log(2))
        return b, c

    def f(x, a):
        b, c = params(a)
        if x < 2:
            return math.log(x) / math.log(a)
        if x < 4:
            return b**x
        return math.log(x) / math.log(c)

    def level_set(t, a):                             # f(x)=t 의 해(구간별 역함수로 직접)
        b, c = params(a)
        out = []
        x1 = a**t
        if 0 < x1 < 2:
            out.append(x1)
        if abs(b - 1) > 1e-15 and t > 0:
            x2 = math.log(t) / math.log(b)
            if 2 <= x2 < 4:
                out.append(x2)
        elif abs(t - 1) < 1e-15:
            out.append(float("nan"))                 # 상수 구간(무수히 많음) 표시
        x3 = c**t if t * math.log(c) < 700 else float("inf")
        if x3 >= 4:
            out.append(x3)
        return out

    grid = [1.05 + 0.005 * i for i in range(1, 1400)] + [2 ** math.sqrt(2) + e for e in (-1e-9, 1e-9)] + [4 - 1e-9, 4 + 1e-9, 4.0]
    ok2 = all((len(level_set(0.5, a)) == 3) == (2 ** math.sqrt(2) < a < 4) for a in grid)

    def exists_beta(a):
        t = f(8, a)
        for k in level_set(t, a):
            if k != k or abs(k - 8) < 1e-9:
                continue
            if any(bt > 1 + 1e-12 for bt in level_set(k, a)):
                return True
        return False
    r = 2 * math.sqrt(2)
    grid3 = [1.05 + 0.01 * i for i in range(1, 900)] + [r - 1e-9, r + 1e-9]
    ok3 = all(exists_beta(a) == (a >= r - 1e-12) for a in grid3)
    # 경계 a=2√2: f(2)=f(8) 이 정확히 성립(등호 포함 근거)
    b0, c0 = params(r)
    ok3 &= abs(math.log(2) / math.log(r) - math.log(8) / math.log(c0)) < 1e-12
    report(3, ok1 and ok2 and ok3, f"(3-1) {v1} (3-2) 해 3개 ⇔ 2^√2<a<4 ({len(grid)}점) (3-3) β 존재 ⇔ a≥2√2 ({len(grid3)}점), a=2√2 에서 f(2)=f(8)")


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
