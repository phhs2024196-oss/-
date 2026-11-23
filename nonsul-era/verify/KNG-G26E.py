import sys, pathlib, itertools, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import numpy as np
from sympy.parsing.sympy_parser import parse_expr
V = Verifier("KNG-G26E", "items/KNG/PAST")
P = lambda s: parse_expr(s.replace("^", "**"), local_dict={"x": x, "pi": sp.pi})
def kv(s):  # "a=1;b=2" -> dict
    return dict(p.split("=", 1) for p in s.replace(" ", "").split(";") if "=" in p)

def p04():
    xs = np.linspace(-1, 0, 4001)
    mn = lambda a: float(np.min(a ** (-xs) + 2 * a))
    # a>0, a!=1 에서 min=2 가 되는 a 를 격자 부호 변화로 전수 탐색
    grid = [i / 1000 for i in range(1, 20000) if i != 1000]
    g = [mn(a) - 2 for a in grid]
    sols = [grid[i] for i in range(len(g) - 1) if g[i] * g[i + 1] < 0 and abs(grid[i + 1] - grid[i]) < 0.0011]
    V.check(4, "수치 해 개수", len(sols), 1)
    V.check(4, "수치 해 a", sols[0], P(V.ans(4, "1", "a")), 2e-3)
    # 경우 분류: a>1 은 f(0)=2 → a=1/2 로 범위 밖, 0<a<1 은 f(-1)=2
    a = sp.symbols("a", positive=True)
    V.check(4, "a>1 후보", sp.solve(sp.Eq(1 + 2 * a, 2), a)[0], sp.Rational(1, 2))
    V.check(4, "0<a<1 후보", sp.solve(sp.Eq(3 * a, 2), a)[0], P(V.ans(4, "1", "a")))
    V.report(4, sp.Rational(1, 2) < 1 and sp.Rational(2, 3) < 1, "a=1/2 는 a>1 이 아님, a=2/3 는 0<a<1")

def p05():
    f = lambda t: 4 * sp.tan(sp.pi + 4 * t) - 1
    V.check(5, "1 주기", sp.pi / 4, P(V.ans(5, "1")))
    for t0 in (0.01, 0.1, 0.2):
        V.report(5, abs(float(f(t0 + sp.pi / 4)) - float(f(t0))) < 1e-9, f"f(x+π/4)=f(x) x={t0}")
    V.report(5, abs(float(f(0.1 + sp.pi / 8)) - float(f(0.1))) > 1e-3, "π/8 는 주기 아님")
    V.check(5, "2 f(π/4)", sp.simplify(f(sp.pi / 4)), P(V.ans(5, "2")))
    # (3) f(c+t)+f(c-t)=2*(-1) 이 열린구간 전체에서 성립하는 c=0
    cx, cy = [P(v) for v in V.ans(5, "3").strip("()").split(",")]
    ok = all(abs(float(f(cx + t) + f(cx - t)) - 2 * float(cy)) < 1e-9 for t in np.linspace(0.001, 0.39, 50))
    V.report(5, ok, f"(3) 대칭중심 ({cx},{cy}) 수치 확인")
    V.report(5, abs(float(f(0.2) + f(-0.2)) - 2 * 0) > 0.5, "(0,0) 은 대칭중심 아님")

def p06():
    h, a = sp.symbols("h a", real=True)
    f = lambda t: t**2 - 6 * t
    fp = sp.simplify(sp.limit((f(x + h) - f(x)) / h, h, 0))
    V.check(6, "f'", fp, P(kv(V.ans(6, "1"))["fprime"]))
    m = sp.simplify((f(6) - f(1)) / (6 - 1))
    V.check(6, "직선 기울기", m, P(kv(V.ans(6, "1"))["slope"]))
    sol = sp.solve(sp.Eq(fp.subs(x, a) * m, -1), a)
    V.check(6, "a", sol[0], P(kv(V.ans(6, "1"))["a"]))
    V.check(6, "해 개수", len(sol), 1)

def p07():
    a1, d = sp.symbols("a1 d", real=True)
    an = lambda n: a1 + (n - 1) * d
    S5 = sum(an(i) for i in range(1, 6))
    sols = sp.solve([sp.Eq(S5, 85), sp.Eq(an(15)**2, an(25)**2)], [a1, d], dict=True)
    sols = [s for s in sols if s[d] != 0]
    V.check(7, "해 개수(d≠0)", len(sols), 1)
    V.check(7, "a1", sols[0][a1], P(kv(V.ans(7, "1"))["a1"]))
    V.check(7, "d", sols[0][d], P(kv(V.ans(7, "1"))["d"]))
    # 전수 확인: 정수 격자
    hits = [(p, q) for p in range(-60, 61) for q in range(-30, 31) if q != 0
            and 5 * p + 10 * q == 85 and abs(p + 14 * q) == abs(p + 24 * q)]
    V.report(7, hits == [(19, -1)], f"정수 격자 해 {hits}")
    # 오답 재현: a15=a25 만 두면 d=0
    V.check(7, "오답 a15=a25 → d", sp.solve(sp.Eq(an(15), an(25)), d)[0], 0)

def p08():
    a = sp.symbols("a", real=True)
    left = a**2 + a          # x->a- : x^2 + a
    right = 3 * a            # x->a+ : x + (-x+3a)
    val = a**2 + (-a + 3 * a)
    ex = sp.solve(sp.Eq(left, right), a)
    V.check(8, "극한 존재 a", set(ex), {sp.Integer(int(t)) for t in kv(V.ans(8, "1"))["lim"].split(",")})
    ct = sp.solve([sp.Eq(left, right), sp.Eq(right, val)], a, dict=True)
    V.check(8, "연속 a", [s[a] for s in ct], [P(kv(V.ans(8, "1"))["cont"])])
    # 구체값: 정의대로 함수를 만들어 x=a 양쪽에서 수치 극한과 함숫값 비교
    for av in (0, 2, 1, -1, 0.5, 3):
        F = lambda t: (t**2 if t <= av else t) + (av if t < av else -t + 3 * av)
        e = 1e-9
        l, r, v = F(av - e), F(av + e), F(av)
        lim_ok = abs(l - r) < 1e-6
        V.report(8, lim_ok == (av in (0, 2)), f"a={av}: 좌 {l:.6f} 우 {r:.6f} 극한존재 {lim_ok}")
        V.report(8, (lim_ok and abs(r - v) < 1e-6) == (av == 0), f"a={av}: 연속 여부 (함숫값 {v:.6f})")

def p09():
    a, b = sp.symbols("a b", real=True)
    f = x**2 + a * x + b
    s = sp.solve([sp.integrate(sp.diff(f, x), (x, -1, 1)), sp.diff(sp.integrate(f, (x, 0, x)), x).subs(x, 1)], [a, b], dict=True)
    V.check(9, "해 개수", len(s), 1)
    fs = f.subs(s[0])
    V.check(9, "f", fs, P(kv(V.ans(9, "1"))["f"]))
    g = sp.integrate(fs.subs(x, sp.Symbol("t")), (sp.Symbol("t"), 0, x))
    V.check(9, "g", g, P(kv(V.ans(9, "1"))["g"]))
    # 극소 확인(x=1): g' = f 부호 - → +
    V.report(9, sp.diff(g, x).subs(x, sp.Rational(1, 2)) < 0 and sp.diff(g, x).subs(x, sp.Rational(3, 2)) > 0, "g 는 x=1 에서 극소")
    mn, mx = extrema(g, x, 1, 2)
    V.check(9, "M", mx, P(kv(V.ans(9, "1"))["M"]))
    V.check(9, "m", mn, P(kv(V.ans(9, "1"))["m"]))
    # 오답 재현: f(1)+f(-1)=0
    V.check(9, "오답 f(1)+f(-1)=0 → b", sp.solve((1 + b) + (1 + b), b)[0], -1)

def p10():
    k = sp.symbols("k", positive=True)
    cands = {"(x-k)^2(x-2k)": (x - k)**2 * (x - 2 * k), "(x-k)(x-2k)^2": (x - k) * (x - 2 * k)**2}
    good = {}
    for nm, f in cands.items():
        L = sp.limit(f / (x - k), x, k)
        sol = sp.solve(sp.Eq(L, 9), k)
        good[nm] = [s for s in sol if s > 0]
    V.check(10, "(x-k)^2(x-2k) 는 극한 0", sp.limit(cands["(x-k)^2(x-2k)"] / (x - k), x, k), 0)
    V.check(10, "(x-k)(x-2k)^2 해 k", good["(x-k)(x-2k)^2"], [P(kv(V.ans(10, "1"))["k"])])
    fsol = cands["(x-k)(x-2k)^2"].subs(k, 3)
    V.check(10, "f", sp.expand(fsol), sp.expand(P(kv(V.ans(10, "1"))["f"])))
    V.check(10, "f(4)", fsol.subs(x, 4), P(kv(V.ans(10, "1"))["f4"]))
    # 실근이 k, 2k 뿐인 모든 monic 삼차식은 위 2가지(근의 중복도 배치 전수)
    shapes = [(i, j) for i in range(1, 3) for j in range(1, 3) if i + j == 3]
    V.check(10, "꼴의 수", len(shapes), 2)

if __name__ == "__main__": V.run(globals())
