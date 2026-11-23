import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("HSN-G25N", "items/HSN/PAST")

def p07():
    # log9 a + log9 b = 7/2 -> ab = 3^7, 자연수 순서쌍 전수
    cnt = sum(1 for a in range(1, 3**7 + 1) for b in [3**7 // a] if a * b == 3**7
              and abs(sp.N(sp.log(a, 9) + sp.log(b, 9)) - 3.5) < 1e-9)
    V.check(7, "개수", cnt, V.ans(7, "1"))

def p08():
    a, b = sp.symbols("a b", real=True)
    f = lambda av: sp.limit((sp.sqrt(2 * x - av) - x) / (x - 3), x, 3)
    V.check(8, "a=-3 일 때 극한 b", f(-3), V.ans(8, "1", "b"))
    # a 가 -3 이 아니면 극한이 발산(유한값 아님)
    V.check(8, "a=-3 만 유한", [av for av in range(-8, 6) if sp.limit((sp.sqrt(2 * x - av) - x) / (x - 3), x, 3, "+").is_finite], [S(V.ans(8, "1", "a"))])

def p09():
    d = sp.symbols("d")
    an = lambda n: 36 + (n - 1) * d
    ms = []
    for dv in sp.solve(an(10) + an(28), d) + [0]:
        if abs(an(10).subs(d, dv)) == abs(an(28).subs(d, dv)) and dv != 0:
            ms += [m for m in range(1, 200) if an(m).subs(d, dv) == -10]
    V.check(9, "m", ms[0], V.ans(9, "1"))

def p10():
    a = sp.symbols("a", real=True)
    fl, fr = a, 0                       # f(0-)=a, f(0+)=0
    g = lambda v: (v - 1) * (v + 2)
    sols = [s for s in sp.solve(sp.Eq(g(fl), g(fr)), a) if s != 0]
    V.check(10, "a", sols[0], V.ans(10, "1"))

def p11():
    a = sp.symbols("a"); h = sp.symbols("h")
    c = sp.symbols("c")
    # f' = -(4x^2+ax)/2, 적분
    fp = -(4 * x**2 + a * x) / 2
    av = sp.solve(fp.subs(x, 1), a)[0]
    F = sp.integrate(fp.subs(a, av), x) + c
    cv = sp.solve(F.subs(x, 3) + 8, c)[0]
    F = F.subs(c, cv)
    # 원 조건 재확인: 극한식
    chk = sp.limit((F.subs(x, x - 2 * h) - F) / h, h, 0)
    V.check(11, "a", av, V.ans(11, "1", "a"))
    V.check(11, "f", F, sp.sympify(V.ans(11, "1", "f"), locals={"x": x}))
    V.check(11, "극한 재확인", chk - (4 * x**2 + av * x), 0)

def p12():
    A, B, D, C = sp.symbols("A B D C")
    ab, ad, bd, bc = 4, 5, 6, 6
    cos = sp.Rational(ad**2 + bd**2 - ab**2, 2 * ad * bd)
    sin = sp.sqrt(1 - cos**2)
    s1 = sp.Rational(1, 2) * ad * bd * sin
    s2 = sp.Rational(1, 2) * bd * bc * sin
    for key, val in dict(ga=cos, na=sin, da=s1, ra=s2, ma=s1 + s2).items():
        V.check(12, key, val, V.ans(12, "1", key))

def p13():
    a, b = sp.symbols("a b")
    t = sp.symbols("t")
    # f 를 일차까지 가정해 확인: x f(x) = 3x^2+2ax+b, x=0 -> b=0 ; x=1 -> a+b=-1
    sol = sp.solve([0 - (1 + a + b), (3 * 0**2 + 2 * a * 0 + b)], [a, b])
    f = sp.cancel((3 * x**2 + 2 * a * x + b).subs(sol) / x)
    # 원 조건 재확인
    lhs = sp.integrate(t * f.subs(x, t), (t, 1, x))
    V.check(13, "조건", sp.expand(lhs - (x**3 + sol[a] * x**2 + sol[b] * x)), 0)
    V.check(13, "f(b)", f.subs(x, sol[b]), V.ans(13, "1"))

def p14():
    al = sp.symbols("alpha", real=True)
    f = (x + 1)**2 * (x - al)
    # 극솟값 = -4, alpha>-1 (x=-1 이 극대일 조건), 후보 전수
    res = []
    for av in sp.solve(sp.Eq(f.subs(x, (2 * al - 1) / 3), -4), al):
        if av.is_real and av != -1:
            fv = f.subs(al, av)
            crit = sp.solve(sp.diff(fv, x), x)
            mins = [fv.subs(x, c) for c in crit if sp.diff(fv, x, 2).subs(x, c) > 0]
            if mins == [-4]: res.append(av)
    V.check(14, "alpha", res[0], V.ans(14, "1", "alpha"))
    V.check(14, "f", sp.expand(f.subs(al, res[0])), sp.expand(sp.sympify(V.ans(14, "1", "f"), locals={"x": x})))

def p15():
    f = x**3 + x - 1
    xs = sp.symbols("xs", real=True)
    g = lambda y0: sp.nsolve(f - y0, 0.5)   # 역함수 수치
    # 넓이 = ∫_{-1}^{1} (g(x) - max(f(x), -1)) dx, g 는 수치 역함수
    import mpmath as mp
    mp.mp.dps = 30
    ff = lambda t: t**3 + t - 1
    gg = lambda y0: mp.findroot(lambda t: ff(t) - y0, 0.5)
    fm = lambda t: max(ff(t), -1)
    I = mp.quad(lambda t: gg(t) - fm(t), [-1, 0, 1])
    V.check(15, "넓이(수치)", sp.Float(str(I), 25), sp.N(S(V.ans(15, "1")), 25), 1e-9)
    ex = 1 + 2 * integral(x - f, x, 0, 1)
    V.check(15, "넓이(대칭식)", ex, V.ans(15, "1"))
    V.check(15, "별해1", 4 - 2 * integral(f + 1, x, 0, 1), V.ans(15, "1"))

if __name__ == "__main__": V.run(globals())
