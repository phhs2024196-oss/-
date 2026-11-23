import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import itertools
import mpmath as mp

V = Verifier("YSM-G24A", "items/YSM/PAST")
E = lambda v: sp.sympify(v, locals={"x": x, "t": t, "n": n, "k": k})

def p01():
    p, q = sp.symbols("p q", positive=True)
    # 1-1: D=(p,q) 가 y=2√(3x) 위, C=(p+q,q) 가 y=2√x 위
    sol = sp.solve([sp.Eq(q, 2*sp.sqrt(3*p)), sp.Eq(q, 2*sp.sqrt(p + q))], [p, q], dict=True)
    V.report(1, len(sol) == 1, f"1-1 양수해 개수 {len(sol)}")
    pv, qv = sol[0][p], sol[0][q]
    V.check(1, "1-1 p", pv, E(V.ans(1, "1-1", "p")))
    V.check(1, "1-1 q", qv, E(V.ans(1, "1-1", "q")))
    # 1-2: 단면(CD 에 수직인 평면 x=const)의 한 변 = q - 2√x, x∈[p,p+q]
    ell = qv - 2*sp.sqrt(x)
    S = sp.sqrt(3)/4*ell**2
    Vol = sp.simplify(sp.integrate(sp.expand(S), (x, pv, pv + qv)))
    mm = sp.nsimplify(sp.expand(Vol).coeff(sp.sqrt(3), 0)); nn = sp.nsimplify(sp.expand(Vol).coeff(sp.sqrt(3), 1))
    V.report(1, sp.simplify(mm + nn*sp.sqrt(3) - Vol) == 0, f"1-2 V={Vol}")
    V.check(1, "1-2 m", mm, E(V.ans(1, "1-2", "m")))
    V.check(1, "1-2 n", nn, E(V.ans(1, "1-2", "n")))
    V.check(1, "1-2 m+n", mm + nn, E(V.ans(1, "1-2", "s")))
    # 수치 적분 교차확인(영역 CDE 의 단면 길이를 직접 계산): 정삼각형 넓이를 mpmath 로
    mp.mp.dps = 30
    num = mp.quad(lambda xv: mp.sqrt(3)/4*(float(qv) - 2*mp.sqrt(xv))**2, [float(pv), float(pv + qv)])
    V.report(1, abs(num - mp.mpf(str(sp.N(Vol, 30)))) < 1e-15, f"1-2 수치적분 {num}")

def p02():
    h = sp.symbols("h", real=True)
    # 2-1: 도함수의 정의를 극한으로 직접 계산
    d = sp.limit((sp.cos(2*(x + h)) - sp.cos(2*x))/h, h, 0)
    V.check(2, "2-1 도함수", d, E(V.ans(2, "2-1")))
    V.check(2, "2-1 (cos2h-1)/h 극한", sp.limit((sp.cos(2*h) - 1)/h, h, 0), 0)
    V.check(2, "2-1 sin2h/h 극한", sp.limit(sp.sin(2*h)/h, h, 0), 2)
    # 2-2
    f = sp.exp(-x)*sp.cos(2*x)
    f0 = f.subs(x, 0)
    fp0 = sp.limit((f.subs(x, h) - f0)/h, h, 0)   # 정의로
    V.check(2, "2-2 f(0)", f0, 1); V.check(2, "2-2 f'(0)", fp0, -1)
    V.check(2, "2-2 f'(0)=미분", sp.diff(f, x).subs(x, 0), fp0)
    kk = sp.symbols("kk", integer=True, positive=True)
    ans = fp0*sp.summation(kk, (kk, 1, 2023))
    V.check(2, "2-2 값", ans, E(V.ans(2, "2-2")))
    # 수치 확인: x 가 작을 때 (1/x)[sum f(kx) - 2023]
    mp.mp.dps = 60
    xv = mp.mpf(10)**-20
    ff = lambda z: mp.e**(-z)*mp.cos(2*z)
    val = (sum(ff(j*xv) for j in range(1, 2024)) - 2023)/xv
    V.report(2, abs(val - sp.N(ans, 30)) < 1e-6, f"2-2 수치 {mp.nstr(val, 15)}")

def p03():
    # 3-1: 결론의 부정 (a≠0 또는 b≠0) 의 세 경우에서 a²+b²>0 임을 격자에서 확인, 순서는 답에 적힌 대로
    vals = [sp.Rational(i, 3) for i in range(-9, 10)]
    bad = [(a, b) for a in vals for b in vals if (a != 0 or b != 0) and a**2 + b**2 <= 0]
    V.proof(3, "3-1 (a,b)≠(0,0) ⇒ a²+b²>0 (세 경우 전체)", bad)
    cases = {1: lambda a, b: a != 0 and b == 0, 2: lambda a, b: a == 0 and b != 0, 3: lambda a, b: a != 0 and b != 0}
    cover = [(a, b) for a in vals for b in vals if (a != 0 or b != 0) and sum(c(a, b) for c in cases.values()) != 1]
    V.report(3, not cover, "3-1 세 경우가 부정(a≠0 또는 b≠0)을 빠짐없이 겹침 없이 나눔")
    V.report(3, "(Ⅲ)→(Ⅳ)→(Ⅰ)→(Ⅱ)" in V.ans(3, "3-1"), "3-1 답의 순서 표기")
    # 3-2: 일대일대응 일차함수
    c, p, q = sp.symbols("c p q", real=True)
    c_ = sp.symbols("c_", positive=True)
    ratios = set()
    for (y_l, y_r) in ((-8, 0), (0, -8)):   # f(-c)=y_l, f(c)=y_r
        s = sp.solve([sp.Eq(-p*c_ + q, y_l), sp.Eq(p*c_ + q, y_r)], [p, q], dict=True)[0]
        ratios.add(sp.simplify(s[q]/s[p]))
    exp = {sp.simplify(sp.sympify(v.strip(), locals={"c": c_})) for v in V.ans(3, "3-2").split(",")}
    V.report(3, ratios == exp, f"3-2 q/p {ratios} / YAML {exp}")
    # 일대일대응 확인(수치): c=3, 두 직선이 [-3,3]을 [-8,0]으로 전단사
    for (pp, qq) in ((sp.Rational(4, 3), -4), (-sp.Rational(4, 3), -4)):
        ys = sorted([pp*(-3) + qq, pp*3 + qq])
        V.report(3, ys == [-8, 0], f"3-2 c=3 p={pp} 치역 {ys}")
    # 3-3: 전수 열거
    cnt = [(a, b, cc) for a in range(1, 36) for b in range(1, a + 1) for cc in range(1, b + 1)
           if a + b + cc == 36 and a < b + cc]
    V.check(3, "3-3 개수", len(cnt), E(V.ans(3, "3-3")))
    per = {a: sum(1 for T in cnt if T[0] == a) for a in range(12, 18)}
    V.report(3, per == {12: 1, 13: 2, 14: 4, 15: 5, 16: 7, 17: 8}, f"3-3 x1별 개수 {per}")
    V.report(3, min(T[0] for T in cnt) == 12 and max(T[0] for T in cnt) == 17, "3-3 x1 범위 12..17")

if __name__ == "__main__": V.run(globals())
