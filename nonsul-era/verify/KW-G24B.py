"""광운대 2024학년도 자연계열 2교시 [문제 1]·[문제 2] 검산 (기대값은 YAML answer 에서만 읽는다)."""
import sys, pathlib, math; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("KW-G24B", "items/KW/PAST")
L = {"x": x, "n": n, "t": t}
def E(s): return sp.sympify(s, locals=L)

def p01():
    # [1](1): k=5 로 수치·기호 확인 (k>1 이면 같음)
    kk = 5
    f = lambda arg: sp.expand_log(sp.log(arg)/sp.log(kk), force=True)
    an = lambda m: f((kk**(2*m))) - f(kk**m)          # B_n - A_n 의 y좌표 차: f(x^2)-f(x) at x=k^m
    V.check(1, "1-1", limit(sp.simplify(an(n))/sp.simplify(an(n**2)), n, sp.oo), V.ans(1, "1-1"))
    # [1](2): 5^10 의 1 아닌 약수는 5^1..5^10 뿐
    divs = [d for d in sp.divisors(5**10) if d != 1]
    P = sp.prod(divs)
    pos = [r for r in sp.real_roots(sp.Poly(t**2 + 3*t + 4 - P, t)) if r > 0]
    V.check(1, "1-2 양의 실근 개수", len(divs) == 10 and len(pos), V.ans(1, "1-2"))
    # [1](3),(4)
    def g10(v):
        for _ in range(10): v = v + 1
        return v
    V.report(1, sp.simplify(g10(x) - (x + 10)) == 0 and sp.diff(g10(x), x) == 1, "1-3 g^10(x)=x+10, 증가함수(일대일)")
    V.proof(1, "1-3 서로 다른 a,b 에서 g^10 값 다름(정수 -50..50)", search(lambda a, b: a == b or g10(a) != g10(b), range(-50, 51), range(-50, 51)))
    V.check(1, "1-4", sum(g10(j) for j in range(1, 101)) , V.ans(1, "1-4"))
    # [2] 평균값 정리: 여러 a 에서 열린구간에 해가 존재하는지 (h 의 부호변화/영점)
    ok = True
    for a in [-3.7, -1, -0.3, 0.5, 1, 2, 3.7, 10]:
        hv = lambda tt: math.sin(a*a + a - 1) - math.sin(a*a - 1) - a*math.cos(a*tt - 1)
        ts = [a + (i + 0.5)/4000 for i in range(4000)]
        vals = [hv(tt) for tt in ts]
        ok &= (min(vals) <= 0 <= max(vals))
    V.proof(1, "2 열린구간 (a,a+1) 에서 h(t)=0 해 존재(표본 8개 a)", [] if ok else ["반례"])

def p02():
    # [1]
    fx = sp.log(1 + x)/x**2*(sp.exp(2*x)*sp.cos(sp.pi*x) - 1)
    V.check(2, "1", limit(fx, x, 0), V.ans(2, "1"))
    # [2]
    sols = [(pp, qq) for pp in range(1, 60) for qq in range(1, 60) if 4*pp + 2 + qq**2 == 18]
    V.check(2, "2 p", sols[0][0] if len(sols) == 1 else -1, V.ans(2, "2", "p"))
    V.check(2, "2 q", sols[0][1] if len(sols) == 1 else -1, V.ans(2, "2", "q"))
    # [3]
    bad = [m for m in range(1, 31) if sp.expand(x**m - 1 - (x - 1)*sum(x**i for i in range(m))) != 0]
    V.proof(2, "3-1 등식 n=1..30", bad)
    V.check(2, "3-2", sp.rem(x**2024, x**3 + x**2 + x + 1, x), V.ans(2, "3-2"))
    # [4]
    f = sp.exp(sp.sin(x)**2) - sp.exp(sp.cos(x)**2)
    df = sp.diff(f, x)
    V.report(2, sp.simplify(df - 2*sp.sin(x)*sp.cos(x)*(sp.exp(sp.sin(x)**2) + sp.exp(sp.cos(x)**2))) == 0, "4-1 공식 f'(x) = 직접 미분")
    crit = [0, sp.pi/2, sp.pi, 3*sp.pi/2, 2*sp.pi]
    vals = [sp.simplify(f.subs(x, c)) for c in crit]
    h = sp.lambdify(x, f, "math")
    gmax = max(h(2*math.pi*i/20000) for i in range(20001)); gmin = min(h(2*math.pi*i/20000) for i in range(20001))
    V.check(2, "4-1 max", max(vals, key=lambda v: float(v)), E(V.ans(2, "4-1", "max")))
    V.check(2, "4-1 min", min(vals, key=lambda v: float(v)), E(V.ans(2, "4-1", "min")))
    V.report(2, abs(gmax - float(E(V.ans(2, "4-1", "max")))) < 1e-8 and abs(gmin - float(E(V.ans(2, "4-1", "min")))) < 1e-8, f"4-1 격자 최대 {gmax:.6f} 최소 {gmin:.6f}")
    V.check(2, "4-2", sp.N(sp.Integral(f*sp.sin(2*x), (x, 0, sp.pi/4)), 25), sp.N(E(V.ans(2, "4-2")), 25), 1e-18)

if __name__ == "__main__": V.run(globals())
