"""KW-G26B 검산: 문제 조건에서 독립 재계산한 값을 items/KW/PAST 의 answer.value 와 비교."""
import sys, pathlib, itertools, re
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp

V = Verifier("KW-G26B", "items/KW/PAST")
SQ3 = sp.sqrt(3)


def p01():
    # [1]
    V.check(1, "1 이중합", sum(sum(j * j for j in range(m, 11)) for m in range(1, 10)), V.ans(1, "1"))
    # [2](1): k 에 대한 항등식
    k_, X, Y = sp.symbols("k X Y", real=True)
    f = X ** 2 + 2 * (k_ + 1) * X + k_ + 3
    eqs = sp.Poly(sp.expand(f - Y), k_).all_coeffs()
    sol = sp.solve(eqs, [X, Y], dict=True)[0]
    V.check(1, "2-1 x", sol[X], V.ans(1, "2-1", "x"))
    V.check(1, "2-1 y", sol[Y], V.ans(1, "2-1", "y"))

    # [2](2): k 격자에서 선분 AB 와의 교점 개수 = 2 인 k 집합이 YAML 구간과 일치하는가
    m = re.match(r"(.+?)\s*<=\s*k\s*<\s*(.+)", V.ans(1, "2-2"))
    lo, hi = float(sp.sympify(m.group(1))), float(sp.sympify(m.group(2)))

    def n_inter(kv):
        # 선분 AB: y=x+2, -1<=x<=3. 교점 x: f(x)=x+2, 즉 x^2+(2k+1)x+k+1=0 의 서로 다른 실근 중 [-1,3] 안
        D = (2 * kv + 1) ** 2 - 4 * (kv + 1)
        if D <= 0:
            return 0
        r = [(-(2 * kv + 1) + s * D ** 0.5) / 2 for s in (1, -1)]
        return sum(1 for v in r if -1 - 1e-12 <= v <= 3 + 1e-12)
    bad = []
    for i in range(-6000, 6001):
        kv = i / 1000
        want = lo - 1e-9 <= kv < hi - 1e-12
        if (n_inter(kv) == 2) != want:
            bad.append(kv)
    V.proof(1, "2-2 구간 일치(k 격자)", bad)
    V.check(1, "2-2 경계 k=lo 에서 교점 2", n_inter(lo), 2)
    V.check(1, "2-2 경계 k=hi 에서 교점 1(미만)", n_inter(hi) < 2, True)
    # [3]: 서로 다른 세 선분 중 어느 두 선분도 (끝점 공유 제외) 교차하지 않는 경우의 수
    segs = [(i, j) for i in range(5) for j in range(7)]

    def cross(s, t):
        return (s[0] - t[0]) * (s[1] - t[1]) < 0
    cnt = sum(1 for tr in itertools.combinations(segs, 3) if not any(cross(a, b) for a, b in itertools.combinations(tr, 2)))
    V.check(1, "3 비교차 3선분", cnt, V.ans(1, "3"))
    # [4]: 정규분포 표 대신 정확한 분위수로 확인(표 값 2.05 근사와 같은 결론)
    def ok(kn):
        zthr = 300 - 10 * kn
        pz = mp.ncdf(-zthr)  # P(Z >= zthr)
        return pz <= 0.98
    kmax = max(kn for kn in range(1, 200) if ok(kn))
    V.check(1, "4 k 최댓값", kmax, V.ans(1, "4"))


def p02():
    n = sp.Symbol("n")
    an = sp.sympify(V.ans(2, "1-1", "a_n"), locals={"n": n})
    bn = sp.sympify(V.ans(2, "1-1", "b_n"), locals={"n": n})
    for nv in range(1, 9):
        V.check(2, f"1-1 n={nv} a+b√3", sp.simplify((an + bn * SQ3 - (2 + SQ3) ** n).subs(n, nv)), 0)
        V.check(2, f"1-1 n={nv} ab", sp.simplify((an * bn - ((2 + SQ3) ** (2 * n) - (2 - SQ3) ** (2 * n)) / (4 * SQ3)).subs(n, nv)), 0)
    V.check(2, "1-1 a1=2", an.subs(n, 1), 2)
    nv = 300
    a_v = sp.N(an.subs(n, nv), 900)
    b_v = sp.N(bn.subs(n, nv), 900)
    V.check(2, "1-2 극한", sp.N(b_v * (a_v ** 2 - 3 * b_v ** 2) / a_v, 30), sp.N(sp.sympify(V.ans(2, "1-2")), 30), 1e-20)
    # [2]
    fx = sp.Function("f")
    t = sp.Symbol("t")
    sol = sp.dsolve(sp.Eq(fx(x).diff(x) + fx(x), 1 + sp.sin(x)), fx(x), ics={fx(0): sp.Rational(1, 2)})
    fexp = sol.rhs  # 식 ①을 미분한 방정식과 f(0)=1/2
    chk = sp.simplify(fexp + sp.integrate(fexp.subs(x, t), (t, 0, x)) - (x - sp.cos(x) + sp.Rational(3, 2)))
    V.check(2, "2 원식 만족", chk, 0)
    roots = sorted(sp.solveset(sp.Eq(fexp - 1, 0), x, sp.Interval.Ropen(0, 2 * sp.pi)), key=lambda r: float(r))
    V.check(2, "2 근1", roots[0], V.ans(2, "2", "x1"))
    V.check(2, "2 근2", roots[1], V.ans(2, "2", "x2"))
    V.check(2, "2 근 개수", len(roots), 2)
    # [3](1): f(0)=1 인 임의의 f 에 대해 항등식(수치)
    for ff in (1 + x + x ** 2, sp.exp(x / 2), 2 - sp.cos(x)):
        lhs = sp.integrate(2 * ff.subs(x, t) * sp.diff(ff, x).subs(x, t), (t, 0, x))
        V.report(2, sp.simplify(lhs - (ff ** 2 - 1)) == 0 or ff.subs(x, 0) != 1, f"3-1 항등식 f={ff}")
    # [3](2)
    fs = sp.sympify(V.ans(2, "3-2"), locals={"x": x})
    V.check(2, "3-2 (가)", fs.subs(x, 0), 1)
    V.check(2, "3-2 (나) 2f'-f", sp.simplify(2 * sp.diff(fs, x) - fs), 0)
    V.check(2, "3-2 (다) ∫f²+1-f²", sp.simplify(sp.integrate(fs.subs(x, t) ** 2, (t, 0, x)) + 1 - fs ** 2), 0)
    # 유일성: f²≤2ff', f² ≤ ∫f²+1 로부터 얻는 ODE f'=f/2 의 해
    gsol = sp.dsolve(sp.Eq(fx(x).diff(x), fx(x) / 2), fx(x), ics={fx(0): 1})
    V.check(2, "3-2 ODE 해", sp.simplify(gsol.rhs - fs), 0)


if __name__ == "__main__":
    V.run(globals())
