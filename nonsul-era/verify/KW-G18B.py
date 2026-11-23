"""KW-G18B 검산: 문제 조건에서 독립 재계산한 값을 items/KW/PAST 의 answer.value 와 비교."""
import sys, pathlib, itertools, re
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import mpmath as mp

V = Verifier("KW-G18B", "items/KW/PAST")


def parse_sets(s):
    return sorted(tuple(sorted(int(v) for v in g.split(","))) for g in re.findall(r"\{([^{}]*)\}", s))


def p01():
    tt = sp.Symbol("tt")
    sols = [r for r in sp.solve(sp.Eq(tt, 2 * tt**2 - tt - 4), tt) if {r, 3} == {r + 1, 2 * r**2 - r - 4}]
    V.check(1, "1 t", sols[0] if len(sols) == 1 else -999, V.ans(1, "1"))
    # [2] 정수 부분집합: 원소 절댓값 ≤ 6 범위 전수(제곱 닫힘 + 유한이려면 {-1,0,1} 안)
    found = []
    for r in range(1, 5):
        for A in itertools.combinations(range(-3, 4), r):
            if all(a * a in A for a in A):
                found.append(tuple(sorted(A)))
    V.check(1, "2 집합", sorted(found) == parse_sets(V.ans(1, "2")), True)
    V.check(1, "2 개수", len(found), 5)
    # [3](1)
    perms = list(itertools.permutations([1, 2, 3]))
    V.check(1, "3-1 개수", len(perms), V.ans(1, "3-1"))
    # (2) f(1)=3, f(2)=1 인 일대일함수
    (f,) = [p for p in perms if p[0] == 3 and p[1] == 1]
    inv = {v: i + 1 for i, v in enumerate(f)}
    V.check(1, "3-2", inv[inv[2]], V.ans(1, "3-2"))
    # (3) b^2-4ac>0 인 순열
    vals = {b - a * c for a, b, c in perms if b * b - 4 * a * c > 0}
    V.check(1, "3-3", list(vals)[0] if len(vals) == 1 else -999, V.ans(1, "3-3"))


def p02():
    tt = sp.Symbol("tt", positive=True)
    xt = -tt**3 + 15 * tt**2
    v = sp.diff(xt, tt)
    V.check(2, "1 t", [r for r in sp.solve(v, tt)][0], V.ans(2, "1"))
    # 속도 부호가 0<t<10 에서 양, t>10 에서 음 → 처음 바뀌는 시각
    V.check(2, "1 부호", [sp.sign(v.subs(tt, 5)), sp.sign(v.subs(tt, 11))], [1, -1])
    # [2] 연속: 좌극한 b, 우극한 a·x^3 → a, f(1)=a/b
    a_, b_ = sp.symbols("a b", real=True)
    sol = sp.solve([sp.Eq(b_, a_), sp.Eq(a_, a_ / b_)], [a_, b_], dict=True)
    sol = [s for s in sol if s[b_] != 0]
    V.check(2, "2 a", sol[0][a_], V.ans(2, "2", "a"))
    V.check(2, "2 b", sol[0][b_], V.ans(2, "2", "b"))
    # 극한 수치 확인(a=b=1): x=0.9, x=1.1 에서 n 큰 값
    lim = lambda xv, n: (xv ** (n + 3) - xv**2 + xv + 1) / (xv**n + 1)
    V.check(2, "2 좌극한 수치", float(lim(mp.mpf("0.9999"), 200000)), float(sol[0][b_]), tol=1e-3)
    # [3]
    k_, u = sp.symbols("k u", real=True)
    S = lambda kk: sp.integrate(sp.Abs(2 * u - kk), (u, 0, 1))
    V.check(2, "3 S(1/2)", S(sp.Rational(1, 2)), V.ans(2, "3", "S_half"))
    best = min(S(sp.Rational(i, 20)) for i in range(0, 41))
    V.check(2, "3 최솟값", best, V.ans(2, "3", "min"))
    # [4] 수치: f(x)=0 만이 조건을 만족(선형 f=cx 는 |cx|≤x^2 불가 → c=0)
    ok = [c for c in sp.symbols("c", real=True) and [sp.Rational(i, 10) for i in range(-30, 31)] if all(abs(c * xv) <= xv**2 for xv in [sp.Rational(1, 100), sp.Rational(1, 1000)])]
    V.proof(2, "4 선형 f=cx 는 c=0 뿐", [c for c in ok if c != 0])


if __name__ == "__main__":
    V.run(globals())
