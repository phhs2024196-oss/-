import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import itertools, random
V = Verifier("UOS-G17", "items/UOS/PAST")
th, b = sp.symbols("th b", real=True)


def _area_max_over_c(line_b):
    """y=x+line_b 가 AB 일 때 C 를 타원 위에서 움직여 얻는 넓이 최댓값(정확값)."""
    xs = sp.solve(sp.Eq(x**2/5 + (x + line_b)**2/4, 1), x)
    A = [(r, r + line_b) for r in xs]
    ab = sp.sqrt((A[0][0]-A[1][0])**2 + (A[0][1]-A[1][1])**2)
    # C=(sqrt5 cos th, 2 sin th), 거리 |x-y+b|/sqrt2 의 최댓값: 진폭 sqrt(5+4) 와 |b| 의 합
    dmax = (3 + abs(line_b)) / sp.sqrt(2)   # x-y = sqrt5 cos - 2 sin 의 진폭 3 (아래에서 수치 교차확인)
    return sp.simplify(ab * dmax / 2)


def p01():
    # 진폭 3 확인: max of sqrt5 cos t - 2 sin t
    amp = sp.sqrt(5**1 + 4)
    V.check(1, "진폭", amp, 3)
    V.check(1, "1-1 S max", _area_max_over_c(0), V.ans(1, "1-1"))
    # 수치 교차확인: 타원 위 점 전수 탐색
    import numpy as np
    T = np.linspace(0, 2*np.pi, 200001)
    d = np.abs(np.sqrt(5)*np.cos(T) - 2*np.sin(T)) / np.sqrt(2)
    V.check(1, "1-1 수치 d max", float(d.max()), float(3/sp.sqrt(2)), 1e-6)
    # 1-2: b 를 (-3,3) 에서 움직여 최대(양쪽 부호 모두)
    S = sp.sqrt(sp.Rational(1))  # placeholder 방지
    bb = sp.symbols("bb", real=True)
    f = sp.Rational(2, 9)*sp.sqrt(5)*sp.sqrt((3-bb)**3*(3+bb))   # b<=0 가정 식
    # 식 자체 검증: AB 길이·거리로부터 재구성
    r = sp.solve(sp.Eq(x**2/5 + (x + bb)**2/4, 1), x)
    ab = sp.sqrt(2)*sp.Abs(r[0]-r[1])
    S_b = sp.simplify(ab*(3-bb)/sp.sqrt(2)/2)
    for bv in [-2, -1, sp.Rational(-3, 2), 0]:
        V.check(1, f"S(b={bv}) 식", sp.N(S_b.subs(bb, bv)), sp.N(f.subs(bb, bv)))
    crit = [sp.nsimplify(c) for c in sp.solve(sp.diff((3-bb)**3*(3+bb), bb), bb) if c.is_real and -3 < c <= 0]
    mx = max(sp.simplify(f.subs(bb, c)) for c in crit + [0])
    V.check(1, "1-2 S max", mx, V.ans(1, "1-2"))
    # 수치 전수: b 격자 × 타원 점
    best = 0.0
    for bv in np.linspace(-2.99, 2.99, 599):
        rr = np.roots([9, 10*bv, 5*bv**2-20]); al, be = rr
        abv = np.sqrt(2)*abs(al-be)
        dd = np.max(np.abs(np.sqrt(5)*np.cos(T[::50]) - 2*np.sin(T[::50]) + bv)) / np.sqrt(2)
        best = max(best, float(abv*dd/2))
    V.check(1, "1-2 수치 max", best, float(sp.N(sp.sympify(V.ans(1, "1-2")))), 1e-3)


def p02():
    nn, pp = sp.symbols("nn pp", positive=True)
    xx = sp.symbols("xx", integer=True)
    for N in range(2, 9):
        E = sum(i*sp.binomial(N, i)*pp**i*(1-pp)**(N-i) for i in range(N+1))
        E2 = sum(i**2*sp.binomial(N, i)*pp**i*(1-pp)**(N-i) for i in range(N+1))
        V.check(2, f"E n={N}", sp.expand(E), sp.expand(S(V.ans(2, "2-1")).subs(sp.Symbol("n"), N).subs(sp.Symbol("p"), pp)))
        Vr = E2 - E**2
        V.check(2, f"V n={N}", sp.expand(Vr), sp.expand(S(V.ans(2, "2-2")).subs(sp.Symbol("n"), N).subs(sp.Symbol("p"), pp)))
        # 증명의 중간 항등식
        V.check(2, f"E(X^2)=n(n-1)p^2+np n={N}", sp.expand(E2), sp.expand(N*(N-1)*pp**2 + N*pp))
    # 조합 항등식 x*C(n,x)=n*C(n-1,x-1), x(x-1)C(n,x)=n(n-1)C(n-2,x-2)
    V.proof(2, "조합 항등식", search(lambda N, i: i > N or (i*sp.binomial(N, i) == N*sp.binomial(N-1, i-1) and (i < 2 or i*(i-1)*sp.binomial(N, i) == N*(N-1)*sp.binomial(N-2, i-2))), range(2, 15), range(1, 15)))


def p03():
    random.seed(1)
    X, Y = sp.symbols("X Y")
    cnt = 0
    while cnt < 6:
        A = sp.Matrix([0, 0]); B = sp.Matrix([random.randint(3, 9), 0]); C = sp.Matrix([random.randint(1, 9), random.randint(2, 9)])
        ab, ac = B - A, C - A
        cb, bc = C - B, B - C
        # 예각삼각형 판정
        if not (ab.dot(ac) > 0 and (A-B).dot(C-B) > 0 and (A-C).dot(B-C) > 0):
            continue
        cnt += 1
        # 수심: BP ⟂ AC, CP ⟂ AB
        px, py = sp.symbols("px py")
        P = sp.solve([((sp.Matrix([px, py]) - B).dot(ac)), ((sp.Matrix([px, py]) - C).dot(ab))], [px, py])
        ap = sp.Matrix([P[px], P[py]]) - A
        # AP = x AB + y AC 를 풀어 (x, y) 구하기
        sol = sp.solve([ap[0] - (X*ab[0] + Y*ac[0]), ap[1] - (X*ab[1] + Y*ac[1])], [X, Y])
        u, v, w = ab.dot(ab), ab.dot(ac), ac.dot(ac)
        env = {sp.Symbol("u"): u, sp.Symbol("v"): v, sp.Symbol("w"): w}
        V.check(3, f"x {list(B)},{list(C)}", sol[X], S(V.ans(3, "3-1", "x")).subs(env))
        V.check(3, f"y {list(B)},{list(C)}", sol[Y], S(V.ans(3, "3-1", "y")).subs(env))


def p04():
    def a_cnt(N):
        U = range(-6*N, 6*N + 1); c = b_ = 0
        for a in U:
            for d in range(1, 12*N):
                if a + 4*d > 6*N: break
                s = {a + i*d for i in range(5)}
                c += 1
                if 0 not in s: b_ += 1
        return c, b_
    nsym = sp.Symbol("n")
    for N in range(1, 7):
        c, b_ = a_cnt(N)
        V.check(4, f"a_{N}", c, S(V.ans(4, "4-1")).subs(nsym, N))
        V.check(4, f"b_{N}", b_, S(V.ans(4, "4-2", "even" if N % 2 == 0 else "odd")).subs(nsym, N))


if __name__ == "__main__": V.run(globals())
