import sys, pathlib, itertools, math
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import numpy as np
V = Verifier("UOS-G18", "items/UOS/PAST")


def p01():
    V.check(1, "1-1 ∫1/(x^2+1)", sp.integrate(1/(x**2 + 1), (x, 0, 1)), V.ans(1, "1-1"))
    # (b) M_a, m_a 를 정의대로(교점 x좌표 최대·최소) 구해 ∫_0^2 (M_a - m_a) da 계산
    F1 = x**3 - 3*x
    def Fval(v):
        return v**3 - 3*v if v <= 0 else 4*v**2/(v**2 + 1)
    a_ = sp.symbols("a_", positive=True)
    # 기호: x>0 쪽 해 M_a = sqrt(a/(4-a)), x<=0 쪽 최솟해는 삼차 3근 중 가장 작은 것 = 2cos((acos(a/2)+2π)/3) 꼴로 구한다
    def Ma(av):
        return math.sqrt(av/(4 - av)) if av > 0 else 0.0
    def ma(av):
        rts = np.roots([1, 0, -3, -av])
        rts = [r.real for r in rts if abs(r.imag) < 1e-9 and r.real <= 1e-12]
        return min(rts)
    import mpmath as mp
    mp.mp.dps = 25
    val = mp.quad(lambda t: Ma(float(t)) - ma(float(t)), [0, 1, 2])
    V.check(1, "1-2 수치적분 ∫(M-m)", sp.Float(str(val), 15), sp.N(S(V.ans(1, "1-2")), 15), 1e-6)
    # 해설의 영역 넓이식도 정확히 계산
    ex = sp.integrate(F1, (x, -sp.sqrt(3), -1)) + 2 + sp.integrate(2 - 4*x**2/(x**2 + 1), (x, 0, 1))
    V.check(1, "1-2 영역 넓이식", sp.simplify(ex), V.ans(1, "1-2"))
    # 교점 검사: M_a, m_a 가 실제 교점인지 (a=1)
    V.check(1, "교점 확인 F(M_1)=1", sp.Float(Fval(Ma(1.0)), 12), 1, 1e-10)


def p02():
    s2 = sp.sqrt(2)
    P = {"B": (-1, -1, 0), "C": (1, -1, 0), "D": (1, 1, 0), "E": (-1, 1, 0), "A": (0, 0, s2), "F": (0, 0, -s2)}
    P = {k: sp.Matrix(v) for k, v in P.items()}
    # 정팔면체인지: 모든 인접 모서리 2
    edges = [("A", c) for c in "BCDE"] + [("F", c) for c in "BCDE"] + [("B", "C"), ("C", "D"), ("D", "E"), ("E", "B")]
    V.report(2, all(sp.simplify((P[u] - P[v]).norm() - 2) == 0 for u, v in edges), "정팔면체 12모서리 길이 2")
    g1 = (P["A"] + P["B"] + P["C"])/3
    g2 = (P["E"] + P["D"] + P["F"])/3
    d = g2 - g1
    V.check(2, "2-1 AB·G1G2", (P["B"] - P["A"]).dot(d), V.ans(2, "2-1"))
    # (b) d 방향 정사영의 넓이: 평면 ⟂ d 위로의 정사영(6점)의 볼록껍질 넓이
    n_ = d/d.norm()
    u = sp.Matrix([1, 0, 0]) - n_*n_.dot(sp.Matrix([1, 0, 0]))
    u = u/u.norm()
    w = n_.cross(u)
    pts = [(sp.N(p.dot(u), 30), sp.N(p.dot(w), 30)) for p in P.values()]
    # 볼록껍질(수치) → 넓이
    ptsf = sorted(((float(a), float(b)) for a, b in pts))
    def cross(o, a, b):
        return (a[0] - o[0])*(b[1] - o[1]) - (a[1] - o[1])*(b[0] - o[0])
    lower, upper = [], []
    for p in ptsf:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)
    for p in reversed(ptsf):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    area = 0.5*abs(sum(hull[i][0]*hull[(i + 1) % len(hull)][1] - hull[(i + 1) % len(hull)][0]*hull[i][1] for i in range(len(hull))))
    V.report(2, len(hull) == 6, f"그림자 꼭짓점 수 {len(hull)} (정육각형)")
    V.check(2, "2-2 그림자 넓이", sp.Float(area, 15), sp.N(S(V.ans(2, "2-2")), 15), 1e-9)
    # 정육각형 변 길이 모두 같은지
    sides = [math.dist(hull[i], hull[(i + 1) % 6]) for i in range(6)]
    V.report(2, max(sides) - min(sides) < 1e-9 and abs(sides[0] - 2*math.sqrt(3)/3) < 1e-9, f"변 길이 {sides[0]:.6f}")


def p03():
    a, b = sp.symbols("a b")
    ex = sp.sympify(V.ans(3, "3-1"), locals={"a": a, "b": b})
    # 정확 계산: 개수 j~Bin(100,1/3), j 홀수 확률
    p = sum(sp.Rational(math.comb(100, j)*2**(100 - j), 3**100) for j in range(1, 101, 2))
    V.check(3, "홀수일 확률 p", p, sp.Rational(1, 2) - sp.Rational(1, 2*3**100))
    V.check(3, "3-1 E(X)", a*p + b*(1 - p), ex)
    # 작은 N 에서 일반식 p=(1-3^-N)/2 확인(전수)
    for N in (2, 3, 5):
        cnt = sum(1 for roll in itertools.product(range(1, 7), repeat=N) if sum(1 for r in roll if r in (1, 2)) % 2 == 1)
        V.check(3, f"주사위 {N}개 전수", sp.Rational(cnt, 6**N), (1 - sp.Rational(1, 3**N))/2)


def p04():
    # 정의대로 직접: a_n = #{(x,y) 자연수 : y = n/2^x, y <= 2^x}, 합은 n=1..N
    def direct(N):
        tot = 0
        for n in range(1, N + 1):
            x_ = 1
            while 2**x_ <= n:
                if n % 2**x_ == 0 and n//2**x_ <= 2**x_:
                    tot += 1
                x_ += 1
        return tot
    for M in (3, 4, 5, 6):
        N = 2**(2*M) - 1
        V.check(4, f"M={M} 정의 직접 합 {direct(N)}", direct(N), 3*2**M - M - 4)
    # N=2^2018-1: (x,y) 격자점 개수 = Σ_x min(2^x, floor(N/2^x))  (정수 계산)
    N = 2**2018 - 1
    tot = 0
    x_ = 1
    while N//2**x_ >= 1:
        tot += min(2**x_, N//2**x_)
        x_ += 1
    V.check(4, "4-1 N=2^2018-1", tot, V.ans(4, "4-1"))


if __name__ == "__main__":
    V.run(globals())
