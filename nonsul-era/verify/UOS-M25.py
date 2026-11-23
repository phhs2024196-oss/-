import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import re
import numpy as np
import itertools
V = Verifier("UOS-M25", "items/UOS/PAST")


def p01():
    a, b, tt = sp.symbols("a b tt", positive=True)
    A = sp.Matrix([0, 0, 0]); B = sp.Matrix([a, a, 0])
    C = sp.Matrix([3*a/4, a/4, a]); D = sp.Matrix([a/4, 3*a/4, a])
    V_ = [A, B, C, D]
    # 평면 z=tt 와 사면체 모서리의 교점(z 가 다른 모서리만)
    pts = []
    for P_, Q_ in itertools.combinations(V_, 2):
        if sp.simplify(P_[2] - Q_[2]) != 0:
            s_ = sp.simplify((tt - P_[2]) / (Q_[2] - P_[2]))
            pts.append(sp.simplify(P_ + s_*(Q_ - P_)))
    V.report(1, len(pts) == 4, f"단면 꼭짓점 {len(pts)}개")
    # 볼록 사각형 면적: 중심각 순으로 정렬 후 신발끈 (구체값으로 순서 결정)
    num = {a: 4, tt: 1}
    cx = sum(float(p[0].subs(num)) for p in pts)/4; cy = sum(float(p[1].subs(num)) for p in pts)/4
    pts.sort(key=lambda p: np.arctan2(float(p[1].subs(num)) - cy, float(p[0].subs(num)) - cx))
    area = sp.Rational(1, 2)*sum(pts[i][0]*pts[(i+1) % 4][1] - pts[(i+1) % 4][0]*pts[i][1] for i in range(4))
    V.check(1, "단면적", sp.simplify(area if float(area.subs({a: 4, tt: 1})) > 0 else -area), tt*(a - tt))
    vol = sp.integrate(tt*(a - tt), (tt, 0, b))
    V.check(1, "부피", vol, sp.sympify(V.ans(1, "1-1"), locals={"a": a, "b": b}))
    # 전체 사면체 부피와 b=a 일치
    V.check(1, "b=a 전체부피", vol.subs(b, a), sp.Abs(sp.Matrix.hstack(B - A, C - A, D - A).det())/6)


def p02():
    n_ = sp.Symbol("n"); odd = [1, 3, 5, 7, 9]
    for N in range(3, 8):
        # 숫자 = 홀수만으로 서로 다른 3개가 쓰이는 첫 자리 1 인 N 자리 수 개수(공식 해석: 홀수만 사용)
        k = 0
        for tri in itertools.combinations(odd, 3):
            if 1 not in tri: continue
            for s_ in itertools.product(tri, repeat=N):
                if s_[0] == 1 and set(s_) == set(tri): k += 1
        V.check(2, f"n={N} k", k, 2*3**N - 3*2**(N+1) + 6)
        # 확률: 임의의 순서로 서로 다른 번호를 시도, 3번째 시도 안에 맞힘
        pr = sp.Rational(3, k)
        V.check(2, f"n={N} 확률", pr, S(V.ans(2, "2-1")).subs(n_, N))


def p03():
    th = sp.symbols("th", positive=True)
    def S_of(t):
        t = sp.Float(t, 40)
        # 직선 y = tan(t)(x+2) 와 단위원의 교점(오른쪽=P)
        m = sp.tan(t)
        xs = sp.solve(sp.Eq(x**2 + (m*(x + 2))**2, 1), x)
        xp = max(xs)
        s1 = sp.integrate(m*(x + 2), (x, -2, xp))
        s2 = sp.integrate(sp.sqrt(1 - x**2), (x, xp, 1))
        return sp.N(s1 + s2, 30), xp
    expected = S(V.ans(3, "3-1"))
    vals = []
    for t in ["0.01", "0.001", "0.0001"]:
        s_, xp = S_of(t)
        vals.append(s_/sp.Float(t, 30))
    V.check(3, "S/θ θ=1e-4", vals[-1], float(expected), 2e-3)
    V.report(3, abs(vals[-1] - expected) < abs(vals[0] - expected), "θ 감소 시 수렴")
    # 공식 해답의 S(θ) 식과 직접 적분 일치(θ=0.3)
    t = sp.Float("0.3", 30); s_, xp = S_of("0.3")
    # α: 직선과 원의 가까운 교점 T 의 중심각
    m = sp.tan(t); xs = sorted(sp.solve(sp.Eq(x**2 + (m*(x + 2))**2, 1), x)); xt = xs[0]
    al = sp.acos(-xt)       # ∠TOQ (T 가 제2사분면)
    Sform = sp.sin(2*t + al) + t + al/2
    V.check(3, "S(θ) 식", sp.N(Sform, 20), sp.N(s_, 20), 1e-12)


def p04():
    g = lambda xx, a_: 20*(a_ - 1)*xx - (4 + a_) + (4 + a_)*np.cos(xx)
    xs = np.linspace(1e-6, 60, 600001)
    V.proof(4, "(a) g>0 (a=2..12)", [a_ for a_ in range(2, 13) if not np.all(g(xs, a_) > 0)])
    # (b) 교점이 1개인 (a,b) 조사: 교점 = f(x)=10(a-1)x^2-(4+a)x+b sin x 의 영점
    def unique_zero(a_, b_):
        L = 12 + (4 + a_ + b_)
        pos = np.linspace(1e-3, L, 400001); neg = np.linspace(-L, -1e-3, 400001)
        f = lambda xx: 10*(a_ - 1)*xx**2 - (4 + a_)*xx + b_*np.sin(xx)
        for grid in (pos, neg):
            v = f(grid)
            if np.any(np.sign(v[:-1])*np.sign(v[1:]) <= 0): return False
        return True
    ans = V.ans(4, "4-2")
    fixed = {tuple(map(int, m)) for m in re.findall(r"\((\d+),(\d+)\)", ans)}
    fam = re.search(r"\(k,k\+(\d+)\)", ans); off = int(fam.group(1))
    exp = set(fixed)
    found = set()
    for a_ in range(1, 9):
        for b_ in range(1, 31):
            if unique_zero(a_, b_): found.add((a_, b_))
    exp |= {(k_, k_ + off) for k_ in range(2, 9) if k_ + off <= 30}
    V.report(4, found == exp, f"(b) 순서쌍 일치 found={sorted(found)}")
    # a=1: b=6 이상이면 영점 3개 이상
    V.proof(4, "(a=1,b>=6) 교점>1", [b_ for b_ in range(6, 20) if unique_zero(1, b_)])


if __name__ == "__main__": V.run(globals())
