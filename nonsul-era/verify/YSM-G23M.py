import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *

V = Verifier("YSM-G23M", "items/YSM/PAST")
E = lambda v: sp.sympify(v, locals={"x": x, "n": n, "k": k})

def p01():
    nn = sp.Symbol("nn", positive=True)
    f = lambda z: sp.exp(nn*z) + sp.exp(-nn*z)
    # P: f'(x)=2n → x=a_n
    an = [s for s in sp.solve(sp.Eq(sp.diff(f(x), x), 2*nn), x)]
    an = [s for s in an if s.is_real][0] if any(s.is_real for s in an) else an[0]
    V.check(1, "1-1 a_n", sp.simplify(an.subs(nn, 3)), sp.log(1 + sp.sqrt(2))/3)
    # α = lim_{n→0}(1/a_n+1)^{1/n}: a_n=L/n, 로그를 취해 극한
    L = sp.log(1 + sp.sqrt(2)); u = sp.Symbol("u", positive=True)
    lnalpha = sp.limit((1/u)*sp.log(1/(L/u) + 1), u, 0, "+")   # u=n
    V.check(1, "1-1 ln α = 1/ln(1+√2)", lnalpha, 1/L)
    alpha = sp.exp(lnalpha)
    # g'(x) 증가, g'(k) = e^{1/lnα}/3 → k
    gp = sp.diff(sp.log(f(x))/nn, x)
    V.report(1, all(float(sp.diff(gp, x).subs({nn: 2, x: t})) > 0 for t in [-3, -1, 0, 0.5, 2, 5]), "1-1 g' 증가(g''>0)")
    kk = sp.Symbol("kk", positive=True)
    ksol = sp.solve(sp.Eq(gp.subs(x, kk), sp.exp(1/sp.log(alpha))/3), kk)[0]
    V.check(1, "1-1 k", sp.simplify(ksol), E(V.ans(1, "1-1")).subs(n, nn))
    # 1-2: n=1..30 에서 넓이 최대는 n=1, S, 접선 수직 조건 a,b
    area = lambda m: sp.Rational(1, 2)*2*sp.re(sp.N(sp.log(1 + sp.sqrt(2))/m, 30))*sp.N(2*sp.sqrt(2), 30)
    V.report(1, max(range(1, 31), key=lambda m: area(m)) == 1, "1-2 n=1 에서 OPQ 넓이 최대")
    S = 2*sp.sqrt(2)*sp.log(1 + sp.sqrt(2))
    Pp = (sp.log(1 + sp.sqrt(2)), f(x).subs({nn: 1, x: sp.log(1 + sp.sqrt(2))}))
    V.check(1, "1-2 P' y좌표", sp.simplify(Pp[1]), 2*sp.sqrt(2))
    slopeQ = sp.diff(f(x), x).subs({nn: 1, x: -sp.log(1 + sp.sqrt(2))})
    V.check(1, "1-2 Q' 기울기", sp.simplify(slopeQ), -2)
    a, b = sp.symbols("a b", positive=True)
    sol = sp.solve([sp.Eq(b**2, 12*a), sp.Eq((6/b)*slopeQ, -1)], [a, b], dict=True)[0]
    V.check(1, "1-2 b", sol[b], 12); V.check(1, "1-2 a", sol[a], 12)
    V.check(1, "1-2 S√(ab)", sp.simplify(S*sp.sqrt(sol[a]*sol[b])), E(V.ans(1, "1-2")))

def p02():
    import numpy as np
    from scipy import ndimage
    # 좌표: A=(0,0), B=(s,0), O1=(2,0), O2 = B 에서 BC 방향으로 2
    xx = sp.Symbol("xx", positive=True)
    sidesol = sp.solve(sp.Eq(4**2, xx**2 + 4 - 2*xx*2*sp.Rational(1, 2)), xx)[0]    # O1B
    V.check(2, "2-1 O1B", sidesol, sp.sqrt(13) + 1)
    s = sidesol + 2
    V.check(2, "2-1 한 변", s, sp.sqrt(13) + 3)
    sv = float(s)
    C = np.array([sv/2, sv*np.sqrt(3)/2]); B = np.array([sv, 0.0]); O1 = np.array([2.0, 0]); u = (C - B)/np.linalg.norm(C - B)
    O2 = B + 2*u
    V.report(2, abs(np.linalg.norm(O1 - O2) - 4) < 1e-12, "2-1 두 원 접함 |O1O2|=4")
    # 격자: 삼각형 내부에서 두 원에 덮이지 않은 영역의 연결성분 두 개
    h = 0.002
    gx, gy = np.meshgrid(np.arange(0, sv, h) + h/2, np.arange(0, C[1] + h, h) + h/2)
    inside = (gy >= 0) & (gy <= np.sqrt(3)*gx) & (gy <= np.sqrt(3)*(sv - gx))
    cov = ((gx - O1[0])**2 + gy**2 <= 4) | ((gx - O2[0])**2 + (gy - O2[1])**2 <= 4)
    free = inside & ~cov
    lab, nl = ndimage.label(free)
    areas = sorted(((lab == i).sum()*h*h, i) for i in range(1, nl + 1))
    big = [a for a in areas if a[0] > 1e-3]
    V.report(2, len(big) == 2, f"2-1 미살수 영역 성분 {len(big)}개(면적 {[round(a[0],4) for a in big]})")
    S1n, S2n = big[-1][0], big[0][0]
    S1 = 4*sp.sqrt(3) + sp.sqrt(39) - 2*sp.pi; S2 = sp.sqrt(39)/2 - sp.sqrt(3)/2 - 2*sp.pi/3
    V.check(2, "2-1 S1(YAML)", S1, E(V.ans(2, "2-1", "S1"))); V.check(2, "2-1 S2(YAML)", S2, E(V.ans(2, "2-1", "S2")))
    V.report(2, abs(S1n - float(S1)) < 5e-3 and abs(S2n - float(S2)) < 5e-3, f"2-1 격자 수치 S1={S1n:.4f}/{float(S1):.4f}, S2={S2n:.4f}/{float(S2):.4f}")
    # 2-2: C 에서 S1 의 가장 먼 점까지 거리 r (격자) 와 r^2 = 13+3√13
    # S1 의 경계: 원1 의 호(P1→D), 원2 의 호(D→Q2) 와 변 CP1, CQ2. 호 위에서 C 까지 거리² 의 최댓값
    D = (O1 + O2)/2
    a1 = np.arctan2(*(D - O1)[::-1]); p1 = np.pi*2/3                      # 원1: P1 은 O1 에서 120° 방향
    t1 = np.linspace(a1, p1, 200001); arc1 = O1[None, :] + 2*np.c_[np.cos(t1), np.sin(t1)]
    a2 = np.arctan2(*(D - O2)[::-1]); q2 = np.arctan2(*(C - O2)[::-1])    # 원2: Q2 는 O2 에서 C 방향
    a2 = a2 % (2*np.pi); t2 = np.linspace(q2, a2, 200001); arc2 = O2[None, :] + 2*np.c_[np.cos(t2), np.sin(t2)]
    d1 = np.sum((arc1 - C)**2, axis=1); d2 = np.sum((arc2 - C)**2, axis=1)
    r2n = float(max(d1.max(), d2.max()))
    V.report(2, abs(r2n - float(E(V.ans(2, "2-2")))) < 1e-6, f"2-2 r²(S1 경계 호 위 최댓값) {r2n:.6f} vs {float(E(V.ans(2, '2-2'))):.6f}")
    # 정확 계산: CD² (D = 두 원의 접점 = 중점 O1O2), CP1²
    D = (O1 + O2)/2
    CD2 = float(np.sum((C - D)**2)); CP12 = float((sv - 2)**2)
    V.check(2, "2-2 CD²", CD2, E(V.ans(2, "2-2")), tol=1e-9); V.check(2, "2-2 CP1²", CP12, 14 + 2*sp.sqrt(13), tol=1e-9)
    V.report(2, CD2 > CP12, "2-2 CD² > CP1² → r²=CD²")
    th = sp.atan2(sp.sqrt(3), sp.sqrt(13))
    V.check(2, "2-2 cos(θ+π/3)", sp.cos(th + sp.pi/3), (sp.sqrt(13) - 3)/8)

if __name__ == "__main__": V.run(globals())
