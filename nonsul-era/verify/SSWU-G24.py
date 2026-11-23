import sys, pathlib, re, math, itertools; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
V = Verifier("SSWU-G24", "items/SSWU/PAST")
a, b, tt, u = sp.symbols("a b tt u", real=True)

def tangents_from(px, py, tol=1e-9):
    """곡선 y=x^3-3x^2 에 점 (px,py)에서 그은 접선의 접점 x좌표 집합: f(t)+f'(t)(px-t)=py"""
    t_ = sp.symbols("t_")
    f = t_**3 - 3*t_**2
    eq = sp.expand(f + sp.diff(f, t_)*(px - t_) - py)
    rts = sp.Poly(eq, t_).nroots(n=30, maxsteps=200)
    out = []
    for r in rts:
        if abs(sp.im(r)) < 1e-8:
            r = float(sp.re(r))
            if not any(abs(r - q) < 1e-6 for q in out): out.append(r)
    return sorted(out)

def p01():
    lo, hi = [sp.Rational(s) for s in re.findall(r"[\d/]+", V.ans(1, "1"))]  # 1/3<a<3
    ok = True
    for k in range(-40, 80):
        av = sp.Rational(k, 13)
        n1 = len(tangents_from(av, 0))
        if (n1 == 1) != (lo < av < hi): ok = False
    for av in (lo, hi): ok &= len(tangents_from(av, 0)) == 2   # 경계는 2개
    V.report(1, ok, "1-1 접선 1개인 a의 범위 (격자 대조)")
    ans2 = {sp.Rational(s) for s in V.ans(1, "2").replace(" ", "").split(";")}
    sols = set()
    for k in range(-300, 301):
        av = sp.Rational(k, 60)
        ts = tangents_from(av, 0)
        if abs(sum(ts) - 1) < 1e-7: sols.add(av)
    V.report(1, sols == ans2, f"1-2 접점 x좌표 합=1 인 a: 격자해 {sols} / YAML {ans2}")
    lo3, hi3 = [sp.Rational(s) for s in re.findall(r"[\d/]+", V.ans(1, "3"))]
    ok = True
    for k in range(-30, 40):
        bv = sp.Rational(k, 20)
        n3 = len(tangents_from(0, bv))
        if (n3 == 3) != (lo3 < bv < hi3): ok = False
    V.report(1, ok, "1-3 (0,b)에서 접선 3개인 b의 범위 (격자 대조)")

def clip(poly, cl):
    """볼록 다각형 poly 를 볼록 다각형 cl(반시계)로 자르기(Sutherland-Hodgman)"""
    out = poly
    for i in range(len(cl)):
        A, B = cl[i], cl[(i + 1) % len(cl)]
        inp, out = out, []
        side = lambda P: (B[0]-A[0])*(P[1]-A[1]) - (B[1]-A[1])*(P[0]-A[0])
        for j in range(len(inp)):
            P, Q = inp[j], inp[(j + 1) % len(inp)]
            sp_, sq = side(P), side(Q)
            if sp_ >= 0: out.append(P)
            if sp_ * sq < 0:
                r = sp_ / (sp_ - sq); out.append((P[0] + r*(Q[0]-P[0]), P[1] + r*(Q[1]-P[1])))
        if not out: return []
    return out
area = lambda P: abs(sum(P[i][0]*P[(i+1)%len(P)][1] - P[(i+1)%len(P)][0]*P[i][1] for i in range(len(P)))) / 2
peri = lambda P: sum(math.dist(P[i], P[(i+1)%len(P)]) for i in range(len(P)))

def tri_abc(t):  # C=(t,0), AB 는 x=t-3/2 (변 길이 sqrt3, 높이 3/2)
    s = math.sqrt(3)/2
    return [(t-1.5, -s), (t, 0.0), (t-1.5, s)]   # B, C, A (반시계)
DEF = [(0.0, 0.0), (1.5, -math.sqrt(3)/2), (1.5, math.sqrt(3)/2)]  # F, D, E (반시계)

def p02():
    ex = V.ans(2, "1")
    f1, f2 = [S(s.split("=", 1)[1]) for s in ex.replace(" ", "").split(";")]
    ok = True
    for k in range(1, 300):
        tv = 3*k/300
        fa = area(clip(tri_abc(tv), DEF))
        fe = float((f1 if tv <= 1.5 else f2).subs(sp.Symbol("t"), tv))
        if abs(fa - fe) > 1e-9: ok = False
    V.report(2, ok, "2-1 f(t) 구간별식 vs 다각형 교집합 넓이(299점)")
    fm = sp.Piecewise((f1, t <= sp.Rational(3, 2)), (f2, True))
    T_ = sp.Symbol("t"); tm = max(sp.solveset(sp.diff(f2, T_), T_, sp.Interval(1.5, 3)))
    mx = f2.subs(T_, tm)
    V.check(2, "2-2 최댓값", mx, V.ans(2, "2", "max"))
    ts = [3*k/20000 for k in range(1, 20000)]
    ok = max(area(clip(tri_abc(x_), DEF)) for x_ in ts) - float(mx) < 1e-6
    per = peri(clip(tri_abc(float(tm)), DEF))
    V.check(2, "2-2 둘레(교집합 다각형)", sp.Float(per, 20), V.ans(2, "2", "perimeter"), 1e-8)
    V.report(2, ok, "2-2 최댓값 격자 확인")
    # (3) 정육각형 반복
    R = 1.0
    hexv = [(R*math.cos(math.pi/3*k + math.pi/3), R*math.sin(math.pi/3*k + math.pi/3)) for k in range(6)]
    tot, Ss = 0.0, []
    for _ in range(40):
        t1 = [hexv[0], hexv[2], hexv[4]]; t2 = [hexv[1], hexv[3], hexv[5]]
        inter = clip(t1, t2)
        Sn = area(t1) + area(t2) - area(inter) - area(inter)
        Ss.append(Sn); tot += Sn; hexv = inter
    ratio_ok = all(abs(Ss[i+1]/Ss[i] - 1/3) < 1e-9 for i in range(10))
    V.check(2, "2-3 sum S_n (40항, 공비 1/3 확인)", sp.Float(tot, 20), V.ans(2, "3"), 1e-9)
    V.report(2, ratio_ok, "2-3 S_{n+1}/S_n = 1/3")

def p03():
    n_ = sp.symbols("n_", positive=True)
    tpos = sp.symbols("tpos", positive=True)
    A = sp.Matrix([0, 1])
    C = sp.Matrix([tpos, n_*tpos]); B = -2*C
    ang = lambda P: sp.acos((sp.Matrix([0, -1]).dot(P - A))/ (sp.sqrt((P - A).dot(P - A))))   # OAP 의 각 (AO 방향과 AP)
    # 각 OAB = OAC 이면 cos 같음 -> 부호 조심해 제곱 대신 수치로 t 구하고 닫힌식과 비교
    def tn(nv):
        e = sp.cos(ang(B)) - sp.cos(ang(C))
        e = e.subs(n_, nv)
        return sp.nsolve(e, tpos, 1/(4*nv) + 0.01) if False else sp.nsolve(e, tpos, sp.Rational(1, 4*nv)*sp.Rational(11, 10))
    ok, okS = True, True
    sv = {}
    for nv in range(1, 8):
        tv = tn(nv)
        Cn = (tv, nv*tv); Bn = (-2*tv, -2*nv*tv)
        th = math.acos(-(Bn[1]-1)/math.hypot(Bn[0], Bn[1]-1)); al = math.atan2(Cn[0], Cn[1])
        thC = math.acos(-(Cn[1]-1)/math.hypot(Cn[0], Cn[1]-1))
        ok &= abs(th - thC) < 1e-9 and abs(math.tan(al) - 1/nv) < 1e-9 and abs(math.sin(al) - 1/math.sqrt(nv**2+1)) < 1e-9
        Ar = abs(0*(Bn[1]-Cn[1]) + Bn[0]*(Cn[1]-1) + Cn[0]*(1-Bn[1]))/2
        sv[nv] = Ar
    V.report(3, ok, "3-1 sin·tan α_n 및 θ_n 조건(n=1..7)")
    expr = S(V.ans(3, "1", "tan")); V.check(3, "3-1 tan α_n=1/n (n=3)", sp.Rational(1, 3), expr.subs(sp.Symbol("n"), 3))
    # 2) 극한: n 크게
    mp = sp.nsolve
    nv = 10**6
    tv = sp.Rational(1, 4*nv)
    Cn = (tv, nv*tv); Bn = (-2*tv, -2*nv*tv)
    thC = sp.atan(Cn[0]/(1-Cn[1])); al = sp.atan(sp.Rational(1, nv))
    V.check(3, "3-2 θ_n/α_n (n=1e6)", sp.N(thC/al, 12), V.ans(3, "2"), 1e-5)
    # 3) S_n = 3/(8n) 는 n=1..7 기하 넓이와 비교, 합은 부분분수
    okS = all(abs(sv[k] - 3/(8*k)) < 1e-9 for k in sv)
    V.report(3, okS, "3-3 S_n(기하 넓이)=3/(8n), n=1..7")
    tot = sp.summation(sp.Rational(3, 8)/n * sp.Rational(3, 8)/(n+1), (n, 1, sp.oo))
    V.check(3, "3-3 합", tot, V.ans(3, "3"))

def p04():
    c1, c2 = sp.symbols("c1 c2", real=True)
    cnt = 0; ok = True; wrong = []
    ans2 = S(V.ans(4, "2")); ans3 = S(V.ans(4, "3"))
    for (A1, A3) in [(1, 0), (sp.Rational(1, 3), sp.Rational(1, 5)), (2, -1), (-sp.Rational(3, 2), sp.Rational(1, 2))]:
        g = A1*sp.sin(sp.pi*x) + A3*sp.sin(3*sp.pi*x)           # 기함수·주기2·g(1)=0
        k0 = -1/ sp.integrate(g, (x, 0, 1)) if sp.integrate(g, (x, 0, 1)) != 0 else None
        g = g*k0                                                      # ∫_0^1 g = -1 로 정규화
        f = 2 + g                                                     # (가)(나) 성립
        conds = [sp.simplify(f + f.subs(x, -x) - 4) == 0, sp.simplify(f.subs(x, x+2) - f) == 0,
                 same(sp.integrate(x*sp.diff(f, x), (x, 0, 1)), 1)]
        h = f - 2
        r2 = sp.integrate(h, (x, 0, 1)); r3 = sp.integrate(f, (x, 4, 9))
        ok &= all(conds) and same(r2, ans2) and same(r3, ans3)
        zeros = [z for z in sp.Poly(sp.expand_trig(sp.diff(f, x).rewrite(sp.exp)), x).nroots()] if False else None
        # f'(x)=0 이 (-1,1)에 2개 이상: 부호변화 격자
        fp = sp.lambdify(x, sp.diff(f, x), "math")
        xs = [-1 + 2*i/4000 for i in range(1, 4000)]
        sc = sum(1 for i in range(len(xs)-1) if fp(xs[i])*fp(xs[i+1]) < 0)
        ok &= sc >= 2
    V.report(4, ok, "4-1 [증명-수치확인] 예시 f 4종 모두 f'의 부호변화≥2, 4-2 ∫h=-1, 4-3 ∫f=9")
    V.report(4, ok, "4-2·4-3 조건 (가)(나)(다)를 만족하는 f 에서 직접 적분")

if __name__ == "__main__": V.run(globals())
