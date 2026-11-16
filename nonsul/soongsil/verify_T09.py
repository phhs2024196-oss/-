#!/usr/bin/env python3
"""verify_T09.py — S09 삼각함수와 도형 검산 (sympy / mpmath)"""
import sys
import sympy as sp
import mpmath as mp

mp.mp.dps = 30
RES = []


def report(pid, ok, msg):
    RES.append(ok)
    print(f"{pid} {'PASS' if ok else 'FAIL'}  {msg}")


# S09-01: AB=5, AC=3, A=120deg, D = 호 BC(A 미포함)의 중점. AD = 8
def p01():
    b, c = 3, 5
    a = sp.sqrt(b**2 + c**2 - 2*b*c*sp.cos(2*sp.pi/3))
    R = a / (2*sp.sin(2*sp.pi/3))
    # 좌표로 독립 계산: 외접원 중심 O=(0,0), B,C 를 x축 대칭으로 두고 A 를 위쪽 호에
    # 점 D 는 호 BC 의 중점(아래쪽)
    t = sp.symbols('t', real=True)
    # B=(R cos(-phi), R sin(-phi)) ... 중심각 BOC = 2*(pi - A) 아님 -> 원주각 A=120 이므로 A 미포함 호 BC 의 중심각은 240이 아니라
    # A 가 큰 호 위에 있으려면... 수치로 직접: 원 위 세 점을 각도로 잡고 조건을 만족하는 배치를 찾는다
    Rn = mp.mpf(7)/mp.sqrt(3)
    def chord(u, v):
        return 2*Rn*abs(mp.sin((u - v)/2))
    # B 각 0, C 각 gamma, A 각 alpha (A 는 B-C 짧은 호 위: 원주각 120)
    # BC = 7 -> 중심각 = 2 asin(7/(2R))
    g = 2*mp.asin(7/(2*Rn))  # 짧은 쪽 호 중심각 (A 가 이 호 위에 있어야 원주각 120)
    alpha = mp.findroot(lambda al: chord(al, 0) - 5, g/3)
    ok_ac = abs(chord(alpha, g) - 3) < 1e-20
    # D: A 를 포함하지 않는 호(긴 호)의 중점: 각 g/2 + pi
    d = g/2 + mp.pi
    AD = chord(alpha, d)
    ok = ok_ac and abs(AD - 8) < 1e-20 and sp.simplify(a - 7) == 0
    # 대표 오답: D 를 A 쪽 호의 중점으로 잡은 경우
    wrong = 2*Rn  # AD 를 지름으로 착각
    report("S09-01", ok, f"BC={a}, R={sp.nsimplify(R)}, AD={mp.nstr(AD,15)} (오답 AD=2R: {mp.nstr(wrong,8)})")


# S09-02: 외접원 반지름 1, A=60, AB+2AC 최댓값 = 2*sqrt(7)
def p02():
    C = sp.symbols('C', real=True)
    B = 2*sp.pi/3 - C
    F = 2*sp.sin(C) + 4*sp.sin(B)
    crit = sp.solve(sp.diff(F, C), C)
    vals = [sp.simplify(F.subs(C, cc)) for cc in crit if 0 < float(cc) < float(2*sp.pi/3)]
    m = max(vals, key=lambda v: float(v))
    # 경계값보다 큼을 확인
    ok = sp.simplify(m - 2*sp.sqrt(7)) == 0 and float(m) > float(F.subs(C, 0)) and float(m) > float(F.subs(C, 2*sp.pi/3))
    wrong = float(F.subs(C, sp.pi/3))  # 정삼각형이라 착각
    report("S09-02", ok, f"max AB+2AC = {sp.nsimplify(m)} (오답: 정삼각형 가정 {wrong:.4f}=3sqrt3)")


# S09-03: AB=6, C=30deg, BC=a(자연수). 두 삼각형 존재하는 a 에 대한 AC 곱의 합 = 235
def p03():
    b = sp.symbols('b', real=True)
    total = 0
    two = []
    for a in range(1, 200):
        sols = sp.solve(sp.Eq(36, a**2 + b**2 - 2*a*b*sp.cos(sp.pi/6)), b)
        pos = sorted({sp.nsimplify(s) for s in sols if s.is_real and float(s) > 1e-12}, key=float)
        if len(pos) == 2:
            two.append(a)
            total += sp.simplify(pos[0]*pos[1])
    ok = two == list(range(7, 12)) and total == 235
    wrong = sum(a*a - 36 for a in range(7, 13))  # a=12 포함 오답
    report("S09-03", ok, f"두 삼각형 a={two}, 합={total} (오답 a=12 포함: {wrong})")


# S09-04: AB+AC=9, 이등분선 AD=2, BC=3sqrt7 -> 넓이 9sqrt3/2
def p04():
    b, c, A = sp.symbols('b c A', positive=True)
    eqs = [b + c - 9,
           2*b*c*sp.cos(A/2)/(b + c) - 2,
           b**2 + c**2 - 2*b*c*sp.cos(A) - 63]
    sol = sp.nsolve(eqs, [b, c, A], [3.2, 5.8, 2.0])
    bb, cc, AA = [sp.Float(v, 30) for v in sol]
    area = bb*cc*sp.sin(AA)/2
    ok = abs(AA - 2*sp.pi/3) < 1e-20 and abs(area - 9*sp.sqrt(3)/2) < 1e-20 and {round(float(bb)), round(float(cc))} == {3, 6}
    report("S09-04", ok, f"b,c={float(bb):.6f},{float(cc):.6f}, A={float(AA):.6f}, 넓이={float(area):.10f}=9sqrt3/2")


# S09-05: 반지름 1 원에 내접하는 ABCD, 각 BAD=120 -> 둘레 최댓값 2+2sqrt3
def p05():
    # 원 위 B=-th0, D=+th0 (BD=sqrt3), A 는 짧은 호(원주각 120), C 는 긴 호
    th0 = mp.mpf(2)*mp.pi/3  # 중심각 BOD (짧은쪽) = 2*60 = 120
    def ch(u, v):
        return 2*abs(mp.sin((u - v)/2))
    best = 0
    N = 2000
    # A 각 phi in (-pi/3, pi/3) (짧은 호), C 각 psi in (pi/3, 5pi/3)
    fa = lambda p: ch(p, -mp.pi/3) + ch(p, mp.pi/3)
    fc = lambda q: ch(q, -mp.pi/3) + ch(q, mp.pi/3)
    ma = max(fa(-mp.pi/3 + 2*mp.pi/3*k/N) for k in range(1, N))
    mc = max(fc(mp.pi/3 + 4*mp.pi/3*k/N) for k in range(1, N))
    # 각 확인: A 에서 각 BAD
    p = mp.mpf('0.1')
    v1 = mp.matrix([mp.cos(-mp.pi/3) - mp.cos(p), mp.sin(-mp.pi/3) - mp.sin(p)])
    v2 = mp.matrix([mp.cos(mp.pi/3) - mp.cos(p), mp.sin(mp.pi/3) - mp.sin(p)])
    ang = mp.acos((v1[0]*v2[0] + v1[1]*v2[1]) / (mp.norm(v1)*mp.norm(v2)))
    tot = ma + mc + 0
    target = 2 + 2*mp.sqrt(3)
    ok = abs(ang - 2*mp.pi/3) < 1e-20 and abs(ma - 2) < 1e-5 and abs(mc - 2*mp.sqrt(3)) < 1e-5 \
        and abs(fa(0) - 2) < 1e-25 and abs(fc(mp.pi) - 2*mp.sqrt(3)) < 1e-25
    report("S09-05", ok, f"max(AB+AD)={mp.nstr(ma,10)}, max(CB+CD)={mp.nstr(mc,10)}, 둘레최대={mp.nstr(fa(0)+fc(mp.pi),15)}=2+2sqrt3")


# S09-06: y=sin(pi x), y=sin(pi x/2), 0<=x<=4 로 둘러싸인 넓이 합 = 10/pi
def p06():
    x = sp.symbols('x', real=True)
    f = sp.sin(sp.pi*x) - sp.sin(sp.pi*x/2)
    roots = sorted(set(sp.solveset(f, x, sp.Interval(0, 4))), key=float)
    A = sum(sp.Abs(sp.integrate(f, (x, roots[i], roots[i+1]))) for i in range(len(roots) - 1))
    A = sp.simplify(A)
    wrong = sp.simplify(sp.integrate(f, (x, 0, 4)))
    ok = roots == [0, sp.Rational(2, 3), 2, sp.Rational(10, 3), 4] and sp.simplify(A - 10/sp.pi) == 0
    report("S09-06", ok, f"교점 x={roots}, 넓이={A} (오답: 절댓값 없이 적분 {wrong})")


# S09-07: (1, 3sqrt3) 지나는 직선, 양의 x,y 절편 Q,R, QR 최솟값 8
def p07():
    t = sp.symbols('t', positive=True)  # 직선과 x축이 이루는 각
    L = 1/sp.cos(t) + 3*sp.sqrt(3)/sp.sin(t)
    crit = [c for c in sp.solve(sp.diff(L, t), t) if c.is_real and 0 < float(c) < float(sp.pi/2)]
    vals = [sp.simplify(L.subs(t, c)) for c in crit]
    # 수치 격자 최솟값과 비교
    grid = min(float(L.subs(t, sp.Rational(k, 10000)*sp.pi/2)) for k in range(1, 10000, 7))
    ok = len(vals) == 1 and sp.simplify(vals[0] - 8) == 0 and grid >= 8 - 1e-9
    report("S09-07", ok, f"임계점 t={crit}, QR 최솟값={vals[0]}, 격자 최소={grid:.8f}")


# S09-08: tanA:tanB:tanC=1:2:3, AB=3sqrt2 -> 넓이 6
def p08():
    k = sp.symbols('k', real=True)
    # tan(A+B) = -tan C
    sols = sp.solve(sp.Eq((k + 2*k)/(1 - 2*k**2), -3*k), k)
    valid = []
    for s in sols:
        if s == 0:
            continue
        A, B, C = [sp.atan(m*s) for m in (1, 2, 3)]
        # 음수이면 각을 (pi/2, pi) 로
        ang = [a if float(a) > 0 else a + sp.pi for a in (A, B, C)]
        if abs(float(sum(ang)) - float(sp.pi)) < 1e-12:
            valid.append((s, ang))
    s, (A, B, C) = valid[0]
    R = 3*sp.sqrt(2)/(2*sp.sin(C))
    area = sp.simplify(2*R**2*sp.sin(A)*sp.sin(B)*sp.sin(C))
    ok = len(valid) == 1 and s == 1 and area == 6
    report("S09-08", ok, f"유효한 tanA={s} (해 후보 {sols}), 넓이={area}")


# S09-09: 반지름 1 원에 내접하는 정삼각형 ABC, P 는 호 BC(A 미포함) 위. PA*PB*PC 최댓값 2
def p09():
    th = sp.symbols('th', real=True)  # 각 PBC... 매개: PB=2sin th, PC=2sin(pi/3-th), PA=2sin(pi/3+th)
    PB, PC, PA = 2*sp.sin(th), 2*sp.sin(sp.pi/3 - th), 2*sp.sin(sp.pi/3 + th)
    inv = sp.simplify(sp.expand_trig(PA - PB - PC))
    # 좌표로 독립 확인 (무작위 20점)
    import random
    okc = True
    for _ in range(20):
        u = random.uniform(0.01, 2*mp.pi/3 - 0.01)
        A_ = (mp.cos(mp.pi/2), mp.sin(mp.pi/2))
        B_ = (mp.cos(mp.pi/2 + 2*mp.pi/3), mp.sin(mp.pi/2 + 2*mp.pi/3))
        C_ = (mp.cos(mp.pi/2 + 4*mp.pi/3), mp.sin(mp.pi/2 + 4*mp.pi/3))
        P_ = (mp.cos(mp.pi/2 + 2*mp.pi/3 + u), mp.sin(mp.pi/2 + 2*mp.pi/3 + u))
        d = lambda X, Y: mp.sqrt((X[0]-Y[0])**2 + (X[1]-Y[1])**2)
        okc &= abs(d(P_, A_) - d(P_, B_) - d(P_, C_)) < 1e-20
    F = PA*PB*PC
    crit = [c for c in sp.solve(sp.diff(F, th), th) if 0 < float(c) < float(sp.pi/3)]
    vals = [sp.simplify(F.subs(th, c)) for c in crit]
    m = max(vals, key=float)
    ok = inv == 0 and okc and sp.simplify(m - 2) == 0
    report("S09-09", ok, f"[증명-수치확인] PA=PB+PC 항등식/무작위20점, PA*PB*PC 최댓값={m}")


# S09-10: x(pi-x) k <= sin x on [0,pi] 의 k 최댓값 = 1/pi
def p10():
    xs = [mp.pi*j/1000 for j in range(1, 1000)]
    ok_ineq = all(mp.pi*mp.sin(v) - mp.pi*v + v*v >= -1e-25 for v in xs)
    h = lambda v: mp.sin(v)/(v*(mp.pi - v))
    inf_h = min(h(v) for v in xs)
    lim0 = mp.limit(h, 0)
    # k > 1/pi 이면 반례 존재
    kk = 1/mp.pi + mp.mpf('1e-6')
    bad = any(kk*v*(mp.pi - v) > mp.sin(v) for v in [mp.mpf(10)**(-j) for j in range(1, 10)])
    ok = ok_ineq and abs(lim0 - 1/mp.pi) < 1e-20 and inf_h >= 1/mp.pi - 1e-25 and bad
    report("S09-10", ok, f"[증명-수치확인] 격자999점 부등식 성립, inf h={mp.nstr(inf_h,12)}, lim={mp.nstr(lim0,12)}=1/pi, (오답 4/pi^2={mp.nstr(4/mp.pi**2,6)})")


for fn in (p01, p02, p03, p04, p05, p06, p07, p08, p09, p10):
    try:
        fn()
    except Exception as e:  # noqa
        report(fn.__name__.replace('p', 'S09-'), False, f"예외: {e!r}")

n = sum(RES)
print(f"SUMMARY T09: {n}/10 PASS")
sys.exit(0 if n == 10 else 1)
