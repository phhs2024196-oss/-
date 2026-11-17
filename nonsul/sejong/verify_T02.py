#!/usr/bin/env python3
"""verify_T02.py -- J02. 단면이 주어진 입체의 부피와 정적분 (세종) 검산 스크립트."""
import sys, random
import sympy as sp
import mpmath as mp

mp.mp.dps = 30
random.seed(2)
results = []


def report(pid, ok, msg):
    results.append(bool(ok))
    print(f"{pid} {'PASS' if ok else 'FAIL'}  {msg}")


def close(a, b, tol=1e-9):
    a, b = complex(a), complex(b)
    return abs(a - b) <= tol * max(1.0, abs(b))


x, y, h, t, m, s = sp.symbols('x y h t m s', real=True)
tp = sp.symbols('tp', positive=True)


# ---------------------------------------------------------------- J02-01
def p01():
    A = sp.pi / 2 * (sp.exp(-x) / 2) ** 2          # 반원, 지름 e^{-x}
    V = sp.integrate(A, (x, 0, tp))
    ok1 = sp.simplify(V - sp.pi * (1 - sp.exp(-2 * tp)) / 16) == 0
    L = sp.limit(V, tp, sp.oo)
    ok2 = L == sp.pi / 16 and all(float(V.subs(tp, 0.01 * i)) < float(sp.pi / 16) for i in range(1, 3001))
    t0 = sp.solve(sp.Eq(V, sp.Rational(3, 4) * L), tp)
    ok3 = len(t0) == 1 and sp.simplify(t0[0] - sp.log(2)) == 0
    dV = sp.simplify(sp.diff(V, tp).subs(tp, sp.log(2)))
    ok3 &= sp.simplify(dV - sp.pi / 32) == 0
    report('J02-01', ok1 and ok2 and ok3, f"V(t)=pi(1-e^-2t)/16; V<pi/16 3000점[증명-수치확인]; t={t0[0]}, V'={dV}")


# ---------------------------------------------------------------- J02-02
def p02():
    hp = sp.symbols('hp', positive=True)
    yp = sp.symbols('yp', positive=True)
    Sec = (1 + sp.sqrt(yp)) ** 2
    V = sp.integrate(Sec, (yp, 0, hp))
    ok1 = sp.simplify(V - (hp + sp.Rational(4, 3) * hp ** sp.Rational(3, 2) + hp ** 2 / 2)) == 0
    Q = 15
    dhdt = Q / sp.diff(V, hp)            # dV/dt = V'(h) h'
    ok2 = sp.simplify(dhdt - Q / (1 + sp.sqrt(hp)) ** 2) == 0
    ok2 &= all(float(sp.diff(dhdt, hp).subs(hp, 0.01 + 0.01 * i)) < 0 for i in range(900))
    T4 = sp.nsimplify(V.subs(hp, 4) / Q)
    r1 = dhdt.subs(hp, 4)
    r2 = sp.simplify(sp.diff(Sec.subs(yp, hp), hp).subs(hp, 4) * r1)
    ok3 = T4 == sp.Rational(68, 45) and r1 == sp.Rational(5, 3) and r2 == sp.Rational(5, 2)
    report('J02-02', ok1 and ok2 and ok3, f"V(h) 일치; h' 감소[증명-수치확인]; 시각 {T4}, dh/dt={r1}, dS/dt={r2}")


# ---------------------------------------------------------------- J02-03
def p03():
    A = (x * sp.exp(-x)) ** 2 / 4    # 빗변이 c 인 직각이등변삼각형 넓이 c^2/4
    V = sp.integrate(A, (x, 0, tp))
    Vmd = sp.Rational(1, 16) - sp.exp(-2 * tp) * (2 * tp ** 2 + 2 * tp + 1) / 16
    ok1 = sp.simplify(V - Vmd) == 0
    ok2 = all(mp.e ** v > v ** 3 / 6 for v in [mp.mpf(i) / 50 for i in range(1, 1501)])
    L = sp.limit(Vmd, tp, sp.oo)
    ok3 = L == sp.Rational(1, 16)
    report('J02-03', ok1 and ok2 and ok3, f"V(t) 일치; e^x>x^3/6 1500점[증명-수치확인]; lim V={L}")


# ---------------------------------------------------------------- J02-04
def p04():
    V = sp.integrate(sp.sin(x) ** 2, (x, t, t + sp.pi / 3))
    ok1 = sp.simplify(V - (sp.pi / 6 - (sp.sin(2 * t + 2 * sp.pi / 3) - sp.sin(2 * t)) / 4)) == 0
    dV = sp.diff(V, t)
    ok2 = sp.simplify(dV - (sp.sin(t + sp.pi / 3) ** 2 - sp.sin(t) ** 2)) == 0
    crit = sp.solveset(sp.Eq(dV, 0), t, sp.Interval.open(0, 2 * sp.pi / 3))
    ok2 &= crit == sp.FiniteSet(sp.pi / 3)
    Vmax = sp.simplify(V.subs(t, sp.pi / 3))
    Vmin = sp.simplify(V.subs(t, 0))
    grid = [float(V.subs(t, i * (2 * sp.pi / 3) / 2000)) for i in range(2001)]
    ok3 = (sp.simplify(Vmax - (sp.pi / 6 + sp.sqrt(3) / 4)) == 0 and sp.simplify(Vmin - (sp.pi / 6 - sp.sqrt(3) / 8)) == 0
           and close(max(grid), float(Vmax)) and close(min(grid), float(Vmin))
           and sp.simplify(V.subs(t, 2 * sp.pi / 3) - Vmin) == 0)
    report('J02-04', ok1 and ok2 and ok3, f"임계점 pi/3; max={Vmax}, min={Vmin}")


# ---------------------------------------------------------------- J02-05
def p05():
    yp = sp.symbols('yp', positive=True)
    V = sp.integrate(x * sp.exp(x), (x, 0, 1))      # x 단면: 가로 e^x, 높이 x
    ok1 = V == 1
    # y 단면: 0<=y<=1 이면 x in [0,1], 1<=y<=e 이면 x in [ln y,1]; 단면 넓이 = int x dx
    B1 = sp.integrate(x, (x, 0, 1))
    B2 = sp.integrate(x, (x, sp.log(yp), 1))
    ok2 = B1 == sp.Rational(1, 2) and sp.simplify(B2 - (1 - sp.log(yp) ** 2) / 2) == 0
    Vy = B1 * 1 + sp.integrate(B2, (yp, 1, sp.E))
    ok2 &= sp.simplify(Vy - 1) == 0
    up = sp.simplify(sp.integrate(B2, (yp, sp.sqrt(sp.E), sp.E)))
    # 독립 확인: 3차원 직접(몬테카를로 대신 반복적분 수치)
    num = mp.quad(lambda xx: mp.quad(lambda yy: xx, [mp.sqrt(mp.e), mp.e ** xx]) if mp.e ** xx > mp.sqrt(mp.e) else 0, [0.5, 1])
    ok3 = sp.simplify(up - sp.sqrt(sp.E) / 8) == 0 and close(num, mp.sqrt(mp.e) / 8)
    report('J02-05', ok1 and ok2 and ok3, f"V=1 (x단면)=(y단면) [증명-수치확인]; 윗부분={up}, 수치 {mp.nstr(num, 12)}")


# ---------------------------------------------------------------- J02-06
def p06():
    I = [sp.integrate(sp.log(x) ** k, (x, 1, sp.E)) for k in range(0, 13)]
    ok1 = sp.simplify(I[2] - (sp.E - 2)) == 0
    ok1 &= all(sp.simplify(I[k] - (sp.E - k * I[k - 1])) == 0 for k in range(1, 13))
    # (2) e/(k+2) < I_k < e/(k+1): 고정밀 수치 k=1..200
    ok2 = True
    for k in range(1, 201):
        Ik = mp.quad(lambda xx: mp.log(xx) ** k, [1, mp.e])
        ok2 &= (mp.e / (k + 2) < Ik < mp.e / (k + 1))
    n = 400
    V = mp.quad(lambda xx: mp.log(xx) ** (2 * n), [1, mp.e])
    ok3 = abs(n * V - mp.e / 2) < 0.01
    k = sp.symbols('k', positive=True)
    Lo = sp.limit(k * sp.E / (2 * k + 2), k, sp.oo)
    Hi = sp.limit(k * sp.E / (2 * k + 1), k, sp.oo)
    ok3 &= Lo == Hi == sp.E / 2
    report('J02-06', ok1 and ok2 and ok3, f"V1=e-2, I_k=e-kI_(k-1); 부등식 k<=200[증명-수치확인]; nV_n(400)={mp.nstr(n*V, 8)} -> e/2")


# ---------------------------------------------------------------- J02-07
def p07():
    n = sp.symbols('n', positive=True, integer=True)
    sq = sp.symbols('sq', positive=True)
    # 한 변 sq 인 정n각형 넓이 = n * (1/2) * sq * (sq/2)/tan(pi/n)
    area = n * sq ** 2 / (4 * sp.tan(sp.pi / n))
    chord = 2 * sp.sqrt(1 - x ** 2)
    Vn = sp.integrate(area.subs(sq, chord), (x, -1, 1))
    ok2 = sp.simplify(Vn - 4 * n / (3 * sp.tan(sp.pi / n))) == 0
    V3 = sp.simplify(Vn.subs(n, 3)); V4 = sp.simplify(Vn.subs(n, 4))
    # 독립 확인: 정삼각형 단면 sqrt3/4 c^2, 정사각형 c^2
    ok1 = sp.simplify(V3 - sp.integrate(sp.sqrt(3) / 4 * chord ** 2, (x, -1, 1))) == 0 and V4 == sp.integrate(chord ** 2, (x, -1, 1))
    ok1 &= sp.simplify(V3 - 4 * sp.sqrt(3) / 3) == 0 and V4 == sp.Rational(16, 3)
    # 정6각형 넓이 독립 확인 (3sqrt3/2 s^2)
    ok2 &= sp.simplify(area.subs({n: 6}) - 3 * sp.sqrt(3) / 2 * sq ** 2) == 0
    nn = sp.symbols('nn', positive=True)
    L = sp.limit(4 * nn / (3 * sp.tan(sp.pi / nn)) / nn ** 2, nn, sp.oo)
    ok3 = sp.simplify(L - 4 / (3 * sp.pi)) == 0
    report('J02-07', ok1 and ok2 and ok3, f"V3={V3}, V4={V4}; V_n=4n/(3tan(pi/n)); lim V_n/n^2={L}")


# ---------------------------------------------------------------- J02-08
def p08():
    hp = sp.symbols('hp', positive=True)
    L = 5
    xb = hp ** sp.Rational(1, 4)
    A = sp.integrate(hp - x ** 4, (x, -xb, xb))
    V = sp.simplify(L * A)
    ok1 = sp.simplify(V - 8 * hp ** sp.Rational(5, 4)) == 0
    w = sp.symbols('w', positive=True)
    Vw = sp.simplify(V.subs(hp, w ** 4 / 16))
    ok1 &= sp.simplify(Vw - w ** 5 / 4) == 0
    Q = sp.Rational(5, 4)
    # dV/dt = (dV/dw) w' = Q  =>  w^4 w' = Q / (5/4) = 1 (모든 w)
    inv = sp.simplify(Q / sp.diff(Vw, w) * w ** 4)
    ok2 = inv == 1
    # 시뮬레이션: V(t)=Qt 에서 w(t) 수치로 구해 w^4 w' 확인
    for tv in [0.3, 1.7, 4.4, 6.0]:
        wt = lambda T: (4 * Q * T) ** (1 / 5)
        d = (wt(tv + 1e-6) - wt(tv - 1e-6)) / 2e-6
        ok2 &= abs(wt(tv) ** 4 * d - 1) < 1e-5
    Tfull = sp.solve(sp.Eq(Q * t, V.subs(hp, 1)), t)[0]
    wfull = 2 * 1 ** sp.Rational(1, 4)
    dw = sp.Rational(1) / wfull ** 4
    ok3 = Tfull == sp.Rational(32, 5) and dw == sp.Rational(1, 16)
    report('J02-08', ok1 and ok2 and ok3, f"V=8h^(5/4)=w^5/4; w^4 w'=1 [증명-수치확인]; T={Tfull}, dw/dt={dw}")


# ---------------------------------------------------------------- J02-09
def p09():
    k = sp.symbols('k', positive=True)
    S = sp.Function('S')
    sol = sp.dsolve(sp.Eq(S(h).diff(h), k * S(h)), S(h), ics={S(0): 1}).rhs
    ksol = sp.solve(sp.Eq(sol.subs(h, 2), sp.E ** 2), k)
    ok1 = ksol == [1]
    Sh = sol.subs(k, 1)
    ok2 = sp.simplify(Sh - sp.exp(h)) == 0
    side = sp.sqrt(Sh)
    ok2 &= sp.simplify(side - sp.exp(h / 2)) == 0
    # 조건 확인: 매초 1 넣을 때 dS/dt = S'(h)/S(h) = 1 (상수)
    ok2 &= sp.simplify(sp.diff(Sh, h) / Sh) == 1
    V = sp.integrate(Sh, (h, 0, s))
    Tfull = sp.simplify(V.subs(s, 2))      # 매초 1
    T1 = sp.simplify(V.subs(s, 1))
    ok3 = sp.simplify(Tfull - (sp.E ** 2 - 1)) == 0 and sp.simplify(T1 - (sp.E - 1)) == 0
    report('J02-09', ok1 and ok2 and ok3, f"S'=kS, S(0)=1, S(2)=e^2 => k=1, S=e^h; 시간 {Tfull}, {T1}")


# ---------------------------------------------------------------- J02-10
def p10():
    al, be = sp.symbols('alpha beta', real=True)
    Vgen = sp.integrate(((x - al) * (x - be)) ** 2, (x, al, be))
    ok1 = sp.simplify(Vgen - (be - al) ** 5 / 30) == 0
    c = sp.symbols('c', real=True)
    csol = 2 - sp.sqrt(1 + m ** 2)
    # 접선 조건: 거리 = 1, c<2
    ok2 = sp.simplify(sp.Abs(csol - 2) / sp.sqrt(1 + m ** 2) - 1) == 0
    w2 = sp.expand(m ** 2 + 4 * csol)   # 근의 차의 제곱
    ok2 &= sp.simplify(w2 - ((sp.sqrt(1 + m ** 2) - 2) ** 2 + 3)) == 0
    # 무작위 m 에서 실제 교점으로 너비 확인
    for _ in range(20):
        mv = random.uniform(-5, 5)
        cv = 2 - mp.sqrt(1 + mv ** 2)
        r1 = (mv - mp.sqrt(mv ** 2 + 4 * cv)) / 2; r2 = (mv + mp.sqrt(mv ** 2 + 4 * cv)) / 2
        ok2 &= close((r2 - r1) ** 2, (mp.sqrt(1 + mv ** 2) - 2) ** 2 + 3)
        Vnum = mp.quad(lambda xx: (mv * xx + cv - xx ** 2) ** 2, [r1, r2])
        ok2 &= close(Vnum, (r2 - r1) ** 5 / 30)
    Vm = sp.sqrt(w2) ** 5 / 30
    grid = min(float(Vm.subs(m, -6 + i * 0.001)) for i in range(12001))
    vmin = sp.nsimplify(sp.simplify(Vm.subs(m, sp.sqrt(3))))
    ok3 = sp.simplify(vmin - 3 * sp.sqrt(3) / 10) == 0 and close(grid, float(vmin), 1e-6)
    crit = sp.solve(sp.diff(w2, m), m)
    ok3 &= set(crit) == {0, sp.sqrt(3), -sp.sqrt(3)}
    report('J02-10', ok1 and ok2 and ok3, f"V=w^5/30; w^2=(sqrt(1+m^2)-2)^2+3 [증명-수치확인]; min V={vmin} at m=±sqrt3")


for fn in [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10]:
    try:
        fn()
    except Exception as e:  # noqa
        report(fn.__name__.replace('p', 'J02-'), False, f"예외 {e!r}")

npass = sum(results)
print(f"SUMMARY T02: {npass}/{len(results)} PASS")
sys.exit(0 if npass == len(results) == 10 else 1)
