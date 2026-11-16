# -*- coding: utf-8 -*-
"""세종 J04 정적분의 점화식 — 검산 스크립트.
각 I_n 을 sympy integrate 로 직접 계산(작은 n)하거나 mpmath 고정밀 적분(큰 n)으로 계산하여
점화식·특정 항·극한/부등식을 md 의 정답과 비교한다."""
import sys, random
import sympy as sp
import mpmath as mp

mp.mp.dps = 40
x, n, k = sp.symbols('x n k', real=True)
E, PI = sp.E, sp.pi
results = []


def close(a, b, tol=1e-9):
    a, b = mp.mpf(a), mp.mpf(b)
    return abs(a - b) <= tol * max(1, abs(b))


def report(pid, ok, msg):
    ok = bool(ok)
    results.append(ok)
    print(f"{pid} {'PASS' if ok else 'FAIL'}  {msg}")


def eq(a, b):
    return sp.simplify(a - b) == 0


# ---------------- J04-01 : I_n = ∫_0^1 x^n e^x ----------------
def p01():
    I = [sp.integrate(x ** m * sp.exp(x), (x, 0, 1)) for m in range(13)]
    ok1 = all(eq(I[m], E - m * I[m - 1]) for m in range(1, 13)) and eq(I[3], 6 - 2 * E)
    # (2) e/(n+2) < I_n < e/(n+1) : n=0..200 고정밀 적분
    ok2 = True
    for m in range(0, 201):
        v = mp.quad(lambda t: t ** m * mp.e ** t, [0, 0.9, 0.99, 1])
        if not (mp.e / (m + 2) < v < mp.e / (m + 1)):
            ok2 = False
    # (3) n(e-(n+1)I_n) = n I_{n+1} (점화식) -> e : 상하한 극한 + 수치
    lo = sp.limit(n * E / (n + 3), n, sp.oo)
    hi = sp.limit(n * E / (n + 2), n, sp.oo)
    m = 3000
    Im = mp.quad(lambda t: t ** m * mp.e ** t, [0, 0.99, 0.999, 0.9999, 1])
    val = m * (mp.e - (m + 1) * Im)
    ok3 = lo == E and hi == E and abs(val - mp.e) < 3 * mp.e / m
    report('J04-01', ok1 and ok2 and ok3,
           f"점화식 n=1..12 OK, I_3={sp.simplify(I[3])}; [증명-수치확인] 부등식 n=0..200; n=3000 값 {mp.nstr(val, 10)} -> e")


# ---------------- J04-02 : I_n = ∫_0^1 (1+x^2)^{-n} ----------------
def p02():
    I = {m: sp.simplify(sp.integrate((1 + x ** 2) ** (-m), (x, 0, 1))) for m in range(1, 8)}
    ok1 = all(eq(2 * m * I[m + 1], (2 * m - 1) * I[m] + sp.Rational(1, 2 ** m)) for m in range(1, 7))
    ok2 = eq(I[1], PI / 4) and eq(I[2], PI / 8 + sp.Rational(1, 4)) and eq(I[3], 3 * PI / 32 + sp.Rational(1, 4))
    th = sp.symbols('theta')
    c6 = sp.integrate(sp.cos(th) ** 6, (th, 0, PI / 4))
    ok3 = eq(c6, I[4]) and eq(c6, 5 * PI / 64 + sp.Rational(11, 48))
    ok3 = ok3 and all(eq(sp.integrate(sp.cos(th) ** (2 * m - 2), (th, 0, PI / 4)), I[m]) for m in range(1, 7))
    wrong = I[3]
    report('J04-02', ok1 and ok2 and ok3,
           f"점화식 n=1..6 OK; I_2={I[2]}, I_3={I[3]}; [증명-수치확인] ∫cos^(2n-2)=I_n (n=1..6); ∫cos^6={sp.simplify(c6)}; 오답(첨자 밀림)={wrong}")


# ---------------- J04-03 : J_n = ∫_0^π e^x sin^n x ----------------
def p03():
    J = [sp.simplify(sp.integrate(sp.exp(x) * sp.sin(x) ** m, (x, 0, PI))) for m in range(9)]
    ok1 = all(eq((m * m + 1) * J[m], m * (m - 1) * J[m - 2]) for m in range(2, 9))
    ok2 = eq(J[0], sp.exp(PI) - 1) and eq(J[2], 2 * (sp.exp(PI) - 1) / 5) and eq(J[4], 24 * (sp.exp(PI) - 1) / 85)
    # 단조감소 수치 확인 n=0..150
    Jn = [mp.quad(lambda t: mp.e ** t * mp.sin(t) ** m, [0, mp.pi / 2, mp.pi]) for m in range(0, 152)]
    ok2b = all(Jn[m + 1] < Jn[m] for m in range(151))
    # (3) n(n-1)/(n^2+1) -> 1 이고 J_{n+1}/J_n 수치
    lb = sp.limit((n + 1) * n / ((n + 1) ** 2 + 1), n, sp.oo)
    r = Jn[151] / Jn[150]
    ok3 = lb == 1 and Jn[150 + 1] / Jn[150 - 1] < r < 1
    wrong = sp.Rational(3, 4) * sp.Rational(1, 2) * (sp.exp(PI) - 1)
    report('J04-03', ok1 and ok2 and ok2b and ok3,
           f"점화식 n=2..8 OK, J_4={J[4]}; [증명-수치확인] J_(n+1)<J_n n<=150; J151/J150={mp.nstr(r, 8)} -> 1; 오답(n^2 사용)={wrong}")


# ---------------- J04-04 : B(m,n) = ∫_0^1 x^m (1-x)^n ----------------
def p04():
    ok1 = ok2 = True
    for m in range(0, 7):
        for q in range(0, 7):
            B = sp.integrate(x ** m * (1 - x) ** q, (x, 0, 1))
            if B != sp.factorial(m) * sp.factorial(q) / sp.factorial(m + q + 1):
                ok2 = False
            if q >= 1:
                B2 = sp.integrate(x ** (m + 1) * (1 - x) ** (q - 1), (x, 0, 1))
                if B != sp.Rational(q, m + 1) * B2:
                    ok1 = False
    val = sp.integrate(sp.sin(x) ** 5 * sp.cos(x) ** 7, (x, 0, PI / 2))
    ok3 = val == sp.Rational(1, 120) and sp.Rational(1, 2) * sp.factorial(2) * sp.factorial(3) / sp.factorial(6) == val
    report('J04-04', ok1 and ok2 and ok3,
           f"[증명-수치확인] 점화식·닫힌식 0<=m,n<=6 전수 일치; ∫sin^5cos^7={val}; 오답(1/2 누락)={2*val}")


# ---------------- J04-05 : I_n = ∫_0^π x^n sin x ----------------
def p05():
    I = [sp.expand(sp.integrate(x ** m * sp.sin(x), (x, 0, PI))) for m in range(11)]
    ok1 = all(eq(I[m], PI ** m - m * (m - 1) * I[m - 2]) for m in range(2, 11))
    ok2 = eq(I[2], PI ** 2 - 4) and eq(I[3], PI ** 3 - 6 * PI) and eq(I[4], PI ** 4 - 12 * PI ** 2 + 48)
    direct = sp.expand(sp.integrate((x ** 2 - PI * x) ** 2 * sp.sin(x), (x, 0, PI)))
    ok3 = eq(direct, 48 - 4 * PI ** 2) and eq(I[4] - 2 * PI * I[3] + PI ** 2 * I[2], direct)
    ok3 = ok3 and sp.N(direct) > 0 and sp.pi < 2 * sp.sqrt(3)
    wrong = sp.expand(I[4] - PI * I[3] + PI ** 2 * I[2])
    report('J04-05', ok1 and ok2 and ok3,
           f"점화식 n=2..10 OK, I_4={I[4]}; ∫(x^2-πx)^2 sin x={direct}>0 -> π^2<12; 오답(교차항 2 누락)={wrong}")


# ---------------- J04-06 : I_n = ∫_1^e (ln x)^n / x^2 ----------------
def p06():
    I = [sp.simplify(sp.integrate(sp.log(x) ** m / x ** 2, (x, 1, E))) for m in range(9)]
    ok1 = eq(I[0], 1 - 1 / E) and all(eq(I[m], m * I[m - 1] - 1 / E) for m in range(1, 9))
    closed = lambda m: sp.factorial(m) * (1 - sum(sp.Rational(1, sp.factorial(j)) for j in range(m + 1)) / E)
    ok2 = all(eq(I[m], closed(m)) for m in range(9))
    # 부등식 1/(e(n+1)) < I_n < 1/(en), n=1..150
    for m in range(1, 151):
        v = mp.quad(lambda t: mp.log(t) ** m / t ** 2, [1, 2, 2.5, mp.e])
        if not (1 / (mp.e * (m + 1)) < v < 1 / (mp.e * m)):
            ok2 = False
    # (3) (n+1)!(e - S_n) -> 1
    vals = []
    for m in [20, 200, 2000]:
        S = mp.fsum(1 / mp.factorial(j) for j in range(m + 1))
        vals.append(mp.factorial(m + 1) * (mp.e - S))
    mp.mp.dps = 40
    ok3 = all(1 < v < 1 + mp.mpf(1) / m for v, m in zip(vals, [20, 200, 2000])) and abs(vals[-1] - 1) < 1e-3
    report('J04-06', ok1 and ok2 and ok3,
           f"점화식·닫힌식 n<=8 OK; [증명-수치확인] 부등식 n=1..150; (n+1)!(e-S_n): n=2000 -> {mp.nstr(vals[-1], 10)}")


mp.mp.dps = 40


# ---------------- J04-07 : I_n = ∫_0^π sin(nx)/sin x ----------------
def p07():
    # 차 I_{n+2}-I_n = ∫ 2cos((n+1)x) = 0 (기호)
    ok1 = all(sp.simplify(sp.expand_trig(sp.sin((m + 2) * x) - sp.sin(m * x)) - 2 * sp.cos((m + 1) * x) * sp.sin(x)) == 0
              for m in range(1, 8)) and all(sp.integrate(2 * sp.cos((m + 1) * x), (x, 0, PI)) == 0 for m in range(1, 30))
    def In(m):
        return mp.quad(lambda t: mp.sin(m * t) / mp.sin(t) if mp.sin(t) != 0 else m, mp.linspace(0, mp.pi, m + 2))
    ok2 = all(close(In(m), mp.pi if m % 2 else 0, 1e-12) for m in range(1, 31))
    def Fn(m):
        return mp.quad(lambda t: (mp.sin(m * t) / mp.sin(t)) ** 2 if mp.sin(t) != 0 else m * m, mp.linspace(0, mp.pi, 2 * m + 2))
    ok3 = all(close(Fn(m), m * mp.pi, 1e-10) for m in range(1, 21))
    # 항등식 sin^2(nx)-sin^2((n-1)x) = sin((2n-1)x) sin x : 무작위 20점
    random.seed(7)
    ok3 = ok3 and all(abs(mp.sin(m * v) ** 2 - mp.sin((m - 1) * v) ** 2 - mp.sin((2 * m - 1) * v) * mp.sin(v)) < 1e-30
                      for m, v in [(random.randint(1, 30), mp.mpf(random.uniform(0, 3))) for _ in range(20)])
    report('J04-07', ok1 and ok2 and ok3,
           "[증명-수치확인] I_(n+2)-I_n=0 기호확인; I_n=π(홀),0(짝) n=1..30; F_n=nπ n=1..20; 항등식 20점")


# ---------------- J04-08 : I_n = ∫_0^1 x^n / sqrt(1+x^2) ----------------
def p08():
    I = [sp.simplify(sp.integrate(x ** m / sp.sqrt(1 + x ** 2), (x, 0, 1))) for m in range(10)]
    ok1 = all(eq(m * I[m] + (m - 1) * I[m - 2], sp.sqrt(2)) for m in range(2, 10))
    ok2 = eq(I[1], sp.sqrt(2) - 1) and eq(I[3], (2 - sp.sqrt(2)) / 3) and eq(I[5], (7 * sp.sqrt(2) - 8) / 15)
    # (3) √2/(2n+3) < I_n < √2/(2n-1), n=1..200, 극한 √2/2
    ok3 = True
    for m in range(1, 201):
        v = mp.quad(lambda t: t ** m / mp.sqrt(1 + t * t), [0, 0.9, 0.99, 1])
        if not (mp.sqrt(2) / (2 * m + 3) < v < mp.sqrt(2) / (2 * m - 1)):
            ok3 = False
    lo = sp.limit(n * sp.sqrt(2) / (2 * n + 3), n, sp.oo)
    hi = sp.limit(n * sp.sqrt(2) / (2 * n - 1), n, sp.oo)
    ok3 = ok3 and eq(lo, sp.sqrt(2) / 2) and eq(hi, sp.sqrt(2) / 2)
    report('J04-08', ok1 and ok2 and ok3,
           f"점화식 n=2..9 OK; I_3={I[3]}, I_5={I[5]}; [증명-수치확인] 부등식 n=1..200; 극한 √2/2")


# ---------------- J04-09 : W_n = ∫_0^{π/2} cos^n x (불변량) ----------------
def p09():
    W = [sp.integrate(sp.cos(x) ** m, (x, 0, PI / 2)) for m in range(14)]
    ok1 = all(eq(m * W[m], (m - 1) * W[m - 2]) for m in range(2, 14))
    ok2 = all(eq(m * W[m] * W[m - 1], PI / 2) for m in range(1, 14))
    Wg = lambda m: mp.sqrt(mp.pi) * mp.gamma((m + 1) / mp.mpf(2)) / (2 * mp.gamma(m / mp.mpf(2) + 1))
    ok3 = all(close(Wg(m), sp.N(W[m], 40)) for m in range(14))
    v = mp.sqrt(10 ** 8) * Wg(10 ** 8)
    ok3 = ok3 and abs(v - mp.sqrt(mp.pi / 2)) < 1e-7
    report('J04-09', ok1 and ok2 and ok3,
           f"점화식 OK; [증명-수치확인] nW_nW_(n-1)=π/2 n=1..13; √n W_n (n=1e8)={mp.nstr(v, 12)} vs √(π/2)={mp.nstr(mp.sqrt(mp.pi/2), 12)}")


# ---------------- J04-10 : K_n = ∫_0^{π/2} cos^n x cos(nx) ----------------
def p10():
    K = [sp.simplify(sp.integrate(sp.cos(x) ** m * sp.cos(m * x), (x, 0, PI / 2))) for m in range(10)]
    ok1 = all(eq(2 * K[m], K[m - 1]) for m in range(1, 10))
    # 보조함수 미분 항등식 무작위 20점
    random.seed(11)
    ok1 = ok1 and all(sp.Abs(sp.N((sp.diff(sp.cos(x) ** m * sp.sin(m * x), x) - m * sp.cos(x) ** (m - 1) * sp.cos((m + 1) * x)).subs(x, v), 30)) < 1e-25
                      for m, v in [(random.randint(1, 9), random.uniform(0, 1.5)) for _ in range(20)])
    ok2 = all(eq(K[m], PI / 2 ** (m + 1)) for m in range(10))
    P = [None] + [sp.simplify(sp.integrate(sp.cos(x) ** (m - 1) * sp.sin(x) * sp.sin(m * x), (x, 0, PI / 2))) for m in range(1, 10)]
    ok2 = ok2 and all(eq(P[m], K[m]) for m in range(1, 10))
    S = sp.summation(k * PI / 2 ** (k + 1), (k, 1, sp.oo))
    ok3 = eq(S, PI)
    report('J04-10', ok1 and ok2 and ok3,
           f"2K_n=K_(n-1) n=1..9, K_n=π/2^(n+1), P_n=K_n (기호); [증명-수치확인] 보조함수 미분 20점; Σ kP_k={S}")


if __name__ == '__main__':
    for fn in [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10]:
        try:
            fn()
        except Exception as ex:
            results.append(False)
            print(f"{fn.__name__} FAIL  exception {ex!r}")
    npass = sum(results)
    print(f"SUMMARY T04: {npass}/{len(results)} PASS")
    if npass != len(results) or len(results) != 10:
        sys.exit(1)
