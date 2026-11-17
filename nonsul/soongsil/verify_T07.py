# -*- coding: utf-8 -*-
"""S07 귀납적으로 정의된 수열과 수학적 귀납법 — sympy 검산.
각 문항의 점화식을 그대로 반복 계산(Rational 정확 연산)하여 md의 정답과 비교한다."""
import sys
from sympy import Rational as R, symbols, simplify, sqrt, floor, Integer
import mpmath

RESULTS = []


def report(pid, ok, msg):
    RESULTS.append(ok)
    print(f"{pid} {'PASS' if ok else 'FAIL'}  {msg}")


# ---------------- S07-01 ----------------
def p01():
    # a1=1, a_{n+1} = a_n/(1+2n a_n)
    a = {1: R(1)}
    for k in range(1, 200):
        a[k + 1] = a[k] / (1 + 2 * k * a[k])
    a20 = a[20]
    nmin = min(k for k in a if a[k] < R(1, 1000))
    # 일반항 확인 1/a_n = n^2-n+1
    gen_ok = all(1 / a[k] == k * k - k + 1 for k in a)
    # 대표 오답: 1/a_{n+1}=1/a_n+2n 을 1/a_n = 1+n(n+1)로 착각 -> 부등식 n^2+n+1>1000 -> n=32
    wrong = min(k for k in range(1, 100) if k * k + k + 1 > 1000)
    ok = a20 == R(1, 381) and nmin == 33 and gen_ok and wrong == 32
    report("S07-01", ok, f"a20={a20}, min n={nmin}, 1/a_n=n^2-n+1 확인(n<=200), 오답 재현 {wrong}")


# ---------------- S07-02 ----------------
def p02():
    a = {1: 1}
    for k in range(1, 30):
        a[k + 1] = 3 * a[k] + 2 * k - 1
    s = sum(a[k] for k in range(1, 11))
    gen_ok = all(a[k] == 2 * 3 ** (k - 1) - k for k in a)
    # 오답: a_n+n 대신 a_n+1 을 등비로 착각한 계산은 성립하지 않음 -> 3^10-1 만 쓰고 -55 누락
    wrong = 3 ** 10 - 1
    ok = s == 58993 and gen_ok
    report("S07-02", ok, f"sum a1..a10={s}, a_n=2*3^(n-1)-n 확인, 오답(Σn 누락)={wrong}")


# ---------------- S07-03 ----------------
def p03():
    # a1=60, a_{n+1}=|a_n-n|
    a = {1: 60}
    for k in range(1, 200):
        a[k + 1] = abs(a[k] - k)
    s = sum(a[k] for k in range(1, 41))
    ph1 = all(a[k] == 60 - k * (k - 1) // 2 for k in range(1, 12))
    inv = all(0 <= a[k] <= k for k in range(11, 200))  # [증명-수치확인] n>=11 에서 0<=a_n<=n
    ph2 = all(a[11 + 2 * j] == 5 + j and a[12 + 2 * j] == 6 + j for j in range(0, 90))
    # 오답: n>=11 에서 a_{n+1}=a_n+1 로 착각 -> a_n = n-6
    wrong = sum(a[k] for k in range(1, 12)) + sum(k - 6 for k in range(12, 41))
    ok = s == 810 and ph1 and inv and ph2 and wrong == 1020
    report("S07-03", ok, f"sum a1..a40={s}, 1~11항 60-n(n-1)/2, n>=11 0<=a_n<=n 및 a_(n+2)=a_n+1 확인(n<200), 오답={wrong}")


# ---------------- S07-04 ----------------
def p04():
    a = {1: R(1)}
    for k in range(1, 40):
        a[k + 1] = (a[k] - 4) / (a[k] - 3)
    s = sum((2 - a[k]) * (2 - a[k + 1]) for k in range(1, 31))
    gen_ok = all(a[k] == R(2 * k - 1, k) for k in a)
    ok = s == R(30, 31) and gen_ok
    report("S07-04", ok, f"sum={s}, a_n=(2n-1)/n 확인(n<=40)")


# ---------------- S07-05 ----------------
def p05():
    a = {1: R(1)}
    for k in range(1, 60):
        a[k + 1] = R(k + 2, k) * a[k] + 1
    s = sum(a[k] for k in range(1, 21))
    gen_ok = all(a[k] == k * k for k in a)  # [증명-수치확인] n<=60
    ok = s == 2870 and gen_ok
    report("S07-05", ok, f"[증명-수치확인] a_n=n^2 (n<=60), sum a1..a20={s}")


# ---------------- S07-06 ----------------
def p06():
    a = {1: 3}
    for k in range(1, 9):
        a[k + 1] = a[k] ** 2 - 2 * a[k] + 2
    prod_ok = True
    p = 1
    for k in range(1, 9):
        p *= a[k]
        prod_ok &= (p == a[k + 1] - 2)  # [증명-수치확인]
    p5 = a[1] * a[2] * a[3] * a[4] * a[5]
    ok = prod_ok and p5 == 2 ** 32 - 1
    report("S07-06", ok, f"[증명-수치확인] Πa_k=a_(n+1)-2 (n<=8), Π_(k<=5)={p5}=2^32-1")


# ---------------- S07-07 ----------------
def p07():
    a = {1: 0}
    for k in range(1, 60):
        a[k + 1] = a[k] + 3 * k - 29
    S = {}
    t = 0
    for k in range(1, 61):
        t += a[k]
        S[k] = t
    mval = min(S.values())
    arg = [k for k in S if S[k] == mval]
    amin = min(a.values())
    # 오답: a_n 의 최솟값(n=10)에서 멈춘 S_10
    wrong = S[10]
    ok = mval == -1539 and arg == [19] and amin == -126
    report("S07-07", ok, f"min S_n={mval} at n={arg}, min a_n={amin}, 오답 S_10={wrong}")


# ---------------- S07-08 ----------------
def p08():
    # p 를 격자로 훑어 '모든 항 양수' 여부 판정 (N=200항)
    def allpos(p):
        x, y = R(1), R(p)
        if x <= 0 or y <= 0:
            return False
        for _ in range(200):
            x, y = y, 5 * y - 6 * x
            if y <= 0:
                return False
        return True
    grid = [R(k, 100) for k in range(-300, 700)]
    good = [p for p in grid if allpos(p)]
    ok = min(good) == 2 and all(allpos(p) == (p >= 2) for p in grid)
    # 경계 근방
    ok &= allpos(R(2)) and not allpos(2 - R(1, 10 ** 6))
    report("S07-08", ok, f"p 격자(-3~7, 0.01 간격)에서 모든 항 양수 ⇔ p>=2, 경계 p=2-1e-6 실패 확인")


# ---------------- S07-09 ----------------
def p09():
    mpmath.mp.dps = 50
    x = mpmath.mpf(2)
    seq = [None, x]
    for k in range(1, 60):
        x = x + 2 / x
        seq.append(x)
    a50 = seq[50]
    # [증명-수치확인] a_n^2 >= 4n, 4/a_k^2 <= 1/k
    lb_ok = all(seq[k] ** 2 >= 4 * k for k in range(1, 60))
    H49 = sum(R(1, k) for k in range(1, 50))
    ok = int(mpmath.floor(a50)) == 14 and lb_ok and H49 < 25 and 200 < a50 ** 2 < 225
    report("S07-09", ok, f"[증명-수치확인] a50={mpmath.nstr(a50, 12)}, a_n^2>=4n, H49={float(H49):.4f}<25 -> [a50]=14")


# ---------------- S07-10 ----------------
def p10():
    a = {1: R(1), 2: R(1)}
    for k in range(1, 40):
        a[k + 2] = (a[k + 1] ** 2 + 2) / a[k]
    int_ok = all(a[k].q == 1 for k in a)  # [증명-수치확인] 모든 항 정수(n<=41)
    lin_ok = all(a[k + 2] == 4 * a[k + 1] - a[k] for k in range(1, 40))
    ok = a[8] == 2131 and int_ok and lin_ok
    report("S07-10", ok, f"[증명-수치확인] a_(n+2)=4a_(n+1)-a_n, 정수성(n<=41), a8={a[8]}")


if __name__ == "__main__":
    for f in (p01, p02, p03, p04, p05, p06, p07, p08, p09, p10):
        try:
            f()
        except Exception as e:  # noqa
            report(f.__name__, False, f"예외 {e!r}")
    npass = sum(RESULTS)
    print(f"SUMMARY T07: {npass}/{len(RESULTS)} PASS")
    if npass != len(RESULTS):
        sys.exit(1)
