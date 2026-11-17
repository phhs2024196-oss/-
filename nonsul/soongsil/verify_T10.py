#!/usr/bin/env python3
"""verify_T10.py — S10 개수 세기(격자점·정수 조건·가우스 기호) 검산: 반복문/itertools 전수 열거"""
import sys
import itertools
from fractions import Fraction

RES = []


def report(pid, ok, msg):
    RES.append(ok)
    print(f"{pid} {'PASS' if ok else 'FAIL'}  {msg}")


def ilog(base, n):
    """정수 n>=1 에 대한 [log_base n] (정수 연산)"""
    k = 0
    while base ** (k + 1) <= n:
        k += 1
    return k


# S10-01: 자연수 (x,y), log2 x + log4 y <= 5  <=>  x^2 y <= 2^10. 개수 1641
def p01():
    # 부등식을 실수 로그로 직접 판정하지 않고, 동치인 2*log2 x + log2 y <= 10 을
    # 4^(log2 x + log4 y) = x^2 * y 로 정수 비교 (x,y 범위는 넉넉히 전수)
    cnt = sum(1 for x, y in itertools.product(range(1, 40), range(1, 1100)) if x * x * y <= 1024)
    wrong = sum(1 for x, y in itertools.product(range(1, 40), range(1, 1100)) if x * y * y <= 1024)  # 밑 혼동 (log4 x + log2 y)
    report("S10-01", cnt == 1641, f"전수 개수={cnt} (y 범위 상한 여유 확인, 오답 밑 혼동={wrong})")


# S10-02: sum_{k=1}^{100} [(sqrt(8k+1)-1)/2] = 858
def p02():
    from math import isqrt
    # 직접합: 정수 연산으로 [(sqrt(8k+1)-1)/2] (isqrt 후 보정)
    def g(k):
        r = isqrt(8 * k + 1)
        return (r - 1) // 2
    s = sum(g(k) for k in range(1, 101))
    # 실수 판정과 일치 확인 (가장 큰 m: (2m+1)^2 <= 8k+1)
    chk = all(g(k) == max(m for m in range(0, 50) if (2 * m + 1) ** 2 <= 8 * k + 1) for k in range(1, 101))
    # 다른 방법: (k, m) 격자점, m>=1, m(m+1)/2 <= k <= 100
    t = sum(1 for k in range(1, 101) for m in range(1, 20) if m * (m + 1) // 2 <= k)
    wrong = sum(m * (m + 1) for m in range(1, 13)) + 13 * 14  # 마지막 구간 91~104 전체로 센 오답
    report("S10-02", chk and s == t == 858, f"직접합={s}, 격자점세기={t} (오답 마지막 구간 전체={wrong})")


# S10-03: 1<=n<=1000, [cuberoot n] | n 인 n 의 개수 = 172
def p03():
    def icbrt(n):
        m = 0
        while (m + 1) ** 3 <= n:
            m += 1
        return m
    cnt = sum(1 for n in range(1, 1001) if n % icbrt(n) == 0)
    wrong = sum(3 * m + 4 for m in range(1, 11))  # m=10 블록도 3m+4개라 본 오답
    report("S10-03", cnt == 172, f"전수={cnt} (오답 m=10 블록 전체 포함={wrong})")


# S10-04: [log2 n] = 3[log10 n] 인 자연수 n 의 개수 = 59 (n<=10^7 까지 전수 + 이후 불가능 확인)
def p04():
    sols = [n for n in range(1, 10 ** 6) if ilog(2, n) == 3 * ilog(10, n)]
    # k>=4 에서는 2*8^k <= 10^k 이므로 해가 없음을 k=4..60 확인
    no_more = all(2 * 8 ** k <= 10 ** k for k in range(4, 61))
    ok = len(sols) == 59 and no_more and max(sols) == 1023
    wrong = len([n for n in sols if n < 1000])  # k=3 구간 누락
    report("S10-04", ok and wrong == 35, f"n<10^6 전수 해 {len(sols)}개(최대 {max(sols)}), k>=4 불가능 확인={no_more} (오답 k=3 누락={wrong})")


# S10-05: 2<=x,y<=100 정수, [log_x y] = 2 인 (x,y) 개수 = 321
def p05():
    cnt = sum(1 for x in range(2, 101) for y in range(2, 101) if ilog(x, y) == 2)
    wrong = sum(1 for x in range(2, 101) for y in range(2, 101) if x * x <= y <= x ** 3)  # 등호 포함 오답
    report("S10-05", cnt == 321, f"전수={cnt} (오답 y<=x^3 포함={wrong})")


# S10-06: 각 자리 1,2,3, 3이 연속하지 않는 여섯 자리 자연수 개수 = 448
def p06():
    cnt = sum(1 for d in itertools.product('123', repeat=6) if '33' not in ''.join(d))
    a = [None, 3, 8]
    for n in range(3, 7):
        a.append(2 * a[n - 1] + 2 * a[n - 2])
    b = [None, 3, 8]
    for n in range(3, 7):
        b.append(2 * b[n - 1] + b[n - 2])  # 오답 점화
    report("S10-06", cnt == 448 == a[6], f"전수={cnt}, 점화 a_n={a[1:]} (오답 a_n=2a_(n-1)+a_(n-2): {b[6]})")


# S10-07: 1<=a<=20, 1<=b<=25, 7 | 3^a + 5^b 인 (a,b) 개수 = 84
def p07():
    cnt = sum(1 for a, b in itertools.product(range(1, 21), range(1, 26)) if (3 ** a + 5 ** b) % 7 == 0)
    via = sum(1 for a, b in itertools.product(range(1, 21), range(1, 26)) if (a + b) % 6 == 3)
    wrong = sum(1 for a, b in itertools.product(range(1, 19), range(1, 25)) if (3 ** a + 5 ** b) % 7 == 0)
    report("S10-07", cnt == 84 == via, f"전수={cnt}, a+b≡3(mod 6) 개수={via} (오답 a<=18,b<=24 만={wrong})")


# S10-08: x/8 < y < sqrt(x) 인 격자점 개수 = 77
def p08():
    cnt = 0
    for x in range(1, 100):
        for y in range(-5, 20):
            # 정수 비교: x < 8y 와 y^2 < x (y>0 필요: y<sqrt x 이고 y>x/8>0)
            if x < 8 * y and y > 0 and y * y < x:
                cnt += 1
    N = lambda x: sum(1 for y in range(-5, 30) if x < 8 * y and y > 0 and y * y < x)
    sN = sum(N(x) for x in range(1, 71))
    wrong = sum(1 for x in range(1, 100) for y in range(1, 20) if x <= 8 * y and y * y <= x)  # 경계 포함
    report("S10-08", cnt == 77 == sN, f"격자점 전수={cnt}, sum_(x=1..70) N(x)={sN} (오답 경계 포함={wrong})")


# S10-09: sum_{k=1}^{100} [37k/101] = 1800 / 삼각형 내부 격자점
def p09():
    s = sum((37 * k) // 101 for k in range(1, 101))
    # 삼각형 (0,0),(101,0),(101,37) 내부 격자점 전수
    inner = sum(1 for x in range(1, 101) for y in range(1, 37) if 101 * y < 37 * x)
    wrong = sum(Fraction(37 * k, 101) for k in range(1, 101))  # 가우스 무시
    report("S10-09", s == inner == 1800, f"합={s}, 내부격자점={inner} (오답 가우스 무시={wrong})")


# S10-10: 1<=n<=1023, sum_{j=1}^{9} [n/2^j] = n-3 인 n 의 개수 = 120
def p10():
    cnt = sum(1 for n in range(1, 1024) if sum(n // 2 ** j for j in range(1, 10)) == n - 3)
    # [증명-수치확인] 항등식 sum [n/2^j] = n - s_2(n), n=1..1023
    ident = all(sum(n // 2 ** j for j in range(1, 11)) == n - bin(n).count('1') for n in range(1, 1024))
    from math import comb
    report("S10-10", cnt == 120 == comb(10, 3) and ident and comb(9, 3) == 84, f"[증명-수치확인] 전수={cnt}, 항등식 n=1..1023 성립={ident} (오답 9자리로 셈 C(9,3)={comb(9,3)})")


for fn in (p01, p02, p03, p04, p05, p06, p07, p08, p09, p10):
    try:
        fn()
    except Exception as e:  # noqa
        report("S10-" + fn.__name__[1:], False, f"예외: {e!r}")

n = sum(RES)
print(f"SUMMARY T10: {n}/10 PASS")
sys.exit(0 if n == 10 else 1)
