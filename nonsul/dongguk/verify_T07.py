"""verify_T07.py — 동국 D07 (점화식과 확률·경우의 수의 결합) 검산.
모든 정답은 전수 열거 / 정확한 분수 시뮬레이션으로 독립 계산한 뒤 md의 닫힌 식과 비교한다."""
import sys
from fractions import Fraction as F
from itertools import product, permutations
from math import factorial, floor
import sympy as sp

results = []


def seq_limit(a, L, ns=(100, 200, 400)):
    """수열 a(n) 의 정확한 분수값이 L 로 기하급수적으로 접근함을 확인(|a(n)-L| < 1e-25). 통과하면 L 반환."""
    return L if all(abs(a(n) - L) < F(1, 10 ** 25) for n in ns) else None


def report(tag, ok, msg):
    results.append(ok)
    print(f"{tag} {'PASS' if ok else 'FAIL'}  {msg}")


# ---------------------------------------------------------------- D07-01
def p01():
    # 전달 n-1 번, 각 전달에서 1/4 확률로 뒤집힘. 전수 열거.
    def prob_correct(n):
        tot = F(0)
        for flips in product([0, 1], repeat=n - 1):
            w = F(1)
            for f in flips:
                w *= F(1, 4) if f else F(3, 4)
            if sum(flips) % 2 == 0:
                tot += w
        return tot
    ok = True
    for n in range(1, 13):
        ok &= prob_correct(n) == F(1, 2) + F(1, 2) ** n
    # (1) p2, p3, 조건부확률 P(2번 정확 | 3번 정확)
    p2, p3 = prob_correct(2), prob_correct(3)
    both = F(3, 4) * F(3, 4)
    cond = both / p3
    ok &= (p2, p3, cond) == (F(3, 4), F(5, 8), F(9, 10))
    # (3) p_n - 1/2 < 1/100 인 최소 n
    nmin = next(n for n in range(1, 50) if prob_correct(n) - F(1, 2) < F(1, 100))
    ok &= nmin == 7
    # 대표 오답: 2^n>100 을 n-1 로 착각 → 6 (재현: p_{n}을 (1/2)^{n-1} 로 쓴 경우)
    wrong = next(n for n in range(1, 50) if F(1, 2) ** (n - 1) < F(1, 100))  # 지수를 n-1 로 밀린 일반항
    report("D07-01", ok, f"p2=3/4,p3=5/8,cond=9/10, p_n=1/2+(1/2)^n (n<=12 전수), 최소 n={nmin} (오답 재현 {wrong})")


# ---------------------------------------------------------------- D07-02
def p02():
    def c(n):
        return sum(1 for s in product(range(3), repeat=n)
                   if all(s[i] != s[i + 1] for i in range(n - 1)) and s[0] != s[-1])
    ok = (c(3), c(4)) == (6, 18)
    for n in range(3, 11):
        ok &= c(n) + c(n - 1) == 3 * 2 ** (n - 1)
    for n in range(2, 11):
        ok &= c(n) == 2 ** n + 2 * (-1) ** n
    ok &= c(10) == 1026
    lim = seq_limit(lambda n: F(2 ** n + 2 * (-1) ** n, 3 * 2 ** (n - 1)), F(2, 3))
    ok &= lim == F(2, 3)
    report("D07-02", ok, f"c3=6,c4=18, c_n+c_(n-1)=3*2^(n-1), c_n=2^n+2(-1)^n (n<=10 전수), c10=1026, 극한 {lim}")


# ---------------------------------------------------------------- D07-03
def p03():
    # 공 4개 라벨: A 상자 W1 W2, B 상자 K1 K2. 매 시행 각 상자에서 1개씩 골라 교환 (2*2 균등).
    def dist(n):
        states = {(('W', 'W'), ('K', 'K')): F(1)}
        for _ in range(n):
            new = {}
            for (A, B), w in states.items():
                for i in range(2):
                    for j in range(2):
                        A2, B2 = list(A), list(B)
                        A2[i], B2[j] = B[j], A[i]
                        key = (tuple(sorted(A2)), tuple(sorted(B2)))
                        new[key] = new.get(key, 0) + w * F(1, 4)
            states = new
        d = {0: F(0), 1: F(0), 2: F(0)}
        for (A, B), w in states.items():
            d[A.count('W')] += w
        return d
    ok = True
    d1, d2 = dist(1), dist(2)
    ok &= d1[1] == 1 and d2[1] == F(1, 2)
    for n in range(0, 11):
        d = dist(n)
        bn = F(2, 3) - F(2, 3) * F(-1, 2) ** n
        ok &= d[1] == bn
        if n >= 1:
            an = F(1, 6) - F(1, 6) * F(-1, 2) ** (n - 1)
            ok &= d[2] == an and d[0] == 1 - an - bn
        if n >= 1:
            ok &= dist(n)[1] == 1 - dist(n - 1)[1] / 2
    lim = seq_limit(lambda n: dist(n)[2] if n <= 10 else F(1, 6) - F(1, 6) * F(-1, 2) ** (n - 1), F(1, 6))
    ok &= lim == F(1, 6)
    report("D07-03", ok, f"b1=1,b2=1/2, b_n=2/3-2/3(-1/2)^n, a_n=1/6-1/6(-1/2)^(n-1) (n<=10 공 라벨 전수), lim a_n={lim}")


# ---------------------------------------------------------------- D07-04
def p04():
    # 폴리아 항아리: 모든 꺼냄 순서를 정확한 확률로 열거
    def dist(n):
        states = {(1, 1): F(1)}
        for _ in range(n):
            new = {}
            for (r, b), w in states.items():
                t = r + b
                new[(r + 1, b)] = new.get((r + 1, b), 0) + w * F(r, t)
                new[(r, b + 1)] = new.get((r, b + 1), 0) + w * F(b, t)
            states = new
        return {r: w for (r, b), w in states.items()}
    ok = True
    for n in range(1, 41):
        d = dist(n)
        ok &= all(d.get(k, 0) == F(1, n + 1) for k in range(1, n + 2))
    # (3)(i) (n+1)번째에 빨간 공일 확률
    for n in range(1, 30):
        d = dist(n)
        ok &= sum(w * F(r, n + 2) for r, w in d.items()) == F(1, 2)
    # (3)(ii) P(R_n <= (n+2)/3) = floor((n+2)/3)/(n+1) → 1/3
    for n in range(1, 60):
        d = dist(n)
        pr = sum(w for r, w in d.items() if 3 * r <= n + 2)
        ok &= pr == F((n + 2) // 3, n + 1)
    d = dist(300)
    pr = sum(w for r, w in d.items() if 3 * r <= 302)
    ok &= abs(float(pr) - 1 / 3) < 0.003
    nn = sp.symbols('n', positive=True)
    ok &= sp.limit((nn / 3 - 1) / (nn + 1), nn, sp.oo) == sp.Rational(1, 3) == sp.limit((nn + 2) / 3 / (nn + 1), nn, sp.oo)
    report("D07-04", ok, "P(R_n=k)=1/(n+1) (n<=40 전수) [증명-수치확인], 다음 빨강 확률 1/2, 극한 1/3 (n<=59 정확, n=300 근사)")


# ---------------------------------------------------------------- D07-05
def p05():
    def land(t, N):
        # 위치별 도달확률 DP (위치 0 출발, +1 확률 t, +2 확률 1-t) = 경로 전수와 동일
        reach = [F(0)] * (N + 3)
        reach[0] = F(1)
        for pos in range(N + 1):
            reach[pos + 1] += reach[pos] * t
            reach[pos + 2] += reach[pos] * (1 - t)
        return reach[:N + 1]
    # 경로 전수로 DP 검증 (n<=10)
    def brute(t, n):
        tot = F(0)
        for L in range(1, n + 1):
            for steps in product([1, 2], repeat=L):
                if sum(steps) == n:
                    w = F(1)
                    for s in steps:
                        w *= t if s == 1 else 1 - t
                    tot += w
        return tot
    ok = True
    h = F(1, 2)
    r = land(h, 15)
    ok &= r[1:4] == [F(1, 2), F(3, 4), F(5, 8)]
    for n in range(1, 11):
        ok &= brute(h, n) == r[n]
    for n in range(0, 16):
        ok &= r[n] == F(2, 3) + F(1, 3) * F(-1, 2) ** n
    t = sp.symbols('t', positive=True)
    L = sp.solve(sp.Eq(1 / (2 - t), sp.Rational(3, 5)), t)
    ok &= L == [sp.Rational(1, 3)]
    r3 = land(F(1, 3), 15)
    for n in range(0, 16):
        ok &= r3[n] == F(3, 5) + F(2, 5) * F(-2, 3) ** n
        if n >= 1:
            ok &= r3[n] == 1 - F(2, 3) * r3[n - 1]
    report("D07-05", ok, "p1..p3=1/2,3/4,5/8, p_n=2/3+1/3(-1/2)^n (경로 전수), t=1/3, p_n=3/5+2/5(-2/3)^n")


# ---------------------------------------------------------------- D07-06
def p06():
    def xy(n):
        x = y = 0
        for w in product('abc', repeat=n):
            A, B = w.count('a') % 2, w.count('b') % 2
            x += (A == 0 and B == 0)
            y += (A == 1 and B == 1)
        return x, y
    ok = [xy(n)[0] for n in (1, 2, 3)] == [1, 3, 7]
    for n in range(1, 11):
        x, y = xy(n)
        ok &= x - y == 1
        ok &= x == (3 ** n + 2 + (-1) ** n) // 4 and (3 ** n + 2 + (-1) ** n) % 4 == 0
        if n >= 2:
            ok &= x == 3 ** (n - 1) + 1 - xy(n - 1)[0]
    ok &= xy(8)[0] == 1641
    lim = seq_limit(lambda n: F(3 ** n + 2 + (-1) ** n, 4 * 3 ** n), F(1, 4))
    ok &= lim == F(1, 4)
    report("D07-06", ok, f"x1..x3=1,3,7, x_n-y_n=1 [증명-수치확인 n<=10], x_n=(3^n+2+(-1)^n)/4, x8=1641, 극한 {lim}")


# ---------------------------------------------------------------- D07-07
def p07():
    def D(n):
        return sum(1 for p in permutations(range(n)) if all(p[i] != i for i in range(n)))
    Ds = {n: D(n) for n in range(1, 9)}
    ok = (Ds[2], Ds[3], Ds[4]) == (1, 2, 9)
    for n in range(3, 9):
        ok &= Ds[n] == (n - 1) * (Ds[n - 1] + Ds[n - 2])
    for n in range(2, 9):
        ok &= Ds[n] - n * Ds[n - 1] == (-1) ** n
    p6 = F(Ds[6], factorial(6))
    ok &= p6 == F(53, 144) and Ds[7] == 1854
    one7 = F(sum(1 for p in permutations(range(7)) if sum(p[i] == i for i in range(7)) == 1), factorial(7))
    ok &= one7 == F(53, 144)
    report("D07-07", ok, f"D2..D4=1,2,9, D_n=(n-1)(D_(n-1)+D_(n-2)), D_n-nD_(n-1)=(-1)^n (n<=8 순열 전수), p6=53/144, D7=1854, 정확히1개={one7}")


# ---------------------------------------------------------------- D07-08
def p08():
    # 꼭짓점 0=A,1=B,2=C,3=D. 정지 1/2, 시계 1/4, 반시계 1/4 — 이동 순서 전수
    def dist(n):
        d = [F(0)] * 4
        for mv in product([0, 1, -1], repeat=n):
            w = F(1)
            for m in mv:
                w *= F(1, 2) if m == 0 else F(1, 4)
            d[sum(mv) % 4] += w
        return d
    ok = True
    d1, d2 = dist(1), dist(2)
    ok &= (d1[0], d1[1] + d1[3], d1[2]) == (F(1, 2), F(1, 2), 0)
    ok &= (d2[0], d2[2]) == (F(3, 8), F(1, 8))
    for n in range(1, 10):
        d = dist(n)
        ok &= d[1] + d[3] == F(1, 2)
        ok &= d[0] - d[2] == F(1, 2) ** n
        ok &= d[0] == F(1, 4) + F(1, 2) ** (n + 1) and d[2] == F(1, 4) - F(1, 2) ** (n + 1)
    # c_n >= 6/25 인 최소 n (닫힌 식 검증 후 사용)
    nmin = next(n for n in range(1, 60) if F(1, 4) - F(1, 2) ** (n + 1) >= F(6, 25))
    ok &= nmin == 6 and dist(6)[2] >= F(6, 25) and dist(5)[2] < F(6, 25)
    report("D07-08", ok, f"b_n=1/2, a_n-c_n=(1/2)^n, a_n=1/4+(1/2)^(n+1) (n<=9 전수), 최소 n={nmin}")


# ---------------------------------------------------------------- D07-09
def p09():
    # 공정한 무작위 걸음, 흡수 상태 0, N. 연립방정식을 정확히 풀어 도달확률 계산
    def hit(N):
        xs = sp.symbols(f'x0:{N + 1}')
        eqs = [sp.Eq(xs[0], 0), sp.Eq(xs[N], 1)]
        eqs += [sp.Eq(xs[k], (xs[k - 1] + xs[k + 1]) / 2) for k in range(1, N)]
        s = sp.solve(eqs, xs)
        return [s[x] for x in xs]
    ok = True
    p = {}
    for n in range(2, 13):
        p[n] = hit(n)[n - 1]  # n-1 에서 출발해 0보다 n 에 먼저 도달
        ok &= p[n] == sp.Rational(n - 1, n)
    ok &= p[2] == sp.Rational(1, 2)
    for n in range(3, 13):
        ok &= sp.simplify(p[n] - (sp.Rational(1, 2) + p[n - 1] * p[n] / 2)) == 0
        ok &= sp.simplify(1 / (1 - p[n]) - 1 / (1 - p[n - 1]) - 1) == 0
    for N in range(2, 13):
        prod = sp.prod([p[n] for n in range(2, N + 1)])
        ok &= prod == hit(N)[1] == sp.Rational(1, N)
    # 몬테카를로 대신 경로 길이 제한 전수(근사)로 N=4, 출발 1 확인
    tot = F(0)
    def walk(pos, w, depth):
        nonlocal tot
        if pos == 4:
            tot += w; return
        if pos == 0 or depth == 0:
            return
        walk(pos + 1, w / 2, depth - 1); walk(pos - 1, w / 2, depth - 1)
    walk(1, F(1), 40)
    ok &= abs(float(tot) - 0.25) < 1e-3
    report("D07-09", ok, "p_n=(n-1)/n, p_n=1/2+p_(n-1)p_n/2, 1/(1-p_n) 등차 (n<=12 연립 정확해), 곱=1/N, N=10 → 1/10")


# ---------------------------------------------------------------- D07-10
def p10():
    # 최적 정지: 모든 결정적 문턱 전략을 전수 탐색해 최대 기댓값
    from functools import lru_cache
    def best(n):
        # 남은 횟수 n, 전략 = 각 단계 '이 눈 이상이면 멈춤' 문턱 → 전수
        bestv = None
        for th in product(range(1, 8), repeat=n - 1):
            def val(i):
                if i == n - 1:
                    return F(7, 2)
                v_next = val(i + 1)
                return sum((F(x) if x >= th[i] else v_next) for x in range(1, 7)) / 6
            v = val(0)
            bestv = v if bestv is None or v > bestv else bestv
        return bestv
    e = {1: F(7, 2)}
    for n in range(1, 30):
        e[n + 1] = sum(max(F(x), e[n]) for x in range(1, 7)) / 6
    ok = all(best(n) == e[n] for n in range(1, 6))
    ok &= [e[k] for k in range(1, 6)] == [F(7, 2), F(17, 4), F(14, 3), F(89, 18), F(277, 54)]
    for n in range(5, 30):
        ok &= e[n] == 6 - F(47, 54) * F(5, 6) ** (n - 5)
        ok &= 5 <= e[n] < 6 and e[n + 1] == 1 + F(5, 6) * e[n]
    nmin = next(n for n in range(1, 30) if e[n] >= F(11, 2))
    ok &= nmin == 9
    ok &= 125 * 47 > 27 * 216
    report("D07-10", ok, f"e1..e5=7/2,17/4,14/3,89/18,277/54 (문턱 전략 전수), e_n=6-47/54(5/6)^(n-5), 최소 n={nmin}")


for f in (p01, p02, p03, p04, p05, p06, p07, p08, p09, p10):
    try:
        f()
    except Exception as ex:  # noqa
        report(f.__name__, False, f"예외 {ex!r}")

npass = sum(results)
print(f"SUMMARY T07: {npass}/10 PASS")
if npass != 10:
    sys.exit(1)
