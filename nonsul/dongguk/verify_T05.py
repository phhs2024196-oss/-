"""D05. 경우의 수와 수학적 확률 — 검산 스크립트 (전수 열거 + Fraction)"""
import sys
from fractions import Fraction as F
from itertools import product, permutations, combinations
from math import comb, factorial

results = []


def report(pid, ok, msg):
    results.append(ok)
    print(f"{pid} {'PASS' if ok else 'FAIL'}  {msg}")


def nondec(n, vals):
    return [t for t in product(vals, repeat=n) if all(t[i] <= t[i + 1] for i in range(n - 1))]


def p01():
    fs = nondec(5, range(1, 7))
    a1 = len(fs)
    a2 = sum(1 for f in fs if f[2] == 3)
    a3 = F(sum(1 for f in fs if f[2] in (3, 4)), a1)
    wrong = comb(6, 5)  # 증가함수(<)로 착각
    ok = (a1, a2, a3) == (252, 60, F(10, 21)) and wrong == 6
    report("D05-01", ok, f"(1){a1} (2){a2} (3){a3}; 오답(순증가로 착각) {wrong}")


def paths(a, b):
    """(0,0)->(a,b) 최단경로를 문자열로 전수 생성"""
    res = []
    for pos in combinations(range(a + b), b):
        s = ['R'] * (a + b)
        for p in pos:
            s[p] = 'U'
        res.append(s)
    return res


def visited(s):
    x = y = 0
    pts = {(0, 0)}
    for c in s:
        if c == 'R':
            x += 1
        else:
            y += 1
        pts.add((x, y))
    return pts


def p02():
    ps = [visited(s) for s in paths(5, 4)]
    a1 = len(ps)
    P, Q = (2, 1), (3, 3)
    a2 = sum(1 for v in ps if P in v or Q in v)
    a3 = F(sum(1 for v in ps if P not in v and Q not in v), a1)
    wrong = 60 + 60  # 교집합 미차감
    ok = (a1, a2, a3) == (126, 93, F(11, 42))
    report("D05-02", ok, f"(1){a1} (2){a2} (3){a3}; 오답(교집합 미차감) {wrong}")


def p03():
    # 원순열: 자리 0에 남편1(=사람 0)을 고정하고 나머지 5명을 5!가지로 배치
    people = list(range(6))  # 부부 k = {2k, 2k+1}
    a1 = a2 = good = 0
    total = 0
    for rest in permutations(people[1:]):
        seat = (0,) + rest
        adj = set()
        for i in range(6):
            u, v = seat[i], seat[(i + 1) % 6]
            if u // 2 == v // 2:
                adj.add(u // 2)
        total += 1
        if 0 in adj:
            a1 += 1
        if 0 in adj and 1 in adj:
            a2 += 1
        if not adj:
            good += 1
    a3 = F(good, total)
    all3 = factorial(2) * 2 ** 3
    wrong = factorial(4)  # (1)에서 부부 자리바꿈 2! 누락
    ok = (total, a1, a2, a3) == (120, 48, 24, F(4, 15)) and good == 120 - 3 * 48 + 3 * 24 - all3
    report("D05-03", ok, f"전체{total} (1){a1} (2){a2}, 세 쌍 {all3} (3){a3}={good}/{total}; 오답(2! 누락) {wrong}")


def p04():
    a1 = sum(comb(5, k) * comb(5, 5 - k) for k in range(6))  # (1+x)^10 의 x^5 계수 = 곱의 계수
    # 독립 확인: 다항식 전개를 직접
    poly = [0] * 11
    for i in range(6):
        for j in range(6):
            poly[i + j] += comb(5, i) * comb(5, j)
    a2 = sum(comb(5, k) ** 2 for k in range(6))
    a3 = sum(k * comb(5, k) ** 2 for k in range(6))
    # (3) 이중세기: 5명 남/5명 여 중 같은 수를 뽑고, 뽑힌 남자 중 대표 1명
    const = sum(1 for t in product((1, -1), repeat=10) if sum(t) == 0)  # (x+1/x)^10 상수항
    cnt = 0
    for k in range(6):
        for M in combinations(range(5), k):
            for W in combinations(range(5), k):
                cnt += len(M)
    ok = const == 252 and poly[5] == 252 and a1 == 252 and a2 == 252 and a3 == 630 and cnt == 630
    wrong = 5 * comb(10, 5)  # k C(5,k)^2 를 5 C(10,5) 로 착각
    report("D05-04", ok, f"(1)상수항 {const} (2)x^5계수 {poly[5]}, Σ={a2} (3){a3} 전수{cnt}; 오답 {wrong}")


def p05():
    s = set(permutations('AABBCC'))
    a1 = len(s)
    g = [p for p in s if all(p[i] != p[i + 1] for i in range(5))]
    a2 = len(g)
    circ = [p for p in g if p[0] != p[5]]
    a3 = F(len(circ), a1)
    wrong = F(a2, a1)  # 1번-6번 이웃을 빠뜨림
    ok = (a1, a2, a3) == (90, 30, F(4, 15))
    report("D05-05", ok, f"(1){a1} (2){a2} (3){a3}; 오답(양 끝 이웃 무시) {wrong}")


def p06():
    sols = [t for t in product(range(13), repeat=4) if sum(t) == 12]
    a1 = len(sols)
    b = [t for t in sols if max(t) <= 5]
    a2 = len(b)
    a3 = F(sum(1 for t in b if t[0] > t[1]), a2)
    wrong = F(sum(1 for t in b if t[0] >= t[1]), a2)
    ok = (a1, a2, a3) == (455, 125, F(54, 125))
    report("D05-06", ok, f"(1){a1} (2){a2} (3){a3}; x=y {sum(1 for t in b if t[0]==t[1])}개, 오답(등호 포함) {wrong}")


def p07():
    fs3 = list(product(range(3), repeat=5))
    a1 = len(fs3)
    a2 = sum(1 for f in fs3 if len(set(f)) == 3)
    fs4 = list(product(range(4), repeat=5))
    a3 = F(sum(1 for f in fs4 if len(set(f)) == 3), len(fs4))
    wrong = F(comb(4, 3) * (243), 1024)  # 공역 3원소 함수 전체를 곱함
    ok = (a1, a2, a3) == (243, 150, F(75, 128))
    report("D05-07", ok, f"(1){a1} (2){a2} (3){a3}; 오답(전사 조건 누락) {wrong}")


def invol(n):
    return sum(1 for p in permutations(range(n)) if all(p[p[i]] == i for i in range(n)))


def p08():
    a1, a2, i6 = invol(4), invol(5), invol(6)
    a3 = F(i6, factorial(6))
    rec = a2 + 5 * a1  # I6 = I5 + 5 I4
    ok = (a1, a2, i6, a3) == (10, 26, 76, F(19, 180)) and rec == i6 and invol(3) == 4
    report("D05-08", ok, f"(1)I4={a1} (2)I5={a2} (3)I6={i6}, P={a3}; 점화 I5+5I4={rec}")


def p09():
    seqs = set(permutations('CCCCCBBBBB'))  # C=500원 동전, B=1000원 지폐
    a1 = len(seqs)
    first_b = sum(1 for s in seqs if s[0] == 'B')

    def fails(s):
        c = 0
        for ch in s:
            if ch == 'C':
                c += 1
            else:
                if c == 0:
                    return True
                c -= 1
        return False

    a2 = sum(1 for s in seqs if fails(s))
    a3 = F(a1 - a2, a1)
    # 사람을 구별한 10! 경우로도 확인 (확률 동일)
    ok = (a1, first_b, a2, a3) == (252, 126, 210, F(1, 6)) and a2 == comb(10, 4)
    report("D05-09", ok, f"(1){a1},첫사람 지폐 {first_b} (2){a2}=C(10,4) (3){a3}")


def cyc(n):
    return sum(1 for c in product(range(3), repeat=n) if all(c[i] != c[(i + 1) % n] for i in range(n)))


def p10():
    a4, a5, a6 = cyc(4), cyc(5), cyc(6)
    rec_ok = all(cyc(n) + cyc(n - 1) == 3 * 2 ** (n - 1) for n in range(3, 9))
    a3 = F(a6, 3 ** 6)
    ok = (a4, a5, a6, a3) == (18, 30, 66, F(22, 243)) and rec_ok
    wrong = 3 * 2 ** 5  # 6번-1번 이웃 무시(경로로 셈)
    report("D05-10", ok, f"(1)a4={a4} (2)a5={a5}, 점화 a_n+a_(n-1)=3*2^(n-1) n=3..8 [증명-수치확인] (3)a6={a6}, P={a3}; 오답 {wrong}")


for fn in [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10]:
    try:
        fn()
    except Exception as e:  # noqa
        report(fn.__name__, False, repr(e))

n = sum(results)
print(f"SUMMARY T05: {n}/{len(results)} PASS")
if n != len(results):
    sys.exit(1)
