"""D06. 조건부확률·확률분포·표본평균 — 검산 스크립트 (전수 열거 + Fraction, 정규분포는 제시된 표 값 사용)"""
import sys
from fractions import Fraction as F
from itertools import product, combinations
from math import comb
from scipy.stats import norm

results = []


def report(pid, ok, msg):
    results.append(ok)
    print(f"{pid} {'PASS' if ok else 'FAIL'}  {msg}")


def EV(dist):
    e = sum(x * p for x, p in dist.items())
    v = sum(x * x * p for x, p in dist.items()) - e * e
    return e, v


# 문제에 제시하는 표준정규분포표 P(0<=Z<=z)
TABLE = {F(1): F('0.3413'), F(3, 2): F('0.4332'), F(2): F('0.4772'), F(5, 2): F('0.4938')}


def table_ok():
    return all(abs(norm.cdf(float(z)) - 0.5 - float(v)) < 5e-5 for z, v in TABLE.items())


def p01():
    # 표본공간: (불량여부, 1차판정, 2차판정) 전수
    pd, sens, fp = F(1, 10), F(4, 5), F(1, 5)
    joint = {}
    for d, t1, t2 in product((1, 0), repeat=3):
        p = pd if d else 1 - pd
        q = sens if d else fp
        p *= (q if t1 else 1 - q) * (q if t2 else 1 - q)
        joint[(d, t1, t2)] = p
    a1 = sum(p for (d, t1, t2), p in joint.items() if t1)
    a2 = sum(p for (d, t1, t2), p in joint.items() if t1 and d) / a1
    both = sum(p for (d, t1, t2), p in joint.items() if t1 and t2)
    a3 = sum(p for (d, t1, t2), p in joint.items() if t1 and t2 and d) / both
    wrong = sens  # P(판정|불량)과 혼동
    ok = (a1, a2, a3) == (F(13, 50), F(4, 13), F(16, 25))
    report("D06-01", ok, f"(1){a1}, 복귀 {back} (2){a2} (3){a3}; 오답(조건 뒤바꿈) {wrong}")


def p02():
    cards = [1, 2, 2, 3, 3, 3]
    pairs = list(combinations(range(6), 2))
    dist = {}
    for i, j in pairs:
        s = cards[i] + cards[j]
        dist[s] = dist.get(s, 0) + F(1, len(pairs))
    e, v = EV(dist)
    ydist = {3 * x - 4: p for x, p in dist.items()}
    ey, vy = EV(ydist)
    n = 1
    while not (vy / n <= F(1, 4)):
        n += 1
    ok = dist == {3: F(2, 15), 4: F(4, 15), 5: F(6, 15), 6: F(3, 15)} and (e, v) == (F(14, 3), F(8, 9)) \
        and (ey, vy, n) == (10, 8, 32)
    wrong = 1
    while not (3 * v / wrong <= F(1, 4)):  # V(aX+b)를 aV(X)로 착각
        wrong += 1
    report("D06-02", ok, f"(1)분포 {dict(sorted(dist.items()))} (2)E={e},V={v} (3)E(Y)={ey},V(Y)={vy}, n={n}; 오답 {wrong}")


def p03():
    p = F(sum(1 for a, b in product(range(1, 7), repeat=2) if (a + b) % 3 == 0), 36)
    n = 450
    ex, vx = n * p, n * p * (1 - p)
    es, vs = 4 * ex - n, 16 * vx  # S = 3X - (450 - X)
    # S>=210 <=> X>=165, z = (165-150)/10
    sd = next(k for k in range(1, 100) if k * k == vx)
    z = (F(165) - ex) / sd
    a3 = F(1, 2) - TABLE[z]
    ok = p == F(1, 3) and (ex, vx) == (150, 100) and (es, vs) == (150, 1600) and z == F(3, 2) \
        and a3 == F('0.0668') and table_ok()
    wrong = (F(210) - es) / (2 * sd)  # V(4X)를 4V(X)로 착각 -> z=3
    report("D06-03", ok, f"(1)p={p},E={ex},V={vx} (2)E(S)={es},σ(S)={int(vs**0.5)} (3)z={z}, P={float(a3)}; 표 검증 OK, 오답 z={wrong}")


def p04():
    pop = {0: F(1, 4), 1: F(1, 2), 2: F(1, 4)}  # 카드 0,1,1,2
    m, s2 = EV(pop)
    # n=2 복원추출 표본평균 분포 전수
    cards = [0, 1, 1, 2]
    d2 = {}
    for a, b in product(cards, repeat=2):
        x = F(a + b, 2)
        d2[x] = d2.get(x, 0) + F(1, 16)
    e2, v2 = EV(d2)
    n = 1
    while not (s2 / n <= F(1, 100)):
        n += 1
    sd = F(1, 10)
    z = (F(6, 5) - m) / sd
    a3 = 2 * TABLE[z]
    ok = (m, s2) == (1, F(1, 2)) and d2 == {0: F(1, 16), F(1, 2): F(4, 16), 1: F(6, 16), F(3, 2): F(4, 16), 2: F(1, 16)} \
        and (e2, v2) == (1, F(1, 4)) and n == 50 and a3 == F('0.9544')
    wrong = next(k for k in range(1, 100) if s2 / k <= F(1, 10))  # σ(X̄)를 σ²/n 으로 착각
    ok = ok and wrong == 5
    report("D06-04", ok, f"(1)m={m},σ²={s2} (2)E={e2},V={v2} (3)n={n}, P={float(a3)}; 오답 n={wrong}")


def p05():
    A = ['W'] * 3 + ['K'] * 2
    B = ['W'] * 2 + ['K'] * 2
    pw_both = F(0)
    p_both = F(0)
    p_third = F(0)
    for t in range(5):  # 옮길 공
        boxB = B + [A[t]]
        for pair in combinations(range(5), 2):
            pr = F(1, 5) * F(1, comb(5, 2))
            if all(boxB[i] == 'W' for i in pair):
                p_both += pr
                if A[t] == 'W':
                    pw_both += pr
                rest = [boxB[i] for i in range(5) if i not in pair]
                p_third += pr * F(rest.count('W'), 3)
    a1, a2, a3 = p_both, pw_both / p_both, p_third / p_both
    wrong = F(3, 5)  # 사전확률 그대로
    ok = (a1, a2, a3) == (F(11, 50), F(9, 11), F(3, 11))
    report("D06-05", ok, f"(1){a1}, 복귀 {back} (2){a2} (3){a3}; 오답(사전확률) {wrong}")


def p06():
    p = F(2, 3)
    end4 = F(0)
    awin = F(0)
    games_awin = F(0)
    for seq in product((1, 0), repeat=5):  # 5경기를 가상으로 모두 치름
        pr = F(1)
        for g in seq:
            pr *= p if g else 1 - p
        a = b = 0
        for k, g in enumerate(seq, 1):
            a += g
            b += 1 - g
            if a == 3 or b == 3:
                break
        if k == 4:
            end4 += pr
        if a == 3:
            awin += pr
            games_awin += pr * k
    a3 = games_awin / awin
    ok = (end4, awin, a3) == (F(10, 27), F(64, 81), F(31, 8))
    wrong = games_awin + 0  # 조건부 아닌 값
    report("D06-06", ok, f"(1){end4} (2){awin} (3)E(N|A승)={a3}; 오답(조건부 미적용) {wrong}")


def p07():
    # B: N(103, s^2), P(B<=100)=0.1587 -> (100-103)/s = -1 -> s=3
    s = None
    for cand in range(1, 10):
        z = F(103 - 100, cand)
        if z in TABLE and F(1, 2) - TABLE[z] == F('0.1587'):
            s = cand
    pa = F(1, 2) - TABLE[F(106 - 100, 3)]
    pb = F(1, 2) - TABLE[F(106 - 103, s)]
    tot = F(7, 10) * pa + F(3, 10) * pb
    a3 = F(3, 10) * pb / tot
    wrong = pb / (pa + pb)  # 생산 비율 무시
    ok = round(float(wrong), 3) == 0.874 and s == 3 and pa == F('0.0228') and pb == F('0.1587') and tot == F('0.06357') and a3 == F(1587, 2119) \
        and round(float(a3), 3) == 0.749 and table_ok()
    report("D06-07", ok, f"(1)σ={s}, P(A≥106)={float(pa)} (2)P={float(tot)} (3){a3}≈{float(a3):.4f}; 오답 {float(wrong):.3f}")


def p08():
    rolls = list(product(range(1, 7), repeat=3))
    dX, dY = {}, {}
    for r in rolls:
        dX[max(r)] = dX.get(max(r), 0) + F(1, 216)
        dY[min(r)] = dY.get(min(r), 0) + F(1, 216)
    a1 = sum(p for k, p in dX.items() if k <= 4)
    eX, vX = EV(dX)
    eY, vY = EV(dY)
    sym = all(dY[k] == dX[7 - k] for k in range(1, 7))
    ok = a1 == F(8, 27) and eX == F(119, 24) and eY == F(49, 24) and sym and vX == vY
    wrong = F(7, 2)
    report("D06-08", ok, f"(1)P(X≤4)={a1} (2)E(X)={eX} (3)E(Y)={eY}, P(Y=k)=P(X=7-k) {sym}, V(Y)=V(X)={vX}")


def p09():
    p = F(2, 3)
    # (1) 2:2에서 두 판 안에 끝날 확률
    a1 = p * p + (1 - p) * (1 - p)
    # 정확한 흡수확률: 상태 0..4 선형방정식 풀이 (가우스 소거, Fraction)
    import sympy as sp
    x = sp.symbols('x0:5')
    eqs = [sp.Eq(x[0], 0), sp.Eq(x[4], 1)] + [sp.Eq(x[k], sp.Rational(2, 3) * x[k + 1] + sp.Rational(1, 3) * x[k - 1]) for k in (1, 2, 3)]
    sol = sp.solve(eqs, x)
    a2, a3 = sol[x[2]], sol[x[1]]
    # 시뮬레이션 대신 긴 유한 전수: 2:2에서 2m판 이내 A 승리 확률 -> 4/5 수렴
    approx = sum(F(4, 9) ** j * F(4, 9) for j in range(60))
    back = 2 * p * (1 - p)
    ok = back == F(4, 9) and a1 == F(5, 9) and a2 == sp.Rational(4, 5) and a3 == sp.Rational(8, 15) and abs(approx - F(4, 5)) < F(1, 10 ** 15)
    wrong = p * p  # 2:2 복귀 상태를 무시하고 두 판 연승만 셈
    report("D06-09", ok, f"(1){a1}, 복귀 {back} (2){a2} (3){a3}; 부분합 수렴 확인, 오답(복귀 무시) {wrong}")


def p10():
    vals = range(1, 9)
    X = {k: F(1, 8) for k in vals}
    e1, v1 = EV(X)
    # 1회 재추첨: 모든 전략(다시 뽑을 값의 집합) 전수
    best2, bestS = None, None
    for mask in range(256):
        S = {k for k in vals if mask >> (k - 1) & 1}
        d = {}
        for k in vals:
            if k in S:
                for j in vals:
                    d[j] = d.get(j, 0) + F(1, 64)
            else:
                d[k] = d.get(k, 0) + F(1, 8)
        e, v = EV(d)
        if best2 is None or e > best2[0]:
            best2, bestS = (e, v), S
    # 2회 재추첨: 1단계 집합 S1, 2단계 집합 S2 전수
    best3 = F(0)
    for m1 in range(256):
        S1 = {k for k in vals if m1 >> (k - 1) & 1}
        for m2 in range(256):
            S2 = {k for k in vals if m2 >> (k - 1) & 1}
            e = F(0)
            for k in vals:
                if k not in S1:
                    e += F(k, 8)
                else:
                    for j in vals:
                        if j not in S2:
                            e += F(j, 64)
                        else:
                            e += F(1, 64) * F(9, 2)
            best3 = max(best3, e)
    same4 = sum(F(k, 8) for k in vals if k >= 5) + F(1, 2) * best2[0]  # 두 단계 모두 기준 4
    ok = same4 == 6 and (e1, v1) == (F(9, 2), F(21, 4)) and best2 == (F(11, 2), F(17, 4)) and bestS == {1, 2, 3, 4} and best3 == F(97, 16)
    report("D06-10", ok, f"(1)E={e1},V={v1} (2)최적 재추첨 {sorted(bestS)}, E={best2[0]},V={best2[1]} (3)전략 65536개 전수 최댓값 {best3}; 오답(기준 4 유지) {same4}")


for fn in [p01, p02, p03, p04, p05, p06, p07, p08, p09, p10]:
    try:
        fn()
    except Exception as e:  # noqa
        report(fn.__name__, False, repr(e))

n = sum(results)
print(f"SUMMARY T06: {n}/{len(results)} PASS")
if n != len(results):
    sys.exit(1)
