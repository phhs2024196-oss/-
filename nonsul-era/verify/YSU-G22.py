#!/usr/bin/env python3
"""YSU-G22 (연세대 2022학년도 수시 논술 자연계열 수학) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import random
import re
import sys
from fractions import Fraction

import sympy as sp

PREFIX = "YSU-G22"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/YSU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # 격자점 직접 열거
    X = [sum(1 for x in range(a + 1) for y in range(a + 1) if x + y == a) for a in range(1, 7)]
    Y = [sum(1 for x in range(a + 1) for y in range(a + 1) if x + y <= a) for a in range(1, 7)]
    EX, EY = sp.Rational(sum(X), 6), sp.Rational(sum(Y), 6)
    ok1 = sp.simplify(EX - sp.sympify(md_answer(1, "1"))) == 0
    ok2 = sp.simplify(EY - sp.sympify(md_answer(1, "2"))) == 0
    # 대표 오답: 빗변 격자점을 a 개(끝점 하나 누락)로 셈 → 7/2
    report(1, ok1 and ok2, f"X={X} E(X)={EX}  Y={Y} E(Y)={EY}  (오답 재현: 끝점 누락 E={sp.Rational(21, 6)})")


def _b(a_next, hi):
    """쌍곡선 x^2/a^2 - y^2 = 1 (x>=a, y>=0) 과 y=ln(1+x) 의 교점 x 좌표(수치)."""
    import mpmath as mp
    F = lambda x: mp.log(1 + x) - mp.sqrt(x**2 - a_next**2) / a_next
    # F(a)>0, 충분히 큰 x 에서 F<0 → 이분법
    lo, up = mp.mpf(a_next), mp.mpf(max(hi, a_next) * 2 + 10)
    while F(up) > 0:
        up *= 2
    for _ in range(400):                       # 자체 이분법(정밀도 dps 에 맞춰)
        mid = (lo + up) / 2
        if F(mid) > 0:
            lo = mid
        else:
            up = mid
    return (lo + up) / 2, F


def p02():
    import mpmath as mp
    mp.mp.dps = 50
    # (2-1) [증명-수치확인] 조건을 만족하는 무작위 수열에서 a_{n+1}<b_n<a_n, 교점 유일(부호 변화 1회)
    random.seed(1)
    bad = 0
    for _ in range(300):
        an = mp.mpf(random.uniform(0.01, 20))
        lim = 1 / mp.sqrt(1 + 1 / an**2)                 # a_{n+1} 의 상한
        an1 = lim * mp.mpf(random.uniform(0.001, 1))
        b, F = _b(an1, an)
        xs = [an1 + (an * 3 - an1) * k / 400 for k in range(1, 401)]
        sign_changes = sum(1 for u, v in zip(xs, xs[1:]) if F(u) * F(v) < 0)
        if not (an1 < b < an) or sign_changes > 1:
            bad += 1
    ok1 = bad == 0
    # (2-2) 주어진 a_n 이 n=1..200 에서 (2-1) 조건을 만족하는지
    a = lambda n: 1 / mp.sqrt(n) * (1 + mp.mpf(1) / n) ** (-mp.mpf(n) / 2)
    cond = all(1 + 1 / a(n) ** 2 <= 1 / a(n + 1) ** 2 for n in range(1, 201))
    # 극한: sqrt(n) a_n, sqrt(n) a_{n+1} → 기호 계산, 조임으로 sqrt(n) b_n 도 같은 극한
    n = sp.symbols("n", positive=True)
    an_s = 1 / sp.sqrt(n) * (1 + 1 / n) ** (-n / 2)
    L1 = sp.limit(sp.sqrt(n) * an_s, n, sp.oo)
    L2 = sp.limit(sp.sqrt(n) * an_s.subs(n, n + 1), n, sp.oo)
    x = sp.symbols("x", positive=True)
    L3 = sp.limit(sp.log(1 + x) / x, x, 0)
    lim_sym = sp.simplify(L1 * L3)
    ans = sp.sympify(md_answer(2, "2"))
    ok2 = cond and sp.simplify(L1 - L2) == 0 and sp.simplify(lim_sym - ans) == 0
    # 수치 확인: n=10^12 에서 sqrt(n) ln(1+b_n)
    N = mp.mpf(10) ** 12
    bN, _ = _b(a(N + 1), a(N))
    num = mp.sqrt(N) * mp.log(1 + bN)
    ok3 = abs(num - mp.mpf(float(ans.evalf(30)))) < 1e-5
    report(2, ok1 and ok2 and ok3,
           f"[증명-수치확인] 무작위 300개 위반 {bad}; 조건 n<=200 {cond}; lim={lim_sym}; n=1e12 수치 {mp.nstr(num, 10)}")


def _g_enum(N, m):
    return sum(1 for a in range(1, N + 1) if a * m + m * (m - 1) // 2 <= N)


def p03():
    Nn, m, r = sp.symbols("N m r")
    expr = sp.sympify(md_answer(3, "1"), locals={"N": Nn, "m": m, "r": r})
    bad_pos, bad_max = 0, 0
    for N in range(1, 301):
        for mm in range(1, 40, 2):
            g = _g_enum(N, mm)
            v = expr.subs({Nn: N, m: mm, r: N % mm})
            if v >= 0 and v != g:
                bad_pos += 1
            if max(0, v) != g:
                bad_max += 1
    ok1 = bad_pos == 0 and bad_max == 0
    # (3-2) 직접 정의대로: f(N) = 합이 정확히 N 인 (홀수 m, 첫째항 a>=1) 의 m 합
    def f(N):
        return sum(mm for mm in range(1, N + 1, 2) for a in range(1, N + 1) if a * mm + mm * (mm - 1) // 2 == N)
    f21 = f(21)
    S = sum(f(N) for N in range(1, 201))
    ok2 = f21 == 4 and S == int(md_answer(3, "2"))
    report(3, ok1 and ok2, f"g 식 vs 열거 불일치(값>=0) {bad_pos}, max(0,·) 불일치 {bad_max}; f(21)={f21}; sum f(N)={S}")


DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def _angle_ok(trip):
    import math
    angs = sorted(math.degrees(math.atan2(v[1], v[0])) % 360 for v in trip)
    gaps = [angs[1] - angs[0], angs[2] - angs[1], 360 - angs[2] + angs[0]]
    return all(g < 180 - 1e-9 for g in gaps)


def _check_config(verts, edges, L):
    """verts: {이름: (x,y)}, edges: [(이름,이름)]. 삼각형 (0,0),(L,0),(0,L).
    조건 (가)(다)(라)(나: 내부 꼭짓점 주변) 점검 후 경계벡터 합·변별 다각형 수 반환."""
    def where(p):
        x, y = p
        if x > 0 and y > 0 and x + y < L:
            return "in"
        if y == 0 and 0 < x < L:
            return "OA"
        if x == 0 and 0 < y < L:
            return "OB"
        if x + y == L and x > 0 and y > 0:
            return "AB"
        return "bad"
    def unit(p, q):
        dx, dy = q[0] - p[0], q[1] - p[1]
        g = max(abs(dx), abs(dy))
        return (Fraction(dx, g), Fraction(dy, g))
    nbr = {k: [] for k in verts}
    for a, b in edges:
        nbr[a].append(b); nbr[b].append(a)
        d = unit(verts[a], verts[b])
        assert d in [tuple(map(Fraction, v)) for v in DIRS], "조건(가) 위반"
    perp = {"OA": (0, -1), "OB": (-1, 0), "AB": (1, 1)}
    bsum = [0, 0]; cnt = {"OA": 0, "OB": 0, "AB": 0}
    for k, p in verts.items():
        w = where(p)
        assert w != "bad"
        if w == "in":
            us = [unit(p, verts[q]) for q in nbr[k]]
            assert len(us) == 3 and sum(1 for u in us if 0 in u) == 2 and _angle_ok(us), "조건(나)(라) 위반"
            assert sum(u[0] for u in us) == 0 and sum(u[1] for u in us) == 0
        else:
            assert len(nbr[k]) == 1 and where(verts[nbr[k][0]]) == "in"
            u = unit(verts[nbr[k][0]], p)
            assert u == tuple(map(Fraction, perp[w])), "조건(다) 위반"
            bsum[0] += u[0]; bsum[1] += u[1]; cnt[w] += 1
    polys = {s: c + 1 for s, c in cnt.items()}    # 변이 경계점으로 c+1 조각, 조각마다 다른 볼록 다각형
    return tuple(bsum), polys


def p04():
    # (4-1) 8방향에서 서로 다른 3개, 축 평행 2개 + 대각선 1개, 이웃 각 <180° 인 조합 전수
    trips = [t for t in itertools.combinations(DIRS, 3)
             if sum(1 for v in t if 0 in v) == 2 and _angle_ok(t)]
    ok1 = len(trips) == 4 and all(sum(v[0] for v in t) == 0 and sum(v[1] for v in t) == 0 for t in trips)
    # 축 평행 조건을 빼면 반례가 있는지(조건 (라)의 필요성) — 참고 출력
    others = [t for t in itertools.combinations(DIRS, 3) if _angle_ok(t)
              and (sum(v[0] for v in t), sum(v[1] for v in t)) != (0, 0)]
    # (4-2)(4-3) [증명-수치확인] 조건을 만족하는 분할 예시들에서 경계벡터 합=0, 세 변 다각형 수 동일
    random.seed(2)
    L = 2022
    results = []
    for _ in range(50):                      # 예시 1: 내부 꼭짓점 1개 (오른쪽 위·왼쪽·아래)
        p, q = random.randint(1, 1000), random.randint(1, 1000)
        s = (L - p - q) // 2
        if (L - p - q) % 2 or s <= 0:
            continue
        v = {"V": (p, q), "P1": (p + s, q + s), "P2": (0, q), "P3": (p, 0)}
        results.append(_check_config(v, [("V", "P1"), ("V", "P2"), ("V", "P3")], L))
    for _ in range(50):                      # 예시 2: 내부 꼭짓점 4개 (정사각형 둘레 + 팔각 구조)
        c, d = random.randint(1, 300), random.randint(1, 300)
        e = random.randint(1, 300)
        V4, V2 = (c, c), (c + d, c + d)
        V3, V1 = (c + d + e, c + d), (c + d, c + d + e)
        s3 = L - V3[0] - V3[1]; s1 = L - V1[0] - V1[1]
        if s3 % 2 or s1 % 2:
            continue
        v = {"V4": V4, "V2": V2, "V3": V3, "V1": V1,
             "a": (0, c), "b": (c, 0), "c3": (V3[0], 0), "c1": (0, V1[1]),
             "t3": (V3[0] + s3 // 2, V3[1] + s3 // 2), "t1": (V1[0] + s1 // 2, V1[1] + s1 // 2)}
        E = [("V4", "V2"), ("V4", "a"), ("V4", "b"), ("V2", "V3"), ("V2", "V1"),
             ("V3", "c3"), ("V3", "t3"), ("V1", "c1"), ("V1", "t1")]
        results.append(_check_config(v, E, L))
    ok2 = results and all(r[0] == (0, 0) for r in results)
    ok3 = results and all(len(set(r[1].values())) == 1 for r in results)
    # (4-3) 대수 단계: l(0,-1)+m(-1,0)+n(1,1)=0 ⇒ l=m=n
    l, m, n = sp.symbols("l m n")
    sol = sp.solve([n - m, n - l], [l, m], dict=True)[0]
    ok4 = sol == {l: n, m: n}
    report(4, ok1 and ok2 and ok3 and ok4,
           f"[증명-수치확인] 4-1 조합 {len(trips)}개 모두 합 0(축조건 없으면 합≠0 조합 {len(others)}개); "
           f"분할 예시 {len(results)}개 경계벡터합 0·세 변 다각형 수 동일; l=m=n")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
