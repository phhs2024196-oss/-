#!/usr/bin/env python3
"""YSU-G24 (연세대 2024학년도 수시 논술 자연계열 수학) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import random
import re
import sys
from fractions import Fraction

import sympy as sp

PREFIX = "YSU-G24"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/YSU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # (1-1) DP: 상태 (x, y, 현재 방향, 전환 횟수). 오른쪽 출발, 격자점에서만 전환, 원점 전환 금지
    N = 21
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def cnt(px, py, d, k):
        if k > 5:
            return 0
        if (px, py) == (N, N):
            return 1 if k == 5 else 0
        tot = 0
        for nd in (0, 1):                       # 0: 오른쪽, 1: 위쪽
            if nd != d and (px, py) == (0, 0):
                continue
            nx, ny = (px + 1, py) if nd == 0 else (px, py + 1)
            if nx > N or ny > N:
                continue
            tot += cnt(nx, ny, nd, k + (nd != d))
        return tot
    c = cnt(0, 0, 0, 0)
    ok1 = c == int(sp.sympify(md_answer(1, "1")))
    # (1-2) 멈춘 점: 마지막 전환 격자점 (i,j) 에서 오른쪽/위로 직진해 원에 닿음. 원 안의 모든 격자점 전수
    R = 58
    best = sp.Integer(0)
    for i in range(R + 1):
        for j in range(R + 1):
            if i * i + j * j >= R * R:
                continue
            # 위로 직진: (i, sqrt(R^2-i^2)), 오른쪽 직진: (sqrt(R^2-j^2), j). 이동 거리 = 좌표합
            for v in (i + sp.sqrt(R * R - i * i), j + sp.sqrt(R * R - j * j)):
                if float(v) > float(best) + 1e-12:
                    best = v
    # 원 위 격자점에서 정확히 멈추는 경우(i^2+j^2=R^2)도 포함
    for i in range(R + 1):
        j2 = R * R - i * i
        jr = sp.sqrt(j2)
        if jr.is_integer and i + jr > best:
            best = i + jr
    ok2 = sp.simplify(best - sp.sympify(md_answer(1, "2"))) == 0
    report(1, ok1 and ok2, f"paths={c} max_dist={sp.nsimplify(best)}≈{float(best):.6f} (연속 최대 58√2≈{58*2**0.5:.4f})")


def p02():
    n = sp.symbols("n", positive=True, integer=True)
    # (2-1) 정적분 독립 계산 + 큰 n 수치 확인
    I = sp.integrate(sp.log(sp.Rational(1, 2) + x / 2), (x, 0, 1))
    import mpmath as mp
    mp.mp.dps = 30
    def term(N):
        return (mp.loggamma(2 * N + 1) - mp.loggamma(N + 1)) / N - mp.log(2 * N)
    num = term(10**7)
    ans1 = sp.sympify(md_answer(2, "1"))
    ok1 = sp.simplify(I - ans1) == 0 and abs(num - mp.mpf(str(sp.N(ans1, 25)))) < 1e-5
    # (2-2) 조건을 만족하는 b_n 의 최솟값을 작은 n 에서 직접 구해 b_n > n^3 확인 [증명-수치확인]
    def vp(m, p):
        e = 0
        while m % p == 0:
            m //= p; e += 1
        return e
    ok2 = True
    minratio = []
    for N in range(2, 41):
        cs = [c for c in range(1, 200) if 3 * mp.log(N, 2) <= c <= 4 * mp.log(N, 2)]
        fact = sp.factorial(N)
        for c in cs:
            # b_n 최솟값 = min over prime p<=N of p^(v_p(N!)*c+1)  (b_n 은 a_n^c 를 나누지 않아야 함)
            bmin = min(p ** (vp(int(fact), p) * c + 1) for p in sp.primerange(2, N + 1))
            ok2 &= bmin > N**3
            minratio.append(sp.Rational(bmin, N**2) / N)
    ans2 = md_answer(2, "2").strip()
    ok2 &= sp.sympify(ans2) == sp.oo
    report(2, ok1 and ok2, f"int={I} num(n=1e7)={mp.nstr(num, 8)} b_n_min/n^3>1 for n=2..40 [증명-수치확인]")


def p03():
    cards = [k for k in range(-20, 21) if k != 0]
    def EV(pairs):
        s = [Fraction(a + b) for a, b in pairs]
        E = sum(s) / 20
        V = sum(v * v for v in s) / 20 - E * E
        return E, V
    rng = random.Random(2024)
    # (3-1) 무작위 짝짓기 2000개에서 E(X) 일정
    Es = set()
    for _ in range(2000):
        c = cards[:]; rng.shuffle(c)
        Es.add(EV(list(zip(c[::2], c[1::2])))[0])
    ok1 = Es == {Fraction(int(sp.sympify(md_answer(3, "1"))))}
    # (3-2) 2-교환 국소 탐색(무작위 시작 200회)으로 V 최대 탐색 — 정답을 쓰지 않음
    def score(pairs):
        return sum(a * b for a, b in pairs)
    best = None
    for _ in range(200):
        c = cards[:]; rng.shuffle(c)
        pairs = [list(p) for p in zip(c[::2], c[1::2])]
        improved = True
        while improved:
            improved = False
            for i, j in itertools.combinations(range(20), 2):
                a, b = pairs[i]; cc, d = pairs[j]
                cur = a * b + cc * d
                for na, nb, nc, nd in ((a, cc, b, d), (a, d, b, cc)):
                    if na * nb + nc * nd > cur:
                        pairs[i] = [na, nb]; pairs[j] = [nc, nd]
                        improved = True
                        break
                if improved:
                    break
        v = EV(pairs)[1]
        if best is None or v > best[0]:
            best = (v, sorted(tuple(sorted(p, reverse=True)) for p in pairs))
    # 이웃끼리 짝짓기(크기순 정렬 후 인접)의 V 를 독립 계산
    srt = sorted(cards, reverse=True)
    adj = list(zip(srt[::2], srt[1::2]))
    vadj = EV(adj)[1]
    ans = Fraction(int(sp.sympify(md_answer(3, "2"))))
    ok2 = best[0] == vadj == ans and sorted(adj) == best[1]
    report(3, ok1 and ok2, f"E set={sorted(Es)} maxV(local search)={best[0]} V(adjacent)={vadj}")


def p04():
    # 생성 규칙을 유리수 정밀도로 여러 세대 시뮬레이션: 어떤 반직선도 A 를 지나지 않음 + 불변량 w·PA>0 확인
    A = (Fraction(2), Fraction(2))
    P0 = [(6, 4), (1, 5), (0, 3), (0, -1)]
    V0 = [(-1, 0), (0, -1), (1, -1), (1, 1)]
    base = [((Fraction(p[0]), Fraction(p[1])), (Fraction(v[0]), Fraction(v[1]))) for p, v in zip(P0, V0)]

    def through_A(ray):
        (px, py), (vx, vy) = ray
        dx, dy = A[0] - px, A[1] - py
        return vx * dy - vy * dx == 0 and vx * dx + vy * dy >= 0

    def inv(ray):
        (px, py), (vx, vy) = ray
        return (-vy) * (A[0] - px) + vx * (A[1] - py)

    def meet(r1, r2):
        (p1, v1), (p2, v2) = r1, r2
        det = v1[0] * (-v2[1]) - v1[1] * (-v2[0])
        if det == 0:
            return None
        bx, by = p2[0] - p1[0], p2[1] - p1[1]
        t = (bx * (-v2[1]) - by * (-v2[0])) / det
        s = (v1[0] * by - v1[1] * bx) / det
        if t <= 0 or s <= 0:                      # 시작점에서 만나는 경우 제외
            return None
        return (p1[0] + t * v1[0], p1[1] + t * v1[1])

    def generate(rays, a, gens, cap):
        rays = list(rays)
        seen = set(rays)
        for _ in range(gens):
            new = []
            for r1, r2 in itertools.combinations(rays, 2):
                P = meet(r1, r2)
                if P is None:
                    continue
                for v in ((a * r1[1][0] + r2[1][0], a * r1[1][1] + r2[1][1]),
                          (r1[1][0] + a * r2[1][0], r1[1][1] + a * r2[1][1])):
                    r = (P, v)
                    if r not in seen:
                        seen.add(r); new.append(r)
                if len(new) > cap:
                    break
            rays += new
            if len(rays) > cap:
                break
        return rays

    avals = [Fraction(k, 7) for k in range(1, 50)] + [Fraction(1, 100), Fraction(100)]
    # (4-1) l1, l2 만, 1세대
    ok1 = all(not any(through_A(r) for r in generate(base[:2], a, 1, 10)) for a in avals)
    # (4-1) 대표 오답 재현: a<0 이면 가능해짐(예: a=-1/2 → (1,-2)//PA)
    neg = any(through_A(r) for r in generate(base[:2], Fraction(-1, 2), 1, 10))
    # (4-2) 네 반직선, 2세대(반직선 수 상한) + 불변량
    ok2 = True
    nrays = 0
    for a in avals[::4] + [Fraction(1)]:
        rays = generate(base, a, 2, 400)
        nrays = max(nrays, len(rays))
        ok2 &= all(not through_A(r) and inv(r) > 0 for r in rays)
    ok2 &= [inv(r) for r in base] == [2, 1, 1, 1]
    want = (md_answer(4, "1").strip(), md_answer(4, "2").strip())
    okans = all(sp.sympify(w) == sp.S.EmptySet for w in want)
    report(4, ok1 and ok2 and okans and neg,
           f"a {len(avals)}개 표본·최대 {nrays}개 반직선에서 A 미통과, w·P_iA=[2,1,1,1] (a<0 반례 재현={neg}) [증명-수치확인]")


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
