#!/usr/bin/env python3
"""AJOU-G23A (아주대 2023학년도 논술 자연계열 오전) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "AJOU-G23A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/AJOU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def _is_ap(s):
    return s[1] - s[0] == s[2] - s[1]


def _is_gp(s):
    return all(v != 0 for v in s) and s[1] * s[1] == s[0] * s[2]


def _ok(s):
    return _is_ap(s) or _is_gp(s)


def p01():
    import itertools
    # [1-1](1): 미지수 a,b,c,d,e — 행·열 조건으로 연립(영이 아닌 등비수열)
    a, b, c, d, e = sp.symbols("a b c d e", real=True)
    eqs = [sp.Eq(c + d, 20), sp.Eq(a + e, 20),            # 2행·2열 등차
           sp.Eq(a**2, 10 * b), sp.Eq(e**2, 400),          # 1행 (10,a,b), 3행 (40,e,10) 등비
           sp.Eq(c**2, 400), sp.Eq(d**2, 10 * b)]          # 1열 (10,c,40), 3열 (b,d,10) 등비
    sols = [s for s in sp.solve(eqs, [a, b, c, d, e], dict=True)
            if all(s[v] != 0 for v in (a, b, d, e, c))]
    sums = {sp.simplify(sum(s[v] for v in (a, b, c, d, e))) for s in sols}
    ok1 = len(sums) == 1 and sums.pop() == sp.sympify(md_answer(1, "1-1"))
    # [1-1](2): 9 이하 자연수 3x3 표 전수(행·열 모두 등차 또는 등비, 홀짝 조건)
    L = [s for s in itertools.product(range(1, 10), repeat=3) if _ok(s)]
    cnt = 0
    for r1 in L:
        if not (r1[0] % 2 == 1 and r1[1] % 2 == 0 and r1[2] % 2 == 0):
            continue
        for r2 in L:
            if r2[0] % 2 != 0:
                continue
            for r3 in L:
                if r3[0] % 2 != 1:
                    continue
                if all(_ok((r1[j], r2[j], r3[j])) for j in range(3)):
                    cnt += 1
    ok2 = cnt == int(md_answer(1, "1-2"))
    # [1-1](3): 모든 칸 합 = Σ n(a_k+b_k)/2 = n(α+β)/2 = 20, α+β=5 → n; 판별식 → p 상한
    nn, p = sp.symbols("nn p", positive=True)
    n_val = sp.solve(sp.Eq(nn * 5 / 2, 20), nn)[0]
    p_sup = sp.solve(sp.Eq(sp.discriminant(x**2 - 5 * x + p, x), 0), p)[0]
    # 구성 확인: 모든 행이 같은 등차수열(열은 상수=등차)이면 α=n·a1, β=n·an 으로 임의의 서로 다른 α+β=5 실현
    a1 = sp.Rational(1, 7); an = sp.Rational(5, 8) - a1
    tbl = [[a1 + (an - a1) * j / (int(n_val) - 1) for j in range(int(n_val))] for _ in range(int(n_val))]
    tot = sum(sum(r) for r in tbl); al = sum(r[0] for r in tbl); be = sum(r[-1] for r in tbl)
    m = re.match(r"n\s*=\s*(\d+);\s*p\s*<\s*(.+)$", md_answer(1, "1-3").strip())
    ok3 = (m and int(m.group(1)) == n_val and sp.sympify(m.group(2)) == p_sup
           and tot == 20 and sp.simplify(al + be - 5) == 0)
    # [1-2](1): log2 변환 후 등차(4, 공차 4) → 원래 2^(4k), k=1..10 합
    k = sp.symbols("k", integer=True, positive=True)
    dd = sp.Rational(40 - 4, 9)
    S = sp.summation(2**(4 + (k - 1) * dd), (k, 1, 10))
    ok4 = sp.simplify(S - sp.sympify(md_answer(1, "2-1"))) == 0
    # [1-2](2): (g∘f) 변환표가 3-등차등비표가 되는 a,b 결정 후 교점 개수 수치 확인
    A, B = sp.symbols("A B", real=True)
    g = x**3 + A * x**2 + B
    fval = [sp.sin(sp.pi * v / 2) for v in range(1, 10)]
    T = [[g.subs(x, fval[3 * i + j]) for j in range(3)] for i in range(3)]
    cand = []
    for pa, pb in itertools.product([0, 1], repeat=2):   # 1행·2행 각각 등차(0)/등비(1) 가정
        eq = []
        for row, kind in ((T[0], pa), (T[1], pb)):
            eq.append(sp.expand(row[1] * 2 - row[0] - row[2]) if kind == 0 else sp.expand(row[1]**2 - row[0] * row[2]))
        for s in sp.solve(eq, [A, B], dict=True):
            if not (s.get(A, A).is_real and s.get(B, B).is_real):
                continue
            num = [[sp.nsimplify(v.subs(s)) for v in r] for r in T]
            if all(_ok(tuple(r)) for r in num) and all(_ok(tuple(num[i][j] for i in range(3))) for j in range(3)):
                cand.append(s)
    import numpy as np
    vals = set()
    for s in cand:
        h = sp.lambdify(x, g.subs(s) - sp.sin(sp.pi * x / 2), "numpy")
        xs = np.linspace(-3, 3, 600001)
        hv = h(xs)
        vals.add(int(np.sum(np.sign(hv[:-1]) * np.sign(hv[1:]) < 0)))
    ok5 = len(cand) >= 1 and vals == {int(md_answer(1, "2-2"))}
    report(1, ok1 and ok2 and ok3 and ok4 and ok5,
           f"합ok={ok1} 표개수={cnt} n={n_val},p<{p_sup} 합(2^4k)ok={ok4} a,b={cand} 교점={vals}")


def p02():
    import math
    import numpy as np
    # [2-1](1): 원 C1(중심(1,r)) 이 y=p1x 에 접할 조건, C2(중심(r,1)) 가 y=p1x 에 접할 조건을 직접 풀어 비교
    worst = 1e9
    for th in np.linspace(1e-4, math.pi / 2 - 1e-4, 2000):
        p1 = math.tan(th)
        q = math.sqrt(1 + p1 * p1)
        r1 = p1 / (1 + q)          # 중심 (1,r): 직선 아래쪽, (p1-r)/q = r
        r2 = 1 / (p1 + q)          # 중심 (r,1): 직선 위쪽, (1-p1 r)/q = r
        assert abs(abs(p1 - r1) / q - r1) < 1e-12 and abs(abs(p1 * r2 - 1) / q - r2) < 1e-12
        worst = min(worst, float(5 * (r1 + r2)) - 4)
    ok1 = worst > 0
    # [2-1](2): n 큰 값에서 직접 합 계산(θ_k 는 기울기 각 차이, r_k=tan(θ_k/2))
    def L(N):
        ps = [math.tan(kk * math.pi / (2 * (N + 1))) for kk in range(1, N + 1)]
        angs = [math.atan(ps[0])] + [math.atan(ps[i]) - math.atan(ps[i - 1]) for i in range(1, N)] + [math.pi / 2 - math.atan(ps[-1])]
        return sum((kk) * math.tan(angs[kk] / 2) for kk in range(N + 1)) / N
    lim_num = L(200000)
    ok2 = abs(lim_num - float(sp.sympify(md_answer(2, "1-2")))) < 1e-4
    # [2-1](3): p_k=k 일 때 Σ 1/tanθ_k 를 각도로 직접 계산해 82 가 되는 n 탐색
    def S3(N):
        angs = [math.atan(1)] + [math.atan(kk) - math.atan(kk - 1) for kk in range(2, N + 1)] + [math.pi / 2 - math.atan(N)]
        return sum(1 / math.tan(a) for a in angs)
    hits = [N for N in range(1, 60) if abs(S3(N) - 82) < 1e-6]
    ok3 = hits == [int(md_answer(2, "1-3"))]
    # [2-2](1): 근의 합 34 → 근의 개수 = 34/2 (대칭 쌍 평균 2). 예시 함수로 확인
    N = sp.sympify(md_answer(2, "2-1"))
    m = (N - 1) // 2
    f = (x - 2)**2 * sp.prod([(x - 2)**2 - kk**2 for kk in range(1, m + 1)])   # f(2+x)=f(2-x)
    rts = sp.roots(sp.Poly(f, x))
    ok4 = (N % 2 == 1 and sum(rts) == 34 and len(rts) == N
           and sp.expand(f.subs(x, 2 + t) - f.subs(x, 2 - t)) == 0)
    # 짝수 개 근(2 미포함)이면 합 = 4m ≠ 34 확인
    ok4 = ok4 and 34 % 4 != 0
    # [2-2](2): 수치 적분
    import mpmath as mp
    val = mp.quad(lambda u: u * abs(mp.cos(u)) / (3 + mp.cos(u)**2), [0, mp.pi / 2, mp.pi])
    ok5 = abs(val - mp.mpf(str(sp.N(sp.sympify(md_answer(2, "2-2")), 30)))) < 1e-20
    report(2, ok1 and ok2 and ok3 and ok4 and ok5,
           f"[증명-수치확인] min 5(r1+r2)-4={worst:.4f}; lim≈{lim_num:.6f}; n={hits}; 근개수={N}; 적분≈{mp.nstr(val, 12)}")


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
