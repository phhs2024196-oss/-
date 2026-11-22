#!/usr/bin/env python3
"""서울과학기술대 2024학년도 자연계열 오전 수리논술 (SNUT-G24A-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산해 YAML answer.value 와 비교.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다."""
import sys

import sympy as sp

PREFIX = "SNUT-G24A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
a, b, k = sp.symbols("a b k", real=True)
SYM = {"x": x, "t": t, "n": n, "a": a, "b": b, "k": k, "pi": sp.pi}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0, raw=False):
    """yaml 의 subparts[sub].answer.value 를 읽는다(raw=True 면 문자열 그대로)."""
    import pathlib
    import yaml
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/SNUT/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    v = d["subparts"][sub]["answer"]["value"]
    return v if raw else sp.sympify(v, locals=SYM)


def eq(u, v):
    return sp.simplify(sp.nsimplify(u) - sp.nsimplify(v)) == 0


def p01():
    """[1.1] θ 전수(교점 집합 조건 확인), [1.2] 사차함수 조건 검증, [1.3] 부피."""
    import numpy as np
    # [1.1] 각 θ 에 대해 k 를 sinθ 근 조건으로 결정, cosθ 도 근이고 f 의 실근이 정확히 {sinθ, cosθ} 인 θ 수집
    ths = np.linspace(1e-6, 2 * np.pi - 1e-6, 2000001)
    s_, c_ = np.sin(ths), np.cos(ths)
    ks = (4 * s_ - 2) / (s_**2 - s_ + 1)
    kc = (4 * c_ - 2) / (c_**2 - c_ + 1)
    diff = ks - kc
    cand = [ths[i] for i in range(len(ths) - 1) if diff[i] * diff[i + 1] <= 0 and abs(diff[i]) < 1e-2]
    good = []
    for th0 in cand:
        th = sp.nsolve(sp.Symbol("T") * 0 + (4 * sp.sin(x) - 2) / (sp.sin(x) ** 2 - sp.sin(x) + 1) - (4 * sp.cos(x) - 2) / (sp.cos(x) ** 2 - sp.cos(x) + 1), x, th0)
        th = float(th) % (2 * np.pi)
        s, c = np.sin(th), np.cos(th)
        k = (4 * s - 2) / (s * s - s + 1)
        if abs(k) < 1e-9:
            continue
        roots = np.roots([k, -(k + 4), k + 2])
        real = sorted(r.real for r in roots if abs(r.imag) < 1e-9)
        pts = sorted({round(s, 9), round(c, 9)})
        if len(real) >= 1 and sorted({round(r, 9) for r in real}) == pts and all(abs(th - g_) > 1e-6 for g_ in good):
            good.append(th)
    ans1 = load_answer(1, 0)
    ok1 = len(good) == 2 and abs(sum(good) - float(ans1)) < 1e-6
    # [1.2] f 가 조건 (가)(나) 만족 + 유일성(계수 직접 풀이)
    c0, c1, c2, c3 = sp.symbols("c0 c1 c2 c3")
    F = x**4 + c3 * x**3 + c2 * x**2 + c1 * x + c0
    sol = sp.solve([sp.Eq(sp.expand(sp.diff(F, x).subs(x, -x) + sp.diff(F, x)), 0).lhs.coeff(x, i) for i in range(0, 4)] + [sp.diff(F, x).subs(x, 2), F.subs(x, 2) + 13], [c0, c1, c2, c3], dict=True)
    Fs = F.subs(sol[0]) if sol else None
    ans2 = sp.sympify(load_answer(1, 1, raw=True), locals=SYM)
    minimum = sp.diff(Fs, x, 2).subs(x, 2) > 0
    ok2 = Fs is not None and sp.expand(Fs - ans2) == 0 and minimum
    # [1.3] V = ∫ (x+1)2^x
    V = sp.integrate((x + 1) * 2**x, (x, 0, 1))
    ans3 = load_answer(1, 2)
    ok3 = eq(sp.simplify(V), ans3) and abs(float(V) - float(ans3)) < 1e-12
    report(1, ok1 and ok2 and ok3, f"[1.1] θ={[round(g_/np.pi,4) for g_ in good]}π, 합={sum(good)/np.pi:.4f}π / [1.2] f={Fs} / [1.3] V={sp.simplify(V)}")


def p02():
    """[2.1] 둘레 최소 P 를 수치 최소화, [2.2] Q·tan, [2.3] t 를 수치로 탐색."""
    import numpy as np
    def geom(tv):
        A = np.array([tv, np.sqrt(tv)]); B = np.array([9 * tv, 3 * np.sqrt(tv)])
        fn = lambda X: np.hypot(X - A[0], A[1]) + np.hypot(X - B[0], B[1])   # 볼록 → 삼분탐색
        lo, hi = 0.0, 20 * tv
        for _ in range(200):
            m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
            if fn(m1) < fn(m2):
                hi = m2
            else:
                lo = m1
        return A, B, (lo + hi) / 2
    ans1 = sp.sympify(load_answer(2, 0, raw=True), locals=SYM)
    ok1 = all(abs(geom(tv)[2] - float(ans1.subs(t, tv))) < 1e-5 for tv in (0.1, 0.5, 1.3, 4.0))
    # [2.2] 접선 교점 Q 와 ∠PQO
    def tanPQO(tv):
        A, B, px = geom(tv)
        P = np.array([px, 0.0])
        mA, mB = 1 / (2 * np.sqrt(tv)), 1 / (2 * np.sqrt(9 * tv))
        # y = mA (x - A0) + A1 = mB (x - B0) + B1
        qx = ((B[1] - mB * B[0]) - (A[1] - mA * A[0])) / (mA - mB)
        Q = np.array([qx, mA * (qx - A[0]) + A[1]])
        u, v = P - Q, np.array([0.0, 0.0]) - Q
        cosv = u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
        return np.tan(np.arccos(cosv))
    ans2 = load_answer(2, 1)
    ok2 = all(abs(tanPQO(tv) - float(ans2.subs(t, tv))) < 1e-5 for tv in (0.1, 0.5, 1.3, 4.0))
    # [2.3] cosθ + 1/cosθ = -10/3 를 만족하는 t (θ = ∠APB, P 는 수치 최소점)
    def cosAPB(tv):
        A, B, px = geom(tv)
        P = np.array([px, 0.0])
        u, v = A - P, B - P
        return u @ v / (np.linalg.norm(u) * np.linalg.norm(v))
    grid = np.linspace(0.02, 5, 2000)
    h = np.array([cosAPB(g_) + 1 / cosAPB(g_) + 10 / 3 for g_ in grid])
    roots = [grid[i] for i in range(len(grid) - 1) if h[i] * h[i + 1] < 0 and abs(h[i]) < 5 and abs(h[i + 1]) < 5]
    ans3 = load_answer(2, 2)
    ok3 = len(roots) == 1 and abs(roots[0] - float(ans3)) < 5e-3 and abs(cosAPB(float(ans3)) + 1 / 3) < 1e-5
    report(2, ok1 and ok2 and ok3, f"[2.1] P 수치최소=3t / [2.2] tan∠PQO=(3/2)√t / [2.3] t 근={[round(float(r_),4) for r_ in roots]} (cosθ={cosAPB(float(ans3)):.6f})")


def p03():
    """[3.1] 만나는 원 전수, [3.2] 호의 길이 수치적분, [3.3] 급수 부분합."""
    import mpmath as mp
    mp.mp.dps = 30
    dist = lambda tv: mp.e ** (2 * tv)
    # 거리 범위 [1, e^{2π}] 에서 3^n 이 포함되는 n
    N = sum(1 for n_ in range(1, 40) if 1 <= 3**n_ <= float(dist(mp.pi)))
    ans1 = load_answer(3, 0)
    ok1 = N == ans1
    # [3.2] 마지막 원을 만나는 시각 t=ln(3^N)/2, 호의 길이를 속도 벡터 노름의 수치적분으로
    tN = mp.log(3**N) / 2
    X = lambda tv: mp.e ** (2 * tv) * mp.cos(tv)
    Y = lambda tv: mp.e ** (2 * tv) * mp.sin(tv)
    L = mp.quad(lambda tv: mp.sqrt(mp.diff(X, tv) ** 2 + mp.diff(Y, tv) ** 2), [0, tN / 2, tN])
    ans2 = load_answer(3, 1)
    ok2 = abs(L - sp.N(ans2, 30)) < mp.mpf("1e-15")
    # [3.3] 급수
    term = lambda k: (-mp.mpf(1) / 2) ** (k - 1) * mp.quad(lambda xv: mp.pi * mp.sin(mp.pi * xv), [k - 1, k])
    S = mp.nsum(term, [N, mp.inf])
    S2 = sum(term(k) for k in range(N, N + 80))
    ans3 = load_answer(3, 2)
    ok3 = abs(S2 - sp.N(ans3, 30)) < mp.mpf("1e-15")
    report(3, ok1 and ok2 and ok3, f"[3.1] N={N} / [3.2] L={mp.nstr(L, 12)} vs {sp.N(ans2, 12)} / [3.3] 급수 부분합={mp.nstr(S2, 12)}")


if __name__ == "__main__":
    for name in sorted(k for k in dict(globals()) if k.startswith("p") and k[1:].isdigit()):
        globals()[name]()
    passed = sum(RESULTS)
    print(f"SUMMARY G24A: {passed}/{len(RESULTS)} PASS")
    sys.exit(0 if passed == len(RESULTS) else 1)
