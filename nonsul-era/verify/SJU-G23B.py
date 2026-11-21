#!/usr/bin/env python3
"""SJU-G23B (세종대 2023학년도 수시 논술고사 B형) — sympy/mpmath 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import mpmath as mp
import numpy as np
import sympy as sp

PREFIX = "SJU-G23B"
WIDTH = 2
RESULTS = []
mp.mp.dps = 30


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    return dict(seg.strip().split(" = ", 1) for seg in s.split(";"))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    def f(x):  # 우함수·주기 4
        u = x % 4
        u = min(u, 4 - u)       # [0,2]
        return u**3 - 3 * u**2 + 4
    # (1) h(x)=2 인 x in (2,4) 를 격자+이분법으로
    xs = np.linspace(2.0000123, 3.9999, 19997)  # 격자가 근에 정확히 걸리지 않게
    hv = np.array([f(v) for v in xs])
    i = np.where(np.diff(np.sign(hv - 2)) != 0)[0]
    roots = []
    for k in i:
        lo, hi = xs[k], xs[k + 1]
        for _ in range(80):
            m = (lo + hi) / 2
            if (f(lo) - 2) * (f(m) - 2) <= 0:
                hi = m
            else:
                lo = m
        roots.append((lo + hi) / 2)
    ok1 = len(roots) == 1 and abs(roots[0] - float(sp.sympify(md_answer(1, "1")))) < 1e-10
    # (2) (h^-1)'(2) = 1/h'(r) 수치미분
    r = roots[0]
    hp = (f(r + 1e-6) - f(r - 1e-6)) / 2e-6
    ok2 = abs(1 / hp - float(sp.sympify(md_answer(1, "2")))) < 1e-6
    # (3) (h^-1)' 의 값 범위 = {1/h'(x)} : 최솟값
    hps = [(f(v + 1e-7) - f(v - 1e-7)) / 2e-7 for v in np.linspace(2.001, 3.999, 4001)]
    m_inv = min(1 / d for d in hps if d > 0)
    # |g(t)| >= m_inv 인 t 의 최댓값 (t 를 100 까지 촘촘히)
    ts = np.linspace(1e-3, 100, 2_000_001)
    u = ts % 4
    u = np.minimum(u, 4 - u)
    ft = u**3 - 3 * u**2 + 4
    good = np.abs((ft - 4) / ts) >= m_inv - 1e-12
    t0 = ts[good].max()
    n = int(np.floor(t0))
    ok3 = abs(m_inv - 1 / 3) < 1e-6 and n == int(md_answer(1, "3")) and n < t0 < n + 1
    report(1, ok1 and ok2 and ok3, f"h^-1(2)={r:.10f} (h^-1)'(2)={1/hp:.8f} min(h^-1)'={m_inv:.6f} t0~{t0:.5f}")


def p02():
    f = lambda X: mp.quad(lambda s: mp.tan(mp.log(s)), [mp.e, X])
    # (1) 격자에서 최소
    lo, hi = mp.exp(-mp.pi / 2), mp.exp(mp.pi / 2)
    grid = [lo + (hi - lo) * k / 400 for k in range(5, 396)]
    xm = min(grid, key=f)
    xm = mp.findroot(lambda X: mp.tan(mp.log(X)), xm)
    ok1 = abs(xm - mp.mpf(sp.sympify(md_answer(2, "1")))) < 1e-15
    # (2) 수치값 비교
    a = f(1)
    J = mp.quad(lambda X: mp.exp(X) * mp.tan(X)**2, [0, 1])
    expr = sp.sympify(md_answer(2, "2"), locals={"a": sp.Symbol("a"), "e": sp.E})
    val = mp.mpf(sp.N(expr.subs(sp.Symbol("a"), sp.Float(str(a), 30)), 30))
    ok2 = abs(J - val) < 1e-15
    # (3)
    I = mp.quad(lambda X: mp.tan(mp.log(X)) / X**2, [1 / mp.e, mp.e]) - f(1 / mp.e)
    ok3 = abs(I - mp.mpf(sp.sympify(md_answer(2, "3")))) < 1e-15
    report(2, ok1 and ok2 and ok3, f"argmin={float(xm)} a={float(a):.10f} J={float(J):.10f} I={float(I):.2e}")


def p03():
    s, a = sp.symbols("s a", positive=True)
    ans = parts(md_answer(3, "1"))
    loc = {"s": s, "a": a}
    # (1) 좌표로 직접: 둘레 위 점
    def P(sv, av):
        if sv <= 1: return (0, sv)
        if sv <= av + 1: return (sv - 1, 1)
        if sv <= av + 2: return (av, av + 2 - sv)
        return (2 * av + 2 - sv, 0)
    av = 3.7
    pieces = [("f1", 0, 1), ("f2", 1, av + 1), ("f3", av + 1, av + 2), ("f4", av + 2, 2 * av + 2)]
    ok1 = True
    for key, l, r in pieces:
        fe = sp.lambdify((s, a), sp.sympify(ans[key], locals=loc))
        for sv in np.linspace(l, r, 50):
            px, py = P(sv, av)
            ok1 &= abs(fe(sv, av) - np.hypot(px, py)) < 1e-12
    f2 = sp.sqrt((s - 1)**2 + 1)
    l2 = f2.subs(s, 2) + sp.diff(f2, s).subs(s, 2) * (s - 2)
    ok1 &= sp.simplify(l2 - sp.sympify(ans["l2"], locals=loc)) == 0
    # (2) 교점 개수: 여러 a, α, β 에서 직접 세기(조밀 격자 부호변화 + 접점)
    def F(sv, av):
        px, py = P(sv, av)
        return np.hypot(px, py)
    def fprime(tv, av, h=1e-7):
        return (F(tv + h, av) - F(tv - h, av)) / (2 * h)
    def count(tv, av):
        m = fprime(tv, av)
        S = np.linspace(0, 2 * av + 2, 400001)
        d = np.array([F(x, av) for x in S[::1]]) - (F(tv, av) + m * (S - tv))
        sgn = np.sign(np.where(np.abs(d) < 1e-9, 0, d))
        cnt = 0
        # 0 근방(접점·꼭짓점) 구간을 하나로 묶어 셈
        i = 0
        n = len(S)
        while i < n - 1:
            if sgn[i] == 0:
                j = i
                while j < n and sgn[j] == 0:
                    j += 1
                cnt += 1
                i = j
                continue
            if sgn[i] * sgn[i + 1] < 0:
                cnt += 1
            i += 1
        return cnt
    exp2 = parts(md_answer(3, "2"))
    res = []
    for av in (2.0, 3.5):
        for al in (1.3, 2.0, 2.4, av + 0.8):
            want = int(exp2["alpha_in_(1,2]"]) if al <= 2 else int(exp2["alpha_in_(2,a+1)"])
            res.append(count(al, av) == want)
        for be in (av + 1.2, av + 1.5, av + 1.9):
            res.append(count(be, av) == int(exp2["beta"]))
    ok2 = all(res)
    # (3) 실제 예각 집합의 inf·sup 를 격자로, a 크게
    exp3 = parts(md_answer(3, "3"))
    def bounds(av):
        al = np.linspace(1 + 1e-6, av + 1 - 1e-6, 2001)
        be = np.linspace(av + 1 + 1e-6, av + 2 - 1e-6, 2001)
        m1 = (al - 1) / np.sqrt((al - 1)**2 + 1)
        m2 = (be - av - 2) / np.sqrt((be - av - 2)**2 + av**2)
        ang = np.abs(np.arctan(m1)[:, None] - np.arctan(m2)[None, :])
        ang = np.minimum(ang, np.pi - ang)
        return ang.min(), ang.max()
    Ls, Rs = zip(*(bounds(av) for av in (10, 100, 1000, 10000)))
    ok3 = Ls[-1] < 1e-4 and abs(Rs[-1] - float(sp.sympify(exp3["limR"]))) < 1e-3 and float(sp.sympify(exp3["limL"])) == 0
    report(3, ok1 and ok2 and ok3, f"f pieces/l2 ok={ok1}; counts {res}; L(a)->{Ls[-1]:.2e}, R(a): {[round(r, 5) for r in Rs]}")


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
