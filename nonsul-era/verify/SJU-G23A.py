#!/usr/bin/env python3
"""SJU-G23A (세종대 2023학년도 수시 논술고사 A형) — sympy/mpmath 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import mpmath as mp
import numpy as np
import sympy as sp

PREFIX = "SJU-G23A"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)
mp.mp.dps = 30


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    return dict(seg.split("=", 1) for seg in s.replace(" ", "").split(";"))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    g = (t**2 + 2) * sp.exp(2 * t) + sp.exp(4 * t)
    gp = sp.diff(g, t)
    # (1) g'>0 (격자) + g(1), g(0)  [증명-수치확인]
    gpl = sp.lambdify(t, gp)
    ok1 = all(gpl(v) > 0 for v in np.linspace(-30, 5, 3501)) and \
        sp.simplify(g.subs(t, 1) - (3 * sp.E**2 + sp.E**4)) == 0 and g.subs(t, 0) == 3
    # (2) x f'(x) = g(t)/g'(t), x->0+ <=> t->-oo ; 수치로도 f 를 역으로 구해 확인
    lim = sp.limit(g / gp, t, -sp.oo)
    gm = sp.lambdify(t, g, "mpmath"); gpm = sp.lambdify(t, gp, "mpmath")
    def finv(X):  # g 는 증가: 이분법으로 역함수값
        lo, hi = mp.mpf(-3000), mp.mpf(5)
        for _ in range(300):
            mid = (lo + hi) / 2
            if gm(mid) > X:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2
    xs = mp.mpf("1e-400")
    num = xs / gpm(finv(xs))
    ok2 = sp.simplify(lim - sp.sympify(md_answer(1, "2"))) == 0 and abs(num - 0.5) < 0.005
    # (3) 직접 수치적분(역함수를 수치로) vs 치환식
    I_sym = sp.integrate(t * gp, (t, 0, 1))
    I_num = mp.quad(lambda X: finv(X), [3, gm(1)])
    md3 = sp.sympify(md_answer(1, "3"))
    ok3 = sp.simplify(I_sym - md3) == 0 and abs(I_num - float(md3)) < 1e-9
    report(1, ok1 and ok2 and ok3, f"[증명-수치확인] g'>0; lim={lim} (x=1e-400: {float(num):.4f}); I={sp.nsimplify(sp.expand(I_sym))}~{float(I_num):.10f}")


def p02():
    # y=f(x): u=x+y>0, u+ln u=2x
    def fs(X):  # 안정적 근 찾기: 이분법
        lo, hi = mp.mpf("1e-300"), mp.mpf(10) + abs(4 * X)
        for _ in range(400):
            mid = (lo + hi) / 2
            if mid + mp.log(mid) - 2 * X > 0:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2 - X
    x0 = (mp.e + 1) / 2
    ok_pt = abs(fs(x0) - (mp.e - 1) / 2) < 1e-20
    slope = mp.diff(fs, x0)
    ok1 = ok_pt and abs(slope - mp.mpf(sp.N(sp.sympify(md_answer(2, "1"), locals={"e": sp.E}), 30))) < 1e-15
    # (2) 최솟값: 격자 + 정밀화
    grid = [mp.mpf(i) / 100 for i in range(-300, 301)]
    x_min = min(grid, key=fs)
    xm = mp.findroot(lambda X: mp.diff(fs, X), x_min)
    ok2 = abs(fs(xm) - mp.mpf(sp.Rational(md_answer(2, "2")))) < 1e-15 and \
        all(fs(v) >= fs(xm) for v in grid)
    # (3) f(0)=f(alpha)=beta, alpha!=0 를 직접 찾음
    beta = fs(0)
    alpha = mp.findroot(lambda X: fs(X) - beta, 1.3)
    rhs = sp.sympify(md_answer(2, "3").split("=", 1)[1], locals={"alpha": sp.Symbol("alpha")})
    ok3 = abs(alpha) > 1e-6 and abs(mp.mpf(sp.N(rhs.subs("alpha", sp.Float(str(alpha), 30)), 30)) - beta) < 1e-15 and alpha < 2
    # 다른 근이 (x<1/2 쪽) 없는지: f 는 x<1/2 감소
    ok3 = ok3 and all(fs(grid[i]) > fs(grid[i + 1]) for i in range(len(grid) - 1) if grid[i + 1] < 0.5)
    report(2, ok1 and ok2 and ok3, f"slope={float(slope):.10f} min at x={float(xm):.10f} f={float(fs(xm)):.10f} alpha={float(alpha):.6f} beta={float(beta):.6f}")


def p03():
    # f(x)=(x-a)^2(x-b)+7 : 극댓값 7 을 x=a 에서 갖는 최고차계수 1 삼차함수 전체(a<b)
    T = np.linspace(-1, 6, 7001)
    S = np.linspace(0, 1, 401)

    def gfun(a, b, ts):
        xx = ts[:, None] + S[None, :]
        return ((xx - a)**2 * (xx - b) + 7).max(axis=1)

    def corner_at(a, b, t0, h=1e-4):
        gl = gfun(a, b, np.array([t0 - 2 * h, t0 - h, t0]))
        gr = gfun(a, b, np.array([t0, t0 + h, t0 + 2 * h]))
        return abs((gl[2] - gl[1]) / h - (gr[1] - gr[0]) / h) > 1e-2

    feas = []
    for a in np.linspace(-0.5, 2.5, 3001):
        # (가): 꺾인 점(f(γ)=f(γ+1), γ>a) 이 3 이 되도록 b 를 수치로 맞춤 — 그 다음 실제 g 로 확인
        fb = lambda b: ((3 - a)**2 * (3 - b) + 7) - ((4 - a)**2 * (4 - b) + 7)
        lo, hi = a + 1e-6, 50.0
        if fb(lo) * fb(hi) > 0:
            continue
        for _ in range(80):
            mid = (lo + hi) / 2
            if fb(lo) * fb(mid) <= 0:
                hi = mid
            else:
                lo = mid
        b = (lo + hi) / 2
        ts = np.linspace(0, 0.5, 51)
        if np.all(np.abs(gfun(a, b, ts) - 7) < 1e-6) and corner_at(a, b, 3.0):
            feas.append((a, b))
    amin, amax = min(f[0] for f in feas), max(f[0] for f in feas)
    md2 = parts(md_answer(3, "2"))
    ok2 = abs(amin - float(sp.sympify(md2["min"]))) < 2e-3 and abs(amax - float(sp.sympify(md2["max"]))) < 2e-3
    # (1) 가능한 모든 f 에서 f(3)-f(4)
    d34 = max(abs(((3 - a)**2 * (3 - b)) - ((4 - a)**2 * (4 - b))) for a, b in feas)
    ok1 = d34 < 1e-6 and abs(float(sp.sympify(md_answer(3, "1")))) < 1e-12
    # (3) g(4)=7 인 것: 실제 g 로 계산해 가장 가까운 것 → 정밀화(이분법)
    def g4(a, b):
        return gfun(a, b, np.array([4.0]))[0] - 7
    cand = [(abs(g4(a, b)), a, b) for a, b in feas]
    _, a3, b3 = min(cand)
    f3 = (3 - a3)**2 * (3 - b3) + 7
    ok3 = abs(f3 - float(sp.sympify(md_answer(3, "3")))) < 2e-2
    report(3, ok1 and ok2 and ok3, f"feasible a in [{amin:.4f},{amax:.4f}] (n={len(feas)}), max|f(3)-f(4)|={d34:.1e}, g(4)=7 -> a~{a3:.4f}, b~{b3:.4f}, f(3)~{f3:.4f}")


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
