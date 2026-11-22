#!/usr/bin/env python3
"""CAU-G23A (2023학년도 중앙대 수시 논술 자연계열Ⅰ 수학) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "CAU-G23A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """A,B,C 이름표 게임 전수 시뮬레이션(6라운드, 종료 없음, 오직 1명만 자기 이름표) → 189."""
    import itertools
    names = "ABC"

    def rounds(state, r, hits):
        # state: 아직 못 뽑은 사람 집합(이 사람들의 이름표가 상자에 있음)
        if r > 6:
            yield hits
            return
        people = sorted(state)
        for perm in itertools.permutations(people):      # 사람 i 가 perm[i] 이름표를 뽑음
            got = {p for p, q in zip(people, perm) if p == q}
            rest = set(people) - got
            if not rest:                                   # 상자가 비면 종료 → 6라운드까지 진행 안 됨
                continue
            yield from rounds(rest, r + 1, hits + len(got))

    total = sum(1 for h in rounds(set(names), 1, 0) if h == 1)
    # 라운드별 M_n 재계산
    report(1, total == 3 * (2 ** 6 - 1) and total == 189, f"전수 {total}")


def p02():
    """(1) y'(0)=α/10, 접선이 (5,1) 지남 → α / (2) f(x)+2f(x+π/2)=15|sinx|/(2+cosx), 주기 2π → ∫_0^π f."""
    a = sp.symbols("alpha", positive=True)
    y = sp.cbrt(a + x / 6) * sp.cbrt((a + x / 24) ** 2) * (a + x / 60)
    dy0 = sp.simplify(sp.diff(y, x).subs(x, 0))
    sol = [s for s in sp.solve(sp.Eq(1, a ** 2 + dy0 * 5), a) if s.is_positive]
    ok1 = sp.simplify(dy0 - a / 10) == 0 and len(sol) == 1 and sp.simplify(sol[0] - (-1 + sp.sqrt(17)) / 4) == 0
    # (2) 푸리에 없이: f = -(1/15)(g - 2g(x+π/2) + 4g(x+π) - 8g(x+3π/2)) 가 식을 만족하는지 수치 확인 후 적분
    import mpmath as mp
    mp.mp.dps = 30
    g = lambda u: 15 * abs(mp.sin(u)) / (2 + mp.cos(u))
    F = lambda u: -(g(u) - 2 * g(u + mp.pi / 2) + 4 * g(u + mp.pi) - 8 * g(u + 3 * mp.pi / 2)) / 15
    chk = max(abs(F(u) + 2 * F(u + mp.pi / 2) - g(u)) for u in [0.1 * k for k in range(70)])
    per = max(abs(F(u + 2 * mp.pi) - F(u)) for u in [0.37 * k for k in range(20)])
    I = mp.quad(F, [0, mp.pi / 2, mp.pi])
    target = 11 * mp.log(3) - 12 * mp.log(2)
    # 유일성: 연산자 f -> f + 2 f(.+π/2) 는 주기 2π 함수에서 (1+2T), T^4=I 이므로 T 의 고유값 ±1,±i → 1+2λ≠0 → 가역
    ok2 = chk < 1e-20 and per < 1e-20 and abs(I - target) < 1e-20
    report(2, ok1 and ok2, f"y'(0)={dy0}, α={sol}; 함수식 잔차 {mp.nstr(chk, 3)}, ∫={mp.nstr(I, 15)} (11ln3-12ln2={mp.nstr(target, 15)})")


def p03():
    """(1) 원점에서 y=sin(lnx)(x≥1)에 그은 접선의 접점 a_n → Σ_{n=1}^{10} 1/(ln a_n ln a_{n+1}) / (2) α_0, x_0, ∫."""
    import mpmath as mp
    mp.mp.dps = 30
    # 접점 (t, f(t))의 접선이 원점을 지남: f(t) - t f'(t) = 0, u=ln t ≥ 0 으로 바꿔 근을 수치로 찾음
    tt = sp.symbols("tt", positive=True)
    fx = sp.sin(sp.log(tt))
    cond = sp.simplify((fx - tt * sp.diff(fx, tt)).subs(tt, sp.exp(sp.Symbol("uu"))))
    h = sp.lambdify(sp.Symbol("uu"), cond, "mpmath")
    us, k = [], 0
    grid = [mp.mpf(j) / 50 for j in range(0, 50 * 40)]
    for u0, u1 in zip(grid, grid[1:]):
        if h(u0) == 0 or h(u0) * h(u1) < 0:
            us.append(mp.findroot(h, (u0, u1), solver="bisect"))
        if len(us) == 11:
            break
    S = sum(1 / (us[i] * us[i + 1]) for i in range(10))
    ok1 = abs(S - mp.mpf(160) / (41 * mp.pi ** 2)) < 1e-20
    # (2) y=2e^{3x}-3αe^{2x}+8 가 x축과 한 점에서 만남 → 극솟값 0 (u=e^x>0)
    al, u = sp.symbols("alpha u", positive=True)
    cand = [s for s in sp.solve(sp.Eq(2 * al ** 3 - 3 * al ** 3 + 8, 0), al) if s.is_positive]
    a0 = cand[0]
    # 한 점에서 만남 확인: 2u^3-3a0 u^2+8 의 양근 개수
    pr = sp.Poly(2 * u ** 3 - 3 * a0 * u ** 2 + 8, u)
    pos = {r for r in sp.roots(pr).keys() if r.is_positive}
    x0 = sp.log(list(pos)[0])
    I = sp.simplify(sp.integrate(2 * sp.exp(3 * x) - 3 * a0 * sp.exp(2 * x) + 8, (x, 0, x0)))
    ok2 = a0 == 2 and len(pos) == 1 and sp.simplify(I - (8 * sp.log(2) - sp.Rational(13, 3))) == 0
    report(3, ok1 and ok2, f"합={mp.nstr(S, 15)} (160/(41π²)={mp.nstr(mp.mpf(160)/(41*mp.pi**2), 15)}); α0={a0}, x0={x0}, ∫={I}")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda s: int(s[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    m = re.search(r"-s(\d+)\.py$", __file__)
    label = PREFIX + (f"-s{m.group(1)}" if m else "")
    print(f"SUMMARY {label}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
