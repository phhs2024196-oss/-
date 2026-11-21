#!/usr/bin/env python3
"""DGU-G22 (2022학년도 동국대 논술 실전) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "DGU-G22"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """함수방정식 f(x+y)-f(x-y)=2f(y)f'(x): f'(0), f''(0), ∫0^2 f (답 1, 0, 2a^2).
    해 족(f=x, sin kx/k, sinh kx/k)마다 방정식 성립을 확인하고 세 값을 재계산."""
    y = sp.symbols("y", real=True)
    k = sp.symbols("k", positive=True)
    fams = [x, sp.sin(k * x) / k, sp.sinh(k * x) / k]
    ok = True
    msgs = []
    for f in fams:
        eq = sp.simplify(sp.expand_trig(f.subs(x, x + y) - f.subs(x, x - y) - 2 * f.subs(x, y) * sp.diff(f, x)))
        a = f.subs(x, 1)
        d1 = sp.simplify(sp.diff(f, x).subs(x, 0))
        d2 = sp.simplify(sp.diff(f, x, 2).subs(x, 0))
        I = sp.simplify(sp.integrate(f, (x, 0, 2)))
        good = eq == 0 and d1 == 1 and d2 == 0 and sp.simplify(I - 2 * a**2) == 0
        ok &= bool(good)
        msgs.append(f"{f}: f'(0)={d1}, f''(0)={d2}, ∫=2a^2 {sp.simplify(I-2*a**2)==0}")
    # 해설 논리: f'(x)=f'(0)f'(x) 에서 f'≡0 이면 f≡0 → f(1)=a>0 모순 → f'(0)=1
    report(1, ok, " | ".join(msgs))


def _sim(a, nmeet=5, T=200):
    """P·Q 충돌(속도 교환)·벽(±1 반사) 사건 시뮬레이션, 유리수 정확 계산."""
    from fractions import Fraction as Fr
    t = Fr(0); p = Fr(1); q = Fr(-1); vp = Fr(-1, 9); vq = Fr(a)
    meets, after = [], []
    while len(meets) < nmeet and t < T:
        cands = []
        if vp != vq:
            dt = (q - p) / (vp - vq)
            if dt > 0:
                cands.append((dt, "m"))
        for who, pos, v in (("p", p, vp), ("q", q, vq)):
            for b in (Fr(1), Fr(-1)):
                if v != 0:
                    dt = (b - pos) / v
                    if dt > 0:
                        cands.append((dt, "b" + who))
        dt = min(c[0] for c in cands)
        evs = [c[1] for c in cands if c[0] == dt]
        t += dt; p += vp * dt; q += vq * dt
        if "m" in evs:
            vp, vq = vq, vp
            meets.append(t); after.append(vp)
        if "bp" in evs:
            vp = -vp
        if "bq" in evs:
            vq = -vq
    return meets, after


def p02():
    """두 점 충돌·반사: t=9에서 5번째 만남인 a, 4번째 만남 시각, 그 직후 P의 속도 (답 1, 27/4, -1/9)."""
    from fractions import Fraction as Fr
    grid = [Fr(i, 100) for i in range(1, 1001)]
    hits = [a for a in grid if _sim(a)[0][4] == 9]
    t5 = [_sim(a)[0][4] for a in grid]
    # a<1 이면 t5>9, a>1 이면 t5<9 → t5=9 인 a 는 1 하나뿐
    mono = all((T > 9) if a < 1 else (T < 9) if a > 1 else True for a, T in zip(grid, t5))
    meets, after = _sim(1)
    ok = hits == [1] and mono and meets[3] == Fr(27, 4) and after[3] == Fr(-1, 9)
    report(2, ok, f"a 후보={hits}, a<1→t5>9·a>1→t5<9 {mono}, 만남 시각={[str(m) for m in meets]}, 4번째 직후 P 속도={after[3]}")


def p03():
    """물통 수면 상승 속도 = Sv/(A-S). 구체 단면 S(z)=z^2+1 로 부피 보존식을 수치로 풀어 차분 비교."""
    import mpmath as mp
    mp.mp.dps = 30
    A, v = mp.mpf(10), mp.mpf("0.7")
    Sf = lambda z: z**2 + 1
    Vsub = lambda z: z**3 / 3 + z
    # 잠긴 깊이 z(t): A(z - v t) = Vsub(z)  (수면 상승량 = z - v t)
    zt = lambda tt: mp.findroot(lambda z: A * (z - v * tt) - Vsub(z), v * tt)
    ok = True
    errs = []
    for tt in [mp.mpf("0.5"), mp.mpf("1.3"), mp.mpf("2.2")]:
        hdot = mp.diff(lambda s: zt(s) - v * s, tt)
        S = Sf(zt(tt))
        pred = S * v / (A - S)
        errs.append(float(abs(hdot - pred) / pred))
        ok &= abs(hdot - pred) / pred < 1e-9
    # 기호식: z' = Av/(A-S(z)) 에서 h' = z' - v
    Ss, As, vs = sp.symbols("S A v", positive=True)
    sym = sp.simplify(As * vs / (As - Ss) - vs - Ss * vs / (As - Ss)) == 0
    report(3, ok and sym, f"수치 상대오차 {['%.1e' % e for e in errs]}, h'=Av/(A-S)-v=Sv/(A-S) {sym}")


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
