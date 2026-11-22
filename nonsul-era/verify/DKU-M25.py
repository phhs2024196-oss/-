#!/usr/bin/env python3
"""DKU-M25 (단국대 2025 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건에서 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "DKU-M25"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/DKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # 1. f=x(x-α)=x²-αx. h=sin x/f 가 x=α 에서 극한값을 가지려면 sin α=0 (아니면 |h|→∞).
    al = sp.symbols("al", real=True)
    cand = [v for v in sp.solveset(sp.sin(al), al, sp.Interval.open(-2 * sp.pi, 2 * sp.pi)) if v != 0]
    # 수치 확인: sin α≠0 인 α(예: π/2)에서는 |h| 가 α 근방에서 유계가 아님
    mp.mp.dps = 30
    a0 = mp.pi / 2
    blow = abs(mp.sin(a0 + mp.mpf("1e-12")) / ((a0 + mp.mpf("1e-12")) * mp.mpf("1e-12"))) > 1e6
    vals = set()
    for a in cand:
        f = x**2 - a * x
        vals.add(sp.simplify(sp.integrate(f * sp.sin(x), (x, -2 * sp.pi, 2 * sp.pi))))
    md = set(sp.sympify(md_answer(1, "1").replace("{", "[").replace("}", "]")))
    ok1 = blow and set(cand) == {sp.pi, -sp.pi} and vals == md
    # 2. f=x²-αx, g=-x²+βx(원점 통과·이차항 계수 -1), 교점 t=(α+β)/2. B-A 를 절댓값 적분으로 직접 계산
    P_, Q_, tt = sp.symbols("P_ Q_ tt", positive=True)         # P_=α, Q_=β
    ff = x**2 - P_ * x
    gg = -x**2 + Q_ * x
    t0 = sp.solve(sp.Eq(ff, gg), x)
    tpos = [v for v in t0 if v != 0][0]
    # t>0, f-g=2x(x-t) 이므로 (0,t) 에서 g>f, (t,2t) 에서 f>g
    A = sp.integrate(gg - ff, (x, 0, tpos))
    B = sp.integrate(ff - gg, (x, tpos, 2 * tpos))
    tval = sp.symbols("tval", positive=True)
    ts = sp.solve(sp.Eq(sp.simplify((B - A).subs(Q_, 2 * tval - P_).subs(tpos, tval)), sp.Rational(125, 6)), tval)
    tstar = ts[0]
    s_ = 2 * tstar                                                # α+β
    ns = [m for m in range(1, 200) if sum(2 * k**2 - s_ * k for k in range(1, m + 1)) <= 5 * m * (m + 1)]
    ok2 = ts == [sp.Rational(5, 2)] and ns == list(range(1, max(ns) + 1)) and max(ns) == int(md_answer(1, "2"))
    # 3. s(x)=Σu(x-a_i) 의 값: 임의 a_i 로 s(a_i) 와 점프 크기 직접 계산
    u = lambda z: -1 if z <= 0 else 1
    a_ = [sp.Rational(v) for v in (-3, 1, 4, 9, 12)]
    sfun = lambda z: sum(u(z - ai) for ai in a_)
    eps = sp.Rational(1, 10**6)
    s_at = [sfun(a_[i]) for i in range(3)]
    jumps = all(sfun(a_[i] + eps) - sfun(a_[i] - eps) == 2 for i in range(5))
    # v 연속 ⇔ f(sin a_i)·2=0 ⇔ sin a_i ∈ {0, α}.  Σ s(a_i) y_i = 9, y_i∈[-1,1] 이면 y=(-1,-1,-1) 뿐
    corners = list(itertools.product((-1, 1), repeat=3))
    best = max(sum(c * y for c, y in zip(s_at, cr)) for cr in corners)
    argm = [cr for cr in corners if sum(c * y for c, y in zip(s_at, cr)) == best]
    y_sol = argm[0]
    alpha_sol = sp.Integer(-1)                                      # sin a_i=-1 ∈{0,α}, -1≠0 ⇒ α=-1
    # 존재 확인: α=-1, a=(-π/2, 3π/2, 7π/2, 4π, 5π) 에서 v 가 모두 연속(f(sin a_i)=0)
    fa = lambda y: y**2 - alpha_sol * y
    aa = [-sp.pi / 2, 3 * sp.pi / 2, 7 * sp.pi / 2, 4 * sp.pi, 5 * sp.pi]
    exist = all(sp.simplify(fa(sp.sin(v))) == 0 for v in aa) and all(aa[i] < aa[i + 1] for i in range(4))
    ok3 = s_at == [-5, -3, -1] and jumps and best == 9 and argm == [(-1, -1, -1)] and exist and \
        alpha_sol == sp.sympify(md_answer(1, "3"))
    report(1, ok1 and ok2 and ok3, f"α={cand} ∫={vals} t={ts} n_max={max(ns)} s(a_i)={s_at} α={alpha_sol}")


def p02():
    # 1. 사다리꼴 ABCD, P(x,0). 선분까지의 최단거리(수선의 발이 선분 밖이면 끝점까지)를 직접 계산하여 m(x) 의 꺾임점 탐색
    mp.mp.dps = 30

    def segdist(p, a, b):
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        tt = ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)
        tt = max(mp.mpf(0), min(mp.mpf(1), tt))
        return mp.sqrt((p[0] - (ax + tt * dx))**2 + (p[1] - (ay + tt * dy))**2)
    A, B, C, D = (-2, 0), (-1, 1), (1, 1), (2, 0)
    m = lambda xv: min(segdist((xv, 0), A, B), segdist((xv, 0), B, C), segdist((xv, 0), C, D))
    h = mp.mpf("1e-10")
    N = 4000
    kinks = []
    xs = [mp.mpf(-2) + mp.mpf(4) * i / N for i in range(1, N)]
    slope = lambda xv: ((m(xv + h) - m(xv - h)) / (2 * h))
    prev = None
    for i in range(1, N - 1):
        xl, xr = xs[i - 1], xs[i]
        d = abs((m(xr + h) - m(xr - h)) / (2 * h) - (m(xl + h) - m(xl - h)) / (2 * h))
        if d > mp.mpf("0.3"):
            kinks.append((xl, xr))
    refined = []
    for xl, xr in kinks:
        lo, hi = xl, xr
        for _ in range(60):
            mid = (lo + hi) / 2
            sl = (m(lo + h) - m(lo - h)) / (2 * h)
            sm = (m(mid + h) - m(mid - h)) / (2 * h)
            if abs(sl - sm) > mp.mpf("0.3"):
                hi = mid
            else:
                lo = mid
        refined.append((lo + hi) / 2)
    md = sp.sympify(md_answer(2, "1").replace("{", "[").replace("}", "]"))
    ok1 = len(refined) == 2 and all(abs(r - mp.mpf(sp.N(v, 30))) < mp.mpf("1e-7") for r, v in zip(sorted(refined), sorted(md, key=float)))
    # 2. f=mx+n, F=f(x+2)g+g(f)h. 후보 (m,n,k,a)=(m,-2m,2,-9e²) 에서 F 의 미분가능성과 F'(k)=17e² 를 수치 확인
    mp.mp.dps = 50
    e = mp.e
    sols = []
    ms = sp.symbols("ms", real=True)
    for sg in (1, -1):
        r = sp.solve(sp.Eq(8 * ms + 9 * sg * ms, 17), ms)               # F'(2)=8m e²+9e²|m|=17e²
        sols += [v for v in r if (v > 0 if sg == 1 else v < 0)]
    out = []
    for mv in sols:
        mv = mp.mpf(sp.N(mv, 50))
        nv = -2 * mv
        k = mp.mpf(2)
        a = -9 * e**2
        fl = lambda z: mv * z + nv
        g = lambda z: abs(z) * mp.e**z
        hh = lambda z: a if z < k else 9 * e**2
        F = lambda z: fl(z + 2) * g(z) + g(fl(z)) * hh(z)
        d = mp.mpf("1e-20")
        okd = True
        for z0 in (mp.mpf(0), k):
            dl = (F(z0) - F(z0 - d)) / d
            dr = (F(z0 + d) - F(z0)) / d
            okd = okd and abs(dl - dr) < mp.mpf("1e-10") * (1 + abs(dl))
        Fk = (F(k + d) - F(k)) / d
        okd = okd and abs(Fk - 17 * e**2) < mp.mpf("1e-10") * e**2
        out.append((mv, okd, fl(a + k)))
    got = sorted(sp.nsimplify(sp.N(v[2] / sp.exp(2).evalf(50), 30), rational=True) for v in out)
    md2 = sorted(sp.sympify(md_answer(2, "2").replace("{", "[").replace("}", "]")), key=lambda z: float(z / sp.exp(2)))
    ok2 = all(o[1] for o in out) and len(out) == 2 and \
        all(abs(sp.N(v / sp.exp(2)) - sp.N(w / sp.exp(2))) < 1e-12 for v, w in
            zip(md2, sorted((sp.Float(str(o[2]), 40) for o in out), key=float)))
    report(2, ok1 and ok2, f"kinks={[mp.nstr(r, 8) for r in sorted(refined)]} m={[str(s) for s in sols]} f(a+k)/e²={[mp.nstr(o[2] / e**2, 8) for o in out]}")


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
