#!/usr/bin/env python3
"""SMWU-G23 (숙명여대 2023학년도 논술우수자전형 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import math
import random
import re
import sys

import sympy as sp

PREFIX = "SMWU-G23"
WIDTH = 2
RESULTS = []
x, u, n = sp.symbols("x u n", positive=True)


def md_answer(no: int, sub: str):
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SMWU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    return dict(seg.split("=", 1) for seg in s.replace(" ", "").split(";"))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # (1-1) f'(1)=c 미지수로 두고 f'(x+1)=f(x)+(x+1)f'(x) 로 f'(2..4) 계산 → c 와 무관한 값
    c = sp.symbols("c")
    f = {1: sp.Integer(1)}
    for m in (2, 3, 4):
        f[m] = m * f[m - 1]
    fp = {1: c}
    for m in (1, 2, 3):
        fp[m + 1] = f[m] + (m + 1) * fp[m]
    v = sp.expand(fp[4] - 9 * fp[2] - 6 * fp[1])
    ok_sym = v.free_symbols == set() and v == sp.sympify(md_answer(1, "1"))
    # 실제 해 Γ(x+1) 와 주기 인자 Γ(x+1)(1+0.1 sin 2πx) 둘 다로 수치 확인
    import mpmath as mp
    mp.mp.dps = 30
    sols = [lambda t: mp.gamma(t + 1), lambda t: mp.gamma(t + 1) * (1 + mp.mpf("0.1") * mp.sin(2 * mp.pi * t))]
    ok_num = all(abs(mp.diff(F, 4) - 9 * mp.diff(F, 2) - 6 * mp.diff(F, 1) - int(v)) < 1e-15 for F in sols)
    ok1 = ok_sym and ok_num
    # (1-2) g(u)=f(u)/f(u-1): 정수 u=2..8 기호 적분 + 비정수 u 는 베타함수 수치
    P = parts(md_answer(1, "2"))
    g_md = sp.sympify(P["g(u)"], locals={"u": u})
    fu = lambda uv: sp.integrate(x**uv * (1 - x) ** 10, (x, 0, 1))
    ok_g = all(sp.simplify(fu(m) / fu(m - 1) - g_md.subs(u, m)) == 0 for m in range(2, 9))
    fb = lambda uv: mp.beta(uv + 1, 11)
    ok_g2 = all(abs(fb(t) / fb(t - 1) - float(g_md.subs(u, t))) < 1e-14 for t in (2.3, 3.7, 6.25))
    # f(u+1) = f(u-1)/12 의 해를 u>=2 에서 수치로 탐색(격자 + findroot) — 단조이므로 유일
    h = lambda t: fb(t + 1) - fb(t - 1) / 12
    grid = [2 + i / 100 for i in range(0, 2000)]
    roots = [mp.findroot(h, (a, a + 0.01), solver="bisect") for a in grid if h(a) * h(a + 0.01) < 0]
    ok_u = len(roots) == 1 and abs(roots[0] - float(sp.sympify(P["u"]))) < 1e-12
    ok2 = ok_g and ok_g2 and ok_u
    report(1, ok1 and ok2, f"f'(4)-9f'(2)-6f'(1)={v}; g(u) ok={ok_g and ok_g2}; root u={mp.nstr(roots[0], 12) if roots else None}")


def step_geom(f, fp, xn, d):
    """제시문 <나>의 작도를 그대로 수행: P_n 에서의 법선과 y=y_n+d 의 교점 Q_n 의 x좌표."""
    X = sp.Dummy("Xq")
    yn = f(xn)
    normal = yn - (X - xn) / fp(xn)          # 법선 y = y_n - (X-x_n)/f'(x_n)
    return sp.solve(sp.Eq(normal, yn + d), X)[0]


def p02():
    # (2-1) f=3x^2, P1(1,3), d=1/5 : 작도로 x_2..x_8 을 구해 md 의 점화식과 비교
    P = parts(md_answer(2, "1"))
    xn_ = sp.Symbol("x_n")
    rhs = sp.sympify(P["x_(n+1)"], locals={"x_n": xn_})
    f1 = lambda t: 3 * t**2
    fp1 = lambda t: 6 * t
    d = sp.Rational(1, 5)
    xs = [sp.sympify(P["x_1"])]
    ok1 = xs[0] == 1 and f1(1) == 3
    for _ in range(7):
        nxt = step_geom(f1, fp1, xs[-1], d)
        ok1 = ok1 and sp.simplify(nxt - rhs.subs(xn_, xs[-1])) == 0
        xs.append(nxt)
    # (2-2) f=3(x^2-x), P1(1,0): d 를 격자로 훑어 작도(부동소수) 반복 1500회 수렴 여부 → 수렴 집합의 경계
    f2 = lambda t: 3 * (t**2 - t)
    fp2 = lambda t: 3 * (2 * t - 1)
    ok_rec = True
    Xs = sp.symbols("X")
    dd = sp.symbols("d", positive=True)
    gen = sp.expand(step_geom(f2, fp2, Xs, dd))           # 일반 d 에서의 한 단계
    ok_rec = sp.simplify(gen - ((1 - 6 * dd) * Xs + 3 * dd)) == 0
    def converges(dv):
        t = 1.0
        for _ in range(20000):
            fpv = 3 * (2 * t - 1)
            if fpv == 0:
                return True
            t = t - fpv * dv
            if abs(t) > 1e6:
                return False
        return abs(t - 0.5) < 1e-6
    ds = [i / 2000 for i in range(10, 1200)]          # d = 0.005 … 0.6
    conv = [dv for dv in ds if converges(dv)]
    lo, hi = min(conv), max(conv)
    contiguous = all(converges(dv) for dv in ds if lo <= dv <= hi)
    iv = sp.sympify(md_answer(2, "2"), locals={"Interval": sp.Interval})
    ok2 = ok_rec and contiguous and abs(lo - ds[0]) < 1e-12 and abs(hi - float(iv.end)) < 5e-4 and \
        iv.start == 0 and iv.left_open and iv.right_open and not converges(1 / 3)
    report(2, ok1 and ok2, f"x_2..x_4={xs[1:4]}; step={gen}; convergent d in [{lo},{hi}] (d=1/3 diverge)")


def p03():
    # (3-1) 무작위 내접 n각형(중심 내부) 넓이: 신발끈 공식 == 1/2 Σ sin2θ_i, 그리고 ≤ n/2 sin(2π/n)
    random.seed(3)
    nn = sp.symbols("n", positive=True, integer=True)
    best_md = sp.sympify(md_answer(3, "1"), locals={"n": nn})
    ok1 = True
    worst_gap = 1e9
    for trial in range(2000):
        m = random.randint(3, 12)
        while True:
            angs = sorted(random.uniform(0, 2 * math.pi) for _ in range(m))
            gaps = [(angs[(i + 1) % m] - angs[i]) % (2 * math.pi) for i in range(m)]
            if max(gaps) < math.pi:            # 중심이 내부
                break
        pts = [(math.cos(a), math.sin(a)) for a in angs]
        shoelace = 0.5 * abs(sum(pts[i][0] * pts[(i + 1) % m][1] - pts[(i + 1) % m][0] * pts[i][1] for i in range(m)))
        thetas = []
        for i in range(m):                      # θ_i = ∠O A_i A_{i+1}
            A, B = pts[i], pts[(i + 1) % m]
            v1 = (-A[0], -A[1]); v2 = (B[0] - A[0], B[1] - A[1])
            cosang = (v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(*v1) * math.hypot(*v2))
            thetas.append(math.acos(max(-1, min(1, cosang))))
        formula = 0.5 * sum(math.sin(2 * t) for t in thetas)
        bound = float(best_md.subs(nn, m))
        ok1 = ok1 and abs(shoelace - formula) < 1e-9 and formula <= bound + 1e-12 and abs(sum(2 * t for t in thetas) - (m - 2) * math.pi) < 1e-9
        worst_gap = min(worst_gap, bound - formula)
    # 정n각형에서 등호
    ok1 = ok1 and all(sp.simplify(sp.Rational(1, 2) * k * sp.sin(2 * (sp.pi / 2 - sp.pi / k)) - best_md.subs(nn, k)) == 0 for k in range(3, 10))
    # (3-2) [증명-수치확인] 외접 정n각형 넓이 = n tan(π/n) (신발끈), 부등식, 극한
    ok2 = md_answer(3, "2") is None
    for k in range(3, 60):
        R = 1 / math.cos(math.pi / k)          # 외접 정k각형 꼭짓점 반지름
        pts = [(R * math.cos(2 * math.pi * i / k), R * math.sin(2 * math.pi * i / k)) for i in range(k)]
        sh = 0.5 * abs(sum(pts[i][0] * pts[(i + 1) % k][1] - pts[(i + 1) % k][0] * pts[i][1] for i in range(k)))
        ins = float(best_md.subs(nn, k))
        ok2 = ok2 and abs(sh - k * math.tan(math.pi / k)) < 1e-9 and ins < math.pi < sh
    tt = sp.symbols("t", positive=True)
    L1 = sp.limit(best_md.subs(nn, tt), tt, sp.oo)
    L2 = sp.limit(tt * sp.tan(sp.pi / tt), tt, sp.oo)
    ok2 = ok2 and L1 == sp.pi and L2 == sp.pi
    report(3, ok1 and ok2, f"2000 random polygons ok, min(bound-area)={worst_gap:.2e}; limits {L1}, {L2} [증명-수치확인]")


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
