#!/usr/bin/env python3
"""서울과학기술대 2021학년도 수시모집 모의논술(자연계열) 문제 1~3 (SNUT-M21-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산해 YAML answer.value 와 비교."""
import sys

import sympy as sp

PREFIX = "SNUT-M21"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
a, b, k = sp.symbols("a b k", real=True)
SYM = {"x": x, "t": t, "n": n, "a": a, "b": b, "k": k, "pi": sp.pi, "oo": sp.oo}


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



def p01():
    """1.1 세 곡선을 구간마다 값 정렬해 A(중간-최저), B(최고-중간) 넓이 수치적분, 1.2 큰 n 에서 loggamma 로 직접 계산, 1.3 전수 열거."""
    import mpmath as mp
    mp.mp.dps = 30
    # 1.1 교점: 세 곡선 중 둘씩 같아지는 x (0..π) 를 수치로 찾는다
    cs = [mp.cos, lambda X: mp.cos(2 * X), lambda X: mp.cos(X) ** 2]
    pts = set()
    for i in range(3):
        for j in range(i + 1, 3):
            for x0 in [k * mp.pi / 40 for k in range(0, 41)]:
                try:
                    r = mp.findroot(lambda X: cs[i](X) - cs[j](X), x0)
                    if -1e-9 <= r <= mp.pi + 1e-9:
                        pts.add(round(float(r), 8))
                except Exception:
                    pass
    expect = sorted(round(float(v), 8) for v in (0, mp.pi / 2, 2 * mp.pi / 3, mp.pi))
    ok_pts = sorted(pts) == expect
    srt = lambda X: sorted([cs[0](X), cs[1](X), cs[2](X)])
    a_area = mp.quad(lambda X: srt(X)[1] - srt(X)[0], [0, mp.pi / 2, 2 * mp.pi / 3])           # A: 중간~최저, 0..2π/3
    b_area = mp.quad(lambda X: srt(X)[2] - srt(X)[1], [mp.pi / 2, 2 * mp.pi / 3, mp.pi])      # B: 최고~중간, π/2..π
    ans1 = float(load_answer(1, 0))
    ok1 = ok_pts and abs(float(a_area - b_area) - ans1) < 1e-15 and abs(float(a_area) - (math_pi4 := float(mp.pi / 4 + 3 * mp.sqrt(3) / 4 - 1))) < 1e-15
    # 1.2
    nn = mp.mpf(10) ** 9
    val = (mp.loggamma(2 * nn + 1) - nn * mp.log(nn) - mp.loggamma(nn + 1)) / nn
    ans2 = float(load_answer(1, 1))
    ok2 = abs(float(val) - ans2) < 1e-8
    # 1.3
    cnt = sum(1 for b_ in range(21) for c_ in range(21) for s_ in range(18) if b_ + c_ + s_ == 20)
    ok3 = cnt == load_answer(1, 2)
    report(1, ok1 and ok2 and ok3, f"[1.1] 교점 {sorted(pts)}, a={float(a_area):.12f}, b={float(b_area):.12f}, a-b={float(a_area-b_area):.12f} (yaml {ans1:.12f}); [1.2] n=1e9 에서 {float(val):.10f} (yaml {ans2:.10f}); [1.3] 전수={cnt}")


def p02():
    """좌표로 직접: C2 중심·반지름, 교점 A(두 개 모두), 원점에서 가장 먼 Q, 2.1 sinα, 2.2 cosβ, 2.3 접선 조건 연립."""
    X, Y = sp.symbols("X Y", real=True)
    C1 = X ** 2 + Y ** 2 - 9
    C2 = X ** 2 - 8 * X + Y ** 2 - 6 * Y + 9
    Pc = (4, 3)
    ok_c = sp.simplify(C2 - ((X - 4) ** 2 + (Y - 3) ** 2 - 16)) == 0
    inter = sp.solve([C1, C2], [X, Y], dict=True)
    Aset = [(s[X], s[Y]) for s in inter]
    # Q: C2 위에서 O 와 가장 먼 점 = P + 4 * P/|P|
    Qv = (Pc[0] + 4 * sp.Rational(Pc[0], 5), Pc[1] + 4 * sp.Rational(Pc[1], 5))
    th = sp.symbols("th", real=True)
    dist2 = (Pc[0] + 4 * sp.cos(th)) ** 2 + (Pc[1] + 4 * sp.sin(th)) ** 2
    far = sp.maximum(dist2, th, sp.Interval(0, 2 * sp.pi))
    ok_q = sp.simplify(Qv[0] ** 2 + Qv[1] ** 2 - far) == 0 and sp.simplify(C2.subs({X: Qv[0], Y: Qv[1]})) == 0
    def ang(U, V, W):                       # ∠UVW 의 코사인
        a1 = (U[0] - V[0], U[1] - V[1]); a2 = (W[0] - V[0], W[1] - V[1])
        return sp.simplify((a1[0] * a2[0] + a1[1] * a2[1]) / (sp.sqrt(a1[0] ** 2 + a1[1] ** 2) * sp.sqrt(a2[0] ** 2 + a2[1] ** 2)))
    O = (0, 0)
    ans1, ans2 = load_answer(2, 0), load_answer(2, 1)
    ok12 = True
    detail = []
    for A in Aset:
        cos_alpha = ang(O, Qv, A)           # ∠OQA
        sin_alpha = sp.simplify(sp.sqrt(1 - cos_alpha ** 2))
        cos_beta = ang(O, A, Qv)            # ∠OAQ
        ok12 &= sp.simplify(sin_alpha - ans1) == 0 and sp.simplify(cos_beta - ans2) == 0
        detail.append((A, sin_alpha, cos_beta))
    # 2.3: 접점 (a,b) ∈ C1, 접선 ax+by=9 와 P 사이의 거리 = 4
    a_, b_ = sp.symbols("a_ b_", real=True)
    sols = []
    for sg in (1, -1):
        sols += sp.solve([a_ ** 2 + b_ ** 2 - 9, 4 * a_ + 3 * b_ - 9 - sg * 12], [a_, b_], dict=True)
    real_sols = [s for s in sols if all(v.is_real for v in s.values())]
    xs = [s[a_] for s in real_sols]
    # 거리 직접 재확인
    chk = all(sp.simplify(sp.Abs(4 * s[a_] + 3 * s[b_] - 9) / sp.sqrt(s[a_] ** 2 + s[b_] ** 2) - 4) == 0 for s in real_sols)
    ans3 = load_answer(2, 2)
    ok3 = len(real_sols) == 2 and chk and sp.simplify(sum(xs) - ans3) == 0
    report(2, ok_c and ok_q and ok12 and ok3, f"C2 중심(4,3) r=4 {ok_c}, 교점 A={Aset}, Q={Qv}; [2.1/2.2] (sinα, cosβ) A별={[(d[1], d[2]) for d in detail]}; [2.3] 접점 {len(real_sols)}개, x좌표 합={sp.simplify(sum(xs))}")


def p03():
    """규칙대로 점 P_n 을 Fraction 으로 직접 생성(n≤60) 하여 3.1 좌표식·3.2/3.3 직선·3.4 극한점 확인."""
    from fractions import Fraction as F
    pts = [(F(0), F(0))]
    L = F(1)
    dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    for k in range(1, 61):
        if k >= 2:
            L = L * (F(3, 4) if k % 2 == 0 else F(2, 3))
        dx, dy = dirs[(k - 1) % 4]
        pts.append((pts[-1][0] + dx * L, pts[-1][1] + dy * L))
    f31 = load_answer(3, 0)                      # (x_2n, y_2n, x_2n-1, y_2n-1) 식
    ok1 = True
    for m in range(1, 31):
        vals = [sp.Rational(v) if v.is_Rational else v for v in (e.subs(n, m) for e in f31)]
        ok1 &= (sp.Rational(pts[2 * m][0].numerator, pts[2 * m][0].denominator) == vals[0] and sp.Rational(pts[2 * m][1].numerator, pts[2 * m][1].denominator) == vals[1]
                and sp.Rational(pts[2 * m - 1][0].numerator, pts[2 * m - 1][0].denominator) == vals[2] and sp.Rational(pts[2 * m - 1][1].numerator, pts[2 * m - 1][1].denominator) == vals[3])
    fx = load_answer(3, 1)
    gx = load_answer(3, 2)
    ok2 = all(sp.Rational(p[1].numerator, p[1].denominator) == fx.subs(x, sp.Rational(p[0].numerator, p[0].denominator)) for p in pts[0::2])
    ok3 = all(sp.Rational(p[1].numerator, p[1].denominator) == gx.subs(x, sp.Rational(p[0].numerator, p[0].denominator)) for p in pts[1::2])
    Q = sp.solve([sp.Eq(sp.Symbol("yy"), fx.subs(x, sp.Symbol("xx"))), sp.Eq(sp.Symbol("yy"), gx.subs(x, sp.Symbol("xx")))], [sp.Symbol("xx"), sp.Symbol("yy")], dict=True)[0]
    Qp = (Q[sp.Symbol("xx")], Q[sp.Symbol("yy")])
    last = pts[-1]
    near = abs(float(last[0]) - float(Qp[0])) < 1e-8 and abs(float(last[1]) - float(Qp[1])) < 1e-8
    ok4 = tuple(sp.sympify(load_answer(3, 3, raw=True), locals=SYM)) == tuple(Qp) and near
    report(3, ok1 and ok2 and ok3 and ok4, f"[3.1] n=1..30 좌표식 일치 {ok1}; [3.2] P_짝수 직선 y=3x/4 {ok2}; [3.3] P_홀수 직선 {ok3}; [3.4] Q={Qp}, P_60≈({float(last[0]):.9f},{float(last[1]):.9f}) [증명-수치확인]")


if __name__ == "__main__":
    for name in sorted(k for k in dict(globals()) if k.startswith("p") and k[1:].isdigit()):
        globals()[name]()
    passed = sum(RESULTS)
    print(f"SUMMARY M21: {passed}/{len(RESULTS)} PASS")
    sys.exit(0 if passed == len(RESULTS) else 1)
