#!/usr/bin/env python3
"""KKU-G23B (건국대 2023학년도 수시 논술 자연계 B 문제 1~5) — sympy 검산.
공식 해설 미확보 세트: 문제 설정으로부터 독립 재계산해 YAML answer.value 와 비교(정답 하드코딩 금지).
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "KKU-G23B"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/KKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    """원 x^2+y^2=1, l1: y=2, l2: y=-2, A_n=(n,2). 접선을 원 위 접점 (cos t, sin t) 로 잡아 독립 계산."""
    n = sp.symbols("n", positive=True)

    def d(N):
        t = sp.symbols("t", real=True)
        # 접점 (cos t, sin t) 에서의 접선 x cos t + y sin t = 1 이 A_n 을 지남
        sols = sp.solve(sp.Eq(N * sp.cos(t) + 2 * sp.sin(t), 1), t)
        xs = []
        for s in sols:
            c, si = sp.cos(s), sp.sin(s)
            xs.append(sp.nsimplify(sp.simplify((1 + 2 * si) / c)))   # y=-2 일 때 x
        assert len(xs) == 2
        return sp.Abs(sp.simplify(xs[0] - xs[1]))

    vals = {N: sp.simplify(d(N)) for N in range(2, 9)}
    closed = 8 * sp.sqrt(n**2 + 3) / 3
    ok_closed = all(sp.simplify(v - closed.subs(n, N)) == 0 for N, v in vals.items())
    lim = sp.limit(closed / n, n, sp.oo)
    ok = ok_closed and sp.simplify(lim - sp.sympify(md_answer(1, "1"))) == 0
    report(1, ok, f"d_n 접점 계산 n=2..8 이 8√(n²+3)/3 와 일치={ok_closed}, lim d_n/n={lim}")


def p02():
    """A,B 홀수·C,D 짝수(≥1)·E 5의 배수(≥1), 합 40 — 전수 열거."""
    cnt = 0
    for a, b, c, d in itertools.product(range(1, 41), repeat=4):
        e = 40 - a - b - c - d
        if e >= 1 and a % 2 and b % 2 and c % 2 == 0 and d % 2 == 0 and e % 5 == 0:
            cnt += 1
    ok = cnt == int(sp.sympify(md_answer(2, "1")))
    report(2, ok, f"전수 열거 {cnt}가지")


def p03():
    """A(0,0),B(1,0), P = 두 직선 y=x tanθ1, y=(1-x)tanθ2 의 교점. θ1(t)=π/2+t/3, θ2(t)=π/4 + 0·t + c t^2 (θ2'(0)=0)."""
    t, c = sp.symbols("t c", real=True)
    th1 = sp.pi / 2 + t / 3
    th2 = sp.pi / 4 + c * t**2
    # 교점: 방향 벡터로 직접(탄젠트 사용 피함) — P = A + r(cosθ1, sinθ1), P = B + s(-cosθ2, sinθ2)
    r, s = sp.symbols("r s")
    sol = sp.solve([r * sp.cos(th1) - (1 - s * sp.cos(th2)), r * sp.sin(th1) - s * sp.sin(th2)], [r, s], dict=True)[0]
    X = sol[r] * sp.cos(th1)
    Y = sol[r] * sp.sin(th1)
    vx = sp.simplify(sp.diff(X, t).subs(t, 0))
    vy = sp.simplify(sp.diff(Y, t).subs(t, 0))
    speed = sp.simplify(sp.sqrt(vx**2 + vy**2))
    ok = sp.simplify(speed - sp.sympify(md_answer(3, "1"))) == 0 and not speed.has(c)
    report(3, ok, f"속도=({vx}, {vy}) 속력={speed} (θ2''(0)=c 무관)")


def p04():
    """좌표로 A,B,C 를 놓고 D,E,F, 세 외심을 직접 구해 넓이 계산."""
    B = sp.Point(0, 0); C = sp.Point(6, 0)
    ax, ay = sp.symbols("ax ay", real=True)
    sol = [s for s in sp.solve([ax**2 + ay**2 - 16, (ax - 6)**2 + ay**2 - 25], [ax, ay]) if s[1] > 0][0]
    A = sp.Point(*sol)
    D = B + (C - B) * sp.Rational(1, 3)
    E = sp.Line(D, D + (C - A)).intersection(sp.Line(A, B))[0]       # DE ∥ AC
    F = sp.Line(D, D + (B - A)).intersection(sp.Line(A, C))[0]       # DF ∥ AB
    O = sp.Triangle(A, B, C).circumcenter
    O1 = sp.Triangle(B, D, E).circumcenter
    O2 = sp.Triangle(D, C, F).circumcenter
    area = sp.nsimplify(sp.simplify(abs(sp.Triangle(O, O1, O2).area)))
    ok = sp.simplify(area - sp.sympify(md_answer(4, "1"))) == 0
    on_seg = sp.Segment(B, O).contains(O1) and sp.Segment(C, O).contains(O2)
    report(4, ok and on_seg, f"넓이={area} (O1∈BO, O2∈CO: {on_seg})")


def p05():
    """B(0,0), A(0,k), C(0,-1), P(2,-x). l3 ⊥ BP 를 B'=P+λ(B-P) 에 두고 교점을 직접 계산."""
    x, k, lam = sp.symbols("x k lam", positive=True)
    B = sp.Matrix([0, 0]); A = sp.Matrix([0, k]); C = sp.Matrix([0, -1]); P = sp.Matrix([2, -x])
    u = B - P
    Bp = P + lam * u
    v = sp.Matrix([-u[1], u[0]])   # l3 방향

    def meet(Q):
        m, w = sp.symbols("m w")
        s = sp.solve(list(P + m * (Q - P) - (Bp + w * v)), [m, w], dict=True)[0]
        return s[w]                 # l3 위 좌표(배수)

    wa, wc = meet(A), meet(C)
    # A', C' 는 B'(w=0) 양쪽에 있음을 확인한 뒤 부호 없이 A'B'/A'C' = wa/(wa-wc)
    opp = all((wa * wc).subs({x: xv, k: kv, lam: lv}) < 0
              for xv in (sp.Rational(1, 2), 2, 5) for kv in (sp.Rational(1, 3), 3) for lv in (sp.Rational(1, 4), sp.Rational(3, 4)))
    f = sp.factor(sp.simplify((wa / (wa - wc)) / (k / (k + 1))))
    ok_k = sp.diff(f, k) == 0 or sp.simplify(sp.diff(f, k)) == 0
    crit = [r_ for r_ in sp.solve(sp.diff(f, x), x) if r_.is_positive]
    fx = sp.lambdify(x, f.subs(k, 3).subs(lam, sp.Rational(1, 3)))
    xmin = min((i / 1000 for i in range(1, 20001)), key=fx)
    ans = sp.sympify(md_answer(5, "1"))
    ok = opp and ok_k and crit == [ans] and abs(xmin - float(ans)) < 2e-3
    report(5, ok, f"f(x)={f} (AB=k 무관={ok_k}) 임계점={crit} 격자 최소 x≈{xmin}")


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
