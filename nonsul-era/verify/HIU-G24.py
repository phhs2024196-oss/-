#!/usr/bin/env python3
"""HIU-G24 (홍익대 2024학년도 수시 논술 자연계열 서울) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "HIU-G24"
WIDTH = 2
RESULTS = []
x, y = sp.symbols("x y", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HIU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str, loc=None) -> dict:
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    x1, y1 = sp.symbols("x1 y1", positive=True)
    loc = {"x": x, "y": y, "x1": x1, "y1": y1}
    E = x**2 / 9 + y**2 - 1
    # (1) 음함수 미분
    dydx = sp.idiff(E, y, x)
    ok1 = sp.simplify(dydx - sp.sympify(md_answer(1, "1"), locals=loc)) == 0
    # (2) 접선: 기울기·y절편·x절편 (타원 조건 x1^2/9 + y1^2 = 1 사용)
    c = dydx.subs({x: x1, y: y1})
    d_raw = y1 - c * x1
    d = sp.simplify(d_raw.subs(x1**2, 9 * (1 - y1**2)))
    xint = sp.solve(sp.Eq(0, c * x + d_raw), x)[0]
    xint = sp.simplify(xint.subs(y1**2, 1 - x1**2 / 9))
    p2 = parts(md_answer(1, "2"))
    ok2 = (sp.simplify(c - sp.sympify(p2["c"], locals=loc)) == 0
           and sp.simplify(d - sp.sympify(p2["d"], locals=loc)) == 0
           and sp.simplify(xint - sp.sympify(p2["xint"], locals=loc)) == 0)
    # (3) 마름모 넓이 - 타원 넓이 를 타원 위에서 최소화 (매개변수 x1=3cos u, y1=sin u)
    u = sp.symbols("u", positive=True)
    area = (2 * xint * d - sp.pi * 3 * 1).subs({x1: 3 * sp.cos(u), y1: sp.sin(u)})
    crit = [r for r in sp.solve(sp.diff(area, u), u) if 0 < float(r) < float(sp.pi / 2)]
    um = crit[0]
    amin = sp.nsimplify(sp.simplify(area.subs(u, um)), [sp.pi])
    X1, Y1 = sp.simplify(3 * sp.cos(um)), sp.simplify(sp.sin(um))
    # 최소성: 경계 쪽 큰 값 확인
    ok_min = all(float(area.subs(u, v)) >= float(amin) - 1e-12 for v in [0.1 * k for k in range(1, 15)])
    p3 = parts(md_answer(1, "3"))
    ok3 = ok_min and sp.simplify(X1 - sp.sympify(p3["x1"])) == 0 and sp.simplify(Y1 - sp.sympify(p3["y1"])) == 0 \
        and sp.simplify(amin - sp.sympify(p3["area"])) == 0
    # (4) 초점 F(±a,0), G(0,b) 지나는 타원: 장반축^2 = a^2+b^2, 단반축 = b
    a = xint.subs(x1, X1)
    b = d.subs(y1, Y1)
    A2, B2 = sp.simplify(a**2 + b**2), sp.simplify(b**2)
    lhs, rhs = md_answer(1, "4").split("=")
    md_ell = sp.sympify(lhs, locals=loc) - sp.sympify(rhs, locals=loc)
    ok4 = sp.simplify(md_ell - (x**2 / A2 + y**2 / B2 - 1)) == 0
    # 초점 거리 확인: G 에서 두 초점까지 거리 합 = 2*sqrt(A2)
    ok4 = ok4 and sp.simplify(2 * sp.sqrt(a**2 + b**2) - 2 * sp.sqrt(A2)) == 0
    report(1, ok1 and ok2 and ok3 and ok4,
           f"dy/dx={dydx} c={c} d={d} xint={xint} min={amin} at ({X1},{Y1}) fence x^2/{A2}+y^2/{B2}=1")


def p02():
    import itertools
    A = [2, 4, 23, 25, 29, 31]
    B = [5, 7, 11, 13, 35, 37]
    EX, EY = sp.Rational(sum(A), 6), sp.Rational(sum(B), 6)
    p1 = parts(md_answer(2, "1"))
    ok1 = EX == sp.sympify(p1["EX"]) and EY == sp.sympify(p1["EY"])
    win_h = sp.Rational(sum(1 for a, b in itertools.product(A, B) if a > b), 36)
    p2 = parts(md_answer(2, "2"))
    ok2 = (win_h == sp.sympify(p2["p_hong"]) and 1 - win_h == sp.sympify(p2["p_dad"])
           and (p2["advantage"] == "dad") == (1 - win_h > win_h))
    # (3) 이항분포 정규근사 (표값 0.4772 = P(0<=Z<=2) 사용)
    n, p = 405, win_h
    m, sd = n * p, sp.sqrt(n * p * (1 - p))
    z = (200 - m) / sd
    table = {sp.Rational(1, 2): 0.1915, 1: 0.3413, 2: 0.4772, 3: 0.4987}
    prob = 0.5 - table[sp.nsimplify(z)]
    ok3 = abs(prob - float(md_answer(2, "3"))) < 1e-12 and m == 180 and sd == 10
    # (4) 엄마 주사위: 1..60 에서 A∪B 제외한 수 중 6개, 직접 승리 수 계산으로 p1-p2 최대 탐색
    used = set(A) | set(B)
    cand = [k for k in range(1, 61) if k not in used]
    score = {k: sum(k > a for a in A) - sum(k > b for b in B) for k in cand}
    # 가지치기: 같은 점수 값끼리는 교환 가능하므로 각 점수에서 6개까지만 남겨 전수
    by = {}
    for k in cand:
        by.setdefault(score[k], []).append(k)
    pool = [k for s, ks in by.items() for k in ks[:6]]
    best = max(sum(score[k] for k in comb) for comb in itertools.combinations(pool, 6))
    ok4 = sp.Rational(best, 36) == sp.sympify(md_answer(2, "4"))
    argmax_twos = sorted(by.get(2, []))
    report(2, ok1 and ok2 and ok3 and ok4,
           f"E(X)={EX} E(Y)={EY} P(hong)={win_h} P(W>=200)~{prob:.4f} max(p1-p2)={sp.Rational(best, 36)} (f-g=2: {argmax_twos}, f-g=1: {sorted(by.get(1, []))})")


def p03():
    R0, R1, t, X = sp.symbols("R0 R1 t X", positive=True)
    loc = {"R0": R0, "R1": R1, "t": t, "x": X, "pi": sp.pi}
    # (1) 부피 보존
    R = sp.symbols("R", positive=True)
    Rsol = sp.solve(sp.Eq(sp.Rational(4, 3) * sp.pi * (R0**3 - R**3), sp.pi / 3 * t), R)[0]
    md1 = sp.sympify(md_answer(3, "1").split("=", 1)[1], locals=loc)
    ok1 = all(abs(float((Rsol - md1).subs({R0: 5, t: tv}))) < 1e-12 for tv in (0, 7, 100, 300))
    # (2) 단면 원
    S = sp.pi * (sp.sqrt(R1**2 - X**2))**2
    md2 = sp.sympify(md_answer(3, "2").split("=", 1)[1], locals=loc)
    ok2 = sp.simplify(S - md2) == 0
    # 기하: 반꼭지각 alpha, sin(alpha) = R/(3R) = 1/3. 접점 원은 중심에서 꼭짓점 방향으로 R*sin(alpha) 아래
    sa = sp.Rational(1, 3)
    ca = sp.sqrt(1 - sa**2)
    h_off = R1 * sa                       # 중심-접점원 평면 거리
    Vsub = sp.integrate(S, (X, h_off, R1))
    ok3 = sp.simplify(Vsub - sp.sympify(md_answer(3, "3"), locals=loc)) == 0
    # (4) 접점원 평면까지의 원뿔(꼭짓점에서 높이 H, 반지름 H tan(alpha))
    H = 3 * R1 - h_off
    cone = sp.pi * (H * sa / ca)**2 * H / 3
    water = sp.simplify(cone - Vsub)
    R1sol = sp.solve(sp.Eq(sp.Rational(4, 3) * sp.pi * (R0**3 - R1**3), water), R1)[0]
    t1 = sp.solve(sp.Eq(sp.pi / 3 * t, water.subs(R1, R1sol)), t)[0]
    ok4 = sp.simplify(t1.subs(R0, 5) - sp.sympify(md_answer(3, "4"))) == 0
    # 수치 교차확인: 구 캡 공식
    cap = sp.pi * (R1 - h_off)**2 * (3 * R1 - (R1 - h_off)) / 3
    ok3 = ok3 and sp.simplify(cap - Vsub) == 0
    report(3, ok1 and ok2 and ok3 and ok4,
           f"R={Rsol} submerged={sp.simplify(Vsub)} water={water} R1^3={sp.simplify(R1sol**3)} t1(R0=5)={t1.subs(R0, 5)}")


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
