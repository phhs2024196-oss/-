#!/usr/bin/env python3
"""HIU-G22A (홍익대 2022학년도 수시 논술 자연계열 서울 오전) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "HIU-G22A"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HIU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def kv(s: str) -> dict:
    return {k.strip(): v.strip() for k, v in (seg.split("=", 1) for seg in s.split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # 설정: A 원점, P=(-1,0)(처음 B 위치), 막대1 길이 1, 막대2 길이 2, 막대2 는 항상 P 를 지난다.
    t = sp.symbols("t", real=True)
    A = sp.Matrix([0, 0]); P = sp.Matrix([-1, 0])
    # 연결점 B' 은 A 중심 반지름 1 원 위를 속력 1 로 시계 방향(위쪽으로) 이동 → 각 = t
    Bp = A + sp.Matrix([-sp.cos(t), sp.sin(t)])
    speedB = sp.simplify(sp.sqrt(sum(c**2 for c in Bp.diff(t))))
    u = (P - Bp) / sp.sqrt(sum(c**2 for c in (P - Bp)))        # B'→P 단위벡터
    Cp = Bp + 2 * u                                            # 막대2 의 끝
    t0 = sp.pi / 3
    d = sp.nsimplify(sp.sqrt(sum(c**2 for c in Cp.subs(t, t0))).evalf(30), [sp.sqrt(3)])
    v = Cp.diff(t); a = v.diff(t)
    sp_v = sp.sqrt(sum(c**2 for c in v)).subs(t, t0).evalf(30)
    sp_a = sp.sqrt(sum(c**2 for c in a)).subs(t, t0).evalf(30)
    ok1 = sp.simplify(d - sp.sympify(md_answer(1, "1"))) == 0
    ok2 = abs(sp_v - sp.sympify(md_answer(1, "2")).evalf(30)) < 1e-20
    ok3 = abs(sp_a - sp.sympify(md_answer(1, "3")).evalf(30)) < 1e-20
    # 공식 해설의 일반식(속력 sqrt(2-2 sin t/2), 가속도 sqrt(5/4 - sin t/2))도 표본점에서 대조
    okg = all(abs((sp.sqrt(sum(c**2 for c in v)) - sp.sqrt(2 - 2 * sp.sin(t / 2))).subs(t, s).evalf()) < 1e-12
              and abs((sp.sqrt(sum(c**2 for c in a)) - sp.sqrt(sp.Rational(5, 4) - sp.sin(t / 2))).subs(t, s).evalf()) < 1e-12
              for s in (0.3, 0.9, 1.7, 2.5))
    report(1, ok1 and ok2 and ok3 and okg and speedB == 1,
           f"|AC'|={d} speed={sp_v:.12f} acc={sp_a:.12f} 일반식 대조={okg}")


def p02():
    # <표 1> (문항에 주어진 값) 으로 계산 — 실제 정규분포값으로도 근사 확인
    tab = {sp.Rational(1, 4): sp.Rational(10, 100), sp.Rational(1, 2): sp.Rational(19, 100),
           sp.Rational(3, 4): sp.Rational(27, 100), 1: sp.Rational(34, 100), sp.Rational(5, 4): sp.Rational(39, 100),
           sp.Rational(3, 2): sp.Rational(43, 100), sp.Rational(7, 4): sp.Rational(46, 100), 2: sp.Rational(48, 100)}
    from sympy.stats import Normal, P as Pr
    def err_exact(r):
        X0 = Normal("X0", 0, sp.Rational(1, 2)); X1 = Normal("X1", sp.Integer(r)**-2, sp.Rational(1, 2))
        th = sp.Rational(1, 2) * sp.Integer(r)**-2
        return sp.Rational(1, 2) * Pr(X1 <= th) + sp.Rational(1, 2) * Pr(X0 > th)
    def err_table(r):                    # 표준화 결과 P(Z > r^-2) 를 표로
        z = sp.Integer(r)**-2
        return sp.Rational(1, 2) - tab[z]
    # 표준화 식 확인: 정확한 오류확률 = P(Z > r^-2)
    Z = Normal("Z", 0, 1)
    okz = all(abs((err_exact(r) - Pr(Z > sp.Integer(r)**-2)).evalf()) < 1e-12 for r in (1, 2))
    p1, p2 = err_table(1), err_table(2)
    ans1 = sp.Rational(md_answer(2, "1"))
    ok1 = p1 == ans1
    m = kv(md_answer(2, "2"))
    rnd = lambda q: sp.Rational(round(float(q) * 100 + 1e-9), 100)
    ok2 = (sp.Rational(m["P1"]) == p1 and sp.Rational(m["P2"]) == rnd(p2) and sp.Rational(m["P1sq"]) == rnd(p1**2)
           and (m["compare"] == "P2>P1sq") == (rnd(p2) > rnd(p1**2)))
    pc = 2 * p1 * (1 - p1)                       # 정확히 한 구간에서만 오류
    # 전수(4가지 오류/무오류 조합)로도 확인
    pc_enum = sum((p1 if e1 else 1 - p1) * (p1 if e2 else 1 - p1) for e1 in (0, 1) for e2 in (0, 1) if e1 ^ e2)
    m3 = kv(md_answer(2, "3"))
    ok3 = (pc == pc_enum and sp.Rational(m3["PC"]) == rnd(pc)
           and (m3["compare"] == "PC<0.40") == (rnd(pc) < rnd(p2)))
    report(2, okz and ok1 and ok2 and ok3,
           f"P(r=1)={p1} P(r=2)={p2} P1^2={p1**2} PC={pc} (실제 정규 r=1:{float(err_exact(1)):.4f})")


def p03():
    l, w, R = sp.symbols("l w R", positive=True)
    th, th1, th2 = sp.symbols("theta theta1 theta2", positive=True)
    # (1) 좌표: 뒷바퀴 B=(0,0) 진행방향 (0,1), 앞바퀴 A=(0,l) 진행방향 (−sin th, cos th).
    #     중심 O 는 B 를 지나는 x축 위 (−R,0), OA ⊥ 앞바퀴 진행방향.
    Rsol = sp.solve(sp.Eq((0 - (-R)) * (-sp.sin(th)) + (l - 0) * sp.cos(th), 0), R)[0]
    RA = sp.sqrt(Rsol**2 + l**2); RB = Rsol
    m1 = kv(md_answer(3, "1"))
    loc = {"l": l, "theta": th}
    ok1 = all(abs((RA - sp.sympify(m1["R_front"], locals=loc)).subs({l: 1.7, th: s}).evalf()) < 1e-12 and
              abs((RB - sp.sympify(m1["R_rear"], locals=loc)).subs({l: 1.7, th: s}).evalf()) < 1e-12
              for s in (0.2, 0.7, 1.3))
    # (2) 안쪽 뒷바퀴 B=(0,0), 바깥 D=(w,0), 공통 중심 O=(−R,0). 각 앞바퀴의 조향각 = 직선 O-앞바퀴 와 수평 사이 각
    t1 = sp.atan(l / R); t2 = sp.atan(l / (R + w))
    lhs, rhs = [sp.sympify(s, locals={"l": l, "w": w, "theta1": th1, "theta2": th2})
                for s in md_answer(3, "2").split("=")]
    rel = (lhs - rhs).subs({th1: t1, th2: t2})
    ok2 = all(abs(rel.subs({l: 2.3, w: 1.1, R: rv}).evalf()) < 1e-12 for rv in (0.5, 3, 7.2))
    # (3) 네 바퀴 반지름의 최대 − 최소 (수치 표본으로 직접)
    s_md = sp.sympify(md_answer(3, "3"), locals={"l": l, "w": w, "theta2": th2})
    ok3 = True
    for lv, wv, rv in ((2.3, 1.1, 0.5), (3.0, 1.6, 4.0), (1.2, 0.9, 10.0)):
        radii = [rv, rv + wv, (rv**2 + lv**2) ** 0.5, ((rv + wv)**2 + lv**2) ** 0.5]
        smin = max(radii) - min(radii)
        val = s_md.subs({l: lv, w: wv, th2: sp.atan(lv / (rv + wv))}).evalf()
        ok3 &= abs(val - smin) < 1e-12
    report(3, ok1 and ok2 and ok3, "R_A=l/sinθ, R_B=l/tanθ, 관계식·최소폭 표본 대조")


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
