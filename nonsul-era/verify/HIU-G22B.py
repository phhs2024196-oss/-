#!/usr/bin/env python3
"""HIU-G22B (홍익대 2022학년도 수시 논술 자연계열 서울 오후) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys

import sympy as sp

PREFIX = "HIU-G22B"
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
    # 제시문 (다): log2 = 0.3 로 계산. 모든 로그는 log2·log10 으로 환산해 유리수로 다룬다.
    L2 = sp.Rational(3, 10); L5 = 1 - L2
    lg04 = 2 * L2 - 1; lg02 = L2 - 1; lg125 = 3 * L5 - 2
    lg_ratio0 = lg02                         # log(30/150)
    # (1) 150*0.4^t <= 30  ⇔ t*lg04 <= lg02
    t = sp.symbols("t", real=True)
    t1 = sp.solve(sp.Eq(t * lg04, lg_ratio0), t)[0]
    ok1 = t1 == sp.Rational(md_answer(1, "1"))
    # 로그농도 상대값 y(τ) = log(C/150), 위반 ⇔ y > lg02. 시뮬레이션(유리수 정확 계산, 구간별 선형)
    def violation(t_open, period, opens, days_hours):
        """t_open 시간 열고 period - t_open 시간 닫는 주기. 위반 총시간·마지막 위반 시각 반환."""
        y = sp.Integer(0); tot = sp.Integer(0); last = sp.Integer(0); clock = sp.Integer(0)
        while clock < days_hours:
            for dur, slope in ((t_open, lg04), (period - t_open, lg125)):
                y0, y1 = y, y + slope * dur
                # 구간 [clock, clock+dur] 에서 y>lg02 인 길이 (선형)
                if y0 > lg02 and y1 > lg02:
                    seg = dur; end = clock + dur
                elif y0 <= lg02 and y1 <= lg02:
                    seg = 0; end = None
                else:
                    cross = clock + (lg02 - y0) / slope
                    if y0 > lg02:
                        seg = cross - clock; end = cross
                    else:
                        seg = clock + dur - cross; end = clock + dur
                tot += seg
                if end is not None:
                    last = end
                y = y1; clock += dur
        return tot, last
    # (2) t 를 0.05 간격으로 훑어 위반 총시간이 최소가 되는 가장 작은 t
    grid = [sp.Rational(k, 20) for k in range(1, 24 * 20)]
    vals = {tv: violation(tv, 24, None, 24 * 40)[0] for tv in grid}
    vmin = min(vals.values())
    tmin = min(tv for tv, v in vals.items() if v == vmin)
    # 경계를 정확히: tmin 바로 아래 t 에서는 위반이 더 큼을 확인하고 t=md 에서 최소 달성
    t2 = sp.Rational(md_answer(1, "2"))
    ok2 = (violation(t2, 24, None, 24 * 40)[0] == vmin and
           violation(t2 - sp.Rational(1, 1000), 24, None, 24 * 40)[0] > vmin and tmin >= t2 > tmin - sp.Rational(1, 20))
    # (3) 2시간 열고 6시간 닫기 반복: 마지막 위반 종료 시각 = 입주 가능 최소 시각
    tot3, last3 = violation(sp.Integer(2), 8, None, 8 * 60)
    ok3 = last3 == sp.Rational(md_answer(1, "3"))
    report(1, ok1 and ok2 and ok3, f"(1) t={t1} (2) 최소위반={vmin} at t≥{t2} (격자 {tmin}) (3) {last3}h")


def p02():
    A, a1, a3, r, x = sp.symbols("A a1 a3 r x", positive=True)
    loc = {"A": A, "a1": a1, "x": x, "pi": sp.pi}
    # (1) 원통 껍질로 독립 계산: V = ∫0^1 2πr (A r^2 + a1) dr
    V_shell = sp.integrate(2 * sp.pi * r * (A * r**2 + a1), (r, 0, 1))
    m = kv(md_answer(2, "1"))
    S_md = sp.sympify(m["S"], locals=loc); V_md = sp.sympify(m["V"], locals=loc)
    # 단면: 높이 x 에서 안쪽 반지름^2 = x/A
    S_dir = sp.pi * 1**2 - sp.pi * (x / A)
    ok1 = sp.simplify(S_md - S_dir) == 0 and sp.simplify(V_md - V_shell) == 0
    # (2) 부피 보존 2π, a1 = 2 - √3/12
    Aval = sp.solve(sp.Eq(V_shell.subs(a1, 2 - sp.sqrt(3) / 12), 2 * sp.pi), A)[0]
    ok2 = sp.simplify(Aval - sp.sympify(md_answer(2, "2"))) == 0
    # (3) r=1 에서 접선 기울기 → 각
    th1 = sp.atan(sp.diff(Aval * r**2, r).subs(r, 1))
    ok3 = sp.simplify(th1 - sp.sympify(md_answer(2, "3"))) == 0
    # (4) 가득(3π) → 회전 후 가장자리 높이 3 : a3 = 3 - A, 넘친 양 = 3π - 남은 부피
    a3v = 3 - Aval
    rem = sp.integrate(2 * sp.pi * r * (Aval * r**2 + a3v), (r, 0, 1))
    over = sp.simplify(3 * sp.pi - rem)
    ok4 = sp.simplify(over - sp.sympify(md_answer(2, "4"))) == 0
    # (5) 물컵 수면 기울기 2A r ≤ tan 60°
    rmax = sp.solve(sp.Eq(2 * Aval * r, sp.tan(sp.pi / 3)), r)[0]
    ok5 = sp.simplify(rmax - sp.sympify(md_answer(2, "5"))) == 0
    report(2, ok1 and ok2 and ok3 and ok4 and ok5, f"V={V_shell} A={Aval} θ1={th1} 넘침={over} rmax={rmax}")


def p03():
    def col_fix(col):                       # 세로줄 수정: 오류 1개 이하면 모두 수정
        return col if sum(col) >= 2 else (0, 0, 0)

    def classify(n):
        cntA = cntB = cntFirst = 0
        for cols in itertools.product(itertools.product((0, 1), repeat=3), repeat=n):
            fixed = [col_fix(c) for c in cols]
            if fixed[0] != (0, 0, 0):
                cntFirst += 1
            inA = sum(1 for c in fixed if c != (0, 0, 0)) >= 3
            rows = [[fixed[j][i] for j in range(n)] for i in range(3)]
            remain = any(sum(rw) >= 3 for rw in rows)   # 가로줄 오류 3개 이상이면 남음 → 오작동
            cntA += inA; cntB += remain
            if remain and not inA:
                raise AssertionError("B ⊄ A")
        tot = 2 ** (3 * n)
        return sp.Rational(cntFirst, tot), sp.Rational(cntA, tot), sp.Rational(cntB, tot)

    f3, A3, B3 = classify(3)
    f4, A4, B4 = classify(4)
    n = sp.symbols("n", integer=True, positive=True)
    PA_md = sp.sympify(md_answer(3, "2"), locals={"n": n})
    okA = all(PA_md.subs(n, k) == classify(k)[1] for k in (3, 4, 5))
    ok1 = f3 == sp.Rational(md_answer(3, "1"))
    ok3 = (A3 - B3) == sp.Rational(md_answer(3, "3"))
    ok4 = B3 == sp.Rational(md_answer(3, "4"))
    ok5 = B4 == sp.Rational(md_answer(3, "5"))
    report(3, ok1 and okA and ok3 and ok4 and ok5,
           f"전수열거 n=3: P1={f3} P(A)={A3} P(C)={A3 - B3} P(B)={B3}; n=4: P(A)={A4} P(B)={B4}; P(A) 식 n=3..5 일치")


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
