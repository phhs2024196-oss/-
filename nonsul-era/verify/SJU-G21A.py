#!/usr/bin/env python3
"""SJU-G21A (세종대 2021학년도 수시 논술고사 [A]) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SJU-G21A"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    return dict(seg.strip().split("=", 1) for seg in s.split(";"))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    import itertools
    # (1-1) 바구니 종류 직접 열거 → 4개 중복조합 직접 열거
    kinds = [(a, b, 9 - a - b) for a in range(2, 10) for b in range(2, 10) if 9 - a - b >= 2]
    orders = len(list(itertools.combinations_with_replacement(kinds, 4)))
    ok1 = orders == int(md_answer(1, "1"))
    # (1-2) 정규분포 CDF 로 재포장률 일치하는 a 를 수치로
    import mpmath as mp
    mp.mp.dps = 30
    p9 = mp.ncdf(240, 250, mp.mpf(15) / 3)
    a16 = mp.findroot(lambda a: mp.ncdf(a, 250, mp.mpf(15) / 4) - p9, 240)
    ok2 = abs(a16 - float(sp.sympify(md_answer(1, "2")))) < 1e-9
    # (1-3) 이항분포 정확 계산 → E(Y) 를 0.9^9 의 일차식으로
    A = sp.Rational(9, 10)**9
    pk = lambda k: sp.binomial(9, k) * sp.Rational(1, 10)**k * sp.Rational(9, 10)**(9 - k)
    pok = sum(pk(k) for k in range(3))
    EY = 3600 * pok + (3600 - 9000) * (1 - pok)
    P = parts(md_answer(1, "3"))
    av, bv = sp.Integer(P["a"]), sp.Integer(P["b"])
    ok3 = sp.simplify(EY - (A * av + bv)) == 0 and sp.simplify(pok / A - sp.Rational(22, 9)) == 0
    report(1, ok1 and ok2 and ok3, f"kinds={len(kinds)} orders={orders} a16={float(a16):.10f} E(Y)={float(EY):.4f}")


def p02():
    f = sp.Function("f")
    # 후보 f 를 md 에서 읽어 조건 (가)(나)(다) 를 직접 확인
    F = sp.sympify(md_answer(2, "3"), locals={"x": x})
    okA = sp.simplify(F.subs(x, 0) - 1) == 0
    okB = all(sp.diff(F, x).subs(x, v) > 0 for v in [sp.Rational(1, 10), 1, 3, 10])
    tt = sp.symbols("tt", positive=True)
    area = sp.integrate(F, (x, 0, tt))
    import mpmath as mp
    okC = all(abs(mp.quad(lambda s: mp.sqrt(1 + float(sp.diff(F, x).subs(x, s))**2), [0, v]) - float(area.subs(tt, v))) < 1e-10 for v in [0.5, 1, 2.5])
    # (2-1) f'(0)
    ok1 = sp.simplify(sp.diff(F, x).subs(x, 0) - sp.sympify(md_answer(2, "1"))) == 0
    # (2-2) g = f + f'
    ok2 = sp.simplify(F + sp.diff(F, x) - sp.sympify(md_answer(2, "2"), locals={"x": x})) == 0
    # 독립 유도: f''=f, f(0)=1, f'(0)=0 의 해
    sol = sp.dsolve(sp.Eq(f(x).diff(x, 2), f(x)), f(x), ics={f(0): 1, f(x).diff(x).subs(x, 0): 0}).rhs
    ok3 = sp.simplify(sol - F) == 0 and okA and okB and okC
    report(2, ok1 and ok2 and ok3, f"f={sp.simplify(sol)} 조건(가)(나)(다)={okA},{okB},{okC}")


def p03():
    import mpmath as mp
    mp.mp.dps = 40
    e = mp.e
    g = lambda tt: (e - 1) * tt - mp.log(tt)
    t0 = 1 / (e - 1)
    # (3-1) g 최솟값 > 1, 그리고 f(x)<=1 for x<=1 → x(t)>1 (수치: x - ln x = g(t) 의 해가 >1 에 유일)
    ok1 = g(t0) > 1 and all(g(tt) > 1 for tt in [t0 + mp.mpf(k) / 7 for k in range(1, 50)])
    def X(tt):   # x - ln x = g(t), x > 1 (x - ln x 는 x>1 에서 증가) — 이분법 후 뉴턴 정밀화
        lo, hi = mp.mpf(1), mp.mpf(100)
        for _ in range(60):
            mid = (lo + hi) / 2
            if mid - mp.log(mid) < g(tt):
                lo = mid
            else:
                hi = mid
        return mp.findroot(lambda u: u - mp.log(u) - g(tt), (lo + hi) / 2)
    ok1 = ok1 and abs(X(1) - e) < mp.mpf(10)**-30
    d1 = mp.diff(X, 1); d2 = mp.diff(X, 1, 2)
    P = parts(md_answer(3, "2"))
    md1 = mp.mpf(str(sp.N(sp.sympify(P["x1"]), 40))); md2 = mp.mpf(str(sp.N(sp.sympify(P["x2"]), 40)))
    ok2 = abs(d1 - md1) < 1e-15 and abs(d2 - md2) < 1e-12
    # (3-3) [증명-수치확인] x''(t) > 0 격자
    grid = [t0 + mp.mpf("0.001")] + [t0 + mp.mpf(k) / 10 for k in range(1, 200)]
    ok3 = all(mp.diff(X, tt, 2) > 0 for tt in grid)
    report(3, ok1 and ok2 and ok3, f"[증명-수치확인] min g={float(g(t0)):.6f} x'(1)={float(d1):.12f} x''(1)={float(d2):.12f} x''>0 격자 {len(grid)}점")


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
