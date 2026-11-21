#!/usr/bin/env python3
"""KGU-G26A (경기대 2026학년도 수리 A형, 문항 1·2) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산."""
import re
import sys

import sympy as sp

PREFIX = "KGU-G26A"
WIDTH = 2
RESULTS = []


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sid):
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/KGU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    sub = next(s for s in d["subparts"] if str(s["id"]) == str(sid))
    return sub["answer"]["value"]


def Omega(m):
    return sum(sp.factorint(m).values())


def p01():
    """(1) 10000 이하 210의 배수 전수 탐색, (2) 여러 (a,b) 실례로 합을 직접 계산."""
    cnt = sum(1 for m in range(210, 10001, 210) if Omega(m) == 6)
    a1 = sp.sympify(load_answer(1, "1"))
    ok1 = cnt == a1
    # (2) f(a)=5, f(b)=3 인 여러 a, b
    pairs = [(32, 8), (2 * 3 * 5 * 7 * 11, 27), (243, 2 * 3 * 5), (7 ** 5, 11 ** 3), (2 ** 3 * 3 * 5, 4 * 13)]
    sums = set()
    for a, b in pairs:
        assert Omega(a) == 5 and Omega(b) == 3
        sums.add(sum(Omega(n * a * b ** n) for n in range(1, 11)))
    a2 = sp.sympify(load_answer(1, "2"))
    ok2 = len(sums) == 1 and sums.pop() == a2
    report(1, ok1 and ok2, f"(1) 전수 개수={cnt}, md={a1}; (2) 실례 {len(pairs)}쌍 합 일치 및 md={a2}: {ok2}")


def p02():
    """(1) 일반 삼차식에 조건 적용·g(0)=4π 로 a 결정, (2) 평행 접선 사이 거리로 g(m) 재계산 후 극한 확인."""
    x, m = sp.symbols("x m", real=True)
    a, b, c, d = sp.symbols("a b c d", real=True)
    f = a * x ** 3 + b * x ** 2 + c * x + d
    sol = sp.solve(sp.Poly(sp.expand(f + f.subs(x, -x) - 2), x).coeffs(), [b, d], dict=True)[0]
    f1 = f.subs(sol)
    # 접선 개수 조건: f'(x)=m 의 실근 개수가 m>-1:2, m=-1:1, m<-1:0 → a>0 이고 최솟값 c=-1
    fp = sp.diff(f1, x)
    cval = sp.solve(sp.Eq(fp.subs(x, 0), -1), c)[0]   # a>0 일 때 f' 최솟값 = f'(0) = c
    f2 = f1.subs(c, cval)
    ap = sp.symbols("ap", positive=True)
    f2 = f2.subs(a, ap)

    def area(mm, F):
        roots = sp.solve(sp.Eq(sp.diff(F, x), mm), x)
        ints = [sp.simplify(F.subs(x, r) - mm * r) for r in roots]
        dist = sp.Abs(ints[0] - ints[1]) / sp.sqrt(mm ** 2 + 1)
        return sp.simplify(sp.pi * (dist / 2) ** 2)

    asol = sp.solve(sp.Eq(area(0, f2), 4 * sp.pi), ap)
    fF = sp.expand(f2.subs(ap, asol[0]))
    md1 = sp.sympify(load_answer(2, "1"), locals={"x": x})
    # 개수 조건 수치 점검
    cnt_ok = all(len([r for r in sp.solve(sp.Eq(sp.diff(fF, x), mv), x) if r.is_real]) == want
                 for mv, want in [(-2, 0), (-1, 1), (0, 2), (5, 2)])
    ok1 = len(asol) == 1 and sp.simplify(fF - md1) == 0 and cnt_ok
    mp_ = sp.symbols("mp", positive=True)   # m>-1 대신 m>0 구간에서 기호 계산(극한용)
    g = sp.simplify(area(mp_, fF))
    md2 = sp.sympify(load_answer(2, "2"), locals={"m": mp_})
    q, r = sp.div(sp.numer(sp.together(g)), sp.denom(sp.together(g)), mp_)
    ok2 = md2.is_polynomial(mp_) and sp.limit(g - md2, mp_, sp.oo) == 0 and sp.simplify(q - md2) == 0
    # 수치로 m=-0.5, 3 에서 g 재확인
    gm = sp.simplify(area(sp.Rational(-1, 2), fF))
    ok3 = sp.simplify(gm - 4 * sp.pi * sp.Rational(1, 2) ** 3 / (sp.Rational(1, 4) + 1)) == 0
    report(2, ok1 and ok2 and ok3, f"(1) f={fF}, md={md1}, 개수조건={cnt_ok}; (2) g={g}, 몫={q}, md={md2}, lim(g-h)=0")


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
