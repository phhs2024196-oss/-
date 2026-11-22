#!/usr/bin/env python3
"""HYU-G23C (한양대 2023학년도 논술 자연계열 오후2) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 설정으로부터 독립 재계산 후 YAML answer.value 와 비교.
증명 소문항은 핵심 등식을 정확한 정수 연산으로 전수 확인 [증명-수치확인]."""
import re
import sys
from math import comb

import mpmath as mp
import sympy as sp

PREFIX = "HYU-G23C"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HYU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def C(m, k):
    return comb(m, k) if 0 <= k <= m else 0


def a(m):
    return sum(C(3 * m, 3 * k) for k in range(m + 1))


def p01():
    # (1) [증명-수치확인] 1≤k≤n≤60 전수
    ok1 = all(C(3 * m + 3, 3 * k) == C(3 * m, 3 * k - 3) + 3 * C(3 * m, 3 * k - 2) + 3 * C(3 * m, 3 * k - 1) + C(3 * m, 3 * k)
              for m in range(1, 61) for k in range(1, m + 1))
    # (2) [증명-수치확인] a_{n+1} = 2 + Σ_{k=1}^n C(3n+3,3k) 와 a_n + a_{n+1} = 3·2^{3n}, n=1..120
    ok2 = all(a(m + 1) == 2 + sum(C(3 * m + 3, 3 * k) for k in range(1, m + 1)) and a(m) + a(m + 1) == 3 * 2 ** (3 * m)
              for m in range(1, 121))
    # (3) 직접 정수 계산
    v = sum(C(300, 3 * k - 1) for k in range(1, 101))
    ok3 = sp.Integer(v) == sp.sympify(md_answer(1, "3"))
    report(1, ok1 and ok2 and ok3, f"(1)(2) 전수 확인 {ok1},{ok2} [증명-수치확인] (3) Σ={str(v)[:12]}…({len(str(v))}자리) 일치 {ok3}")


def p02():
    # (1) α 표본마다 (f∘f)(x)=1/e 의 실근 개수를 직접 센다:
    #     바깥: f(u)=1/e 의 해 u 를 수치로 모두 찾고(최댓값 접점 포함), 안쪽: f(x)=u 의 실근 수를 부호변화로 센다.
    mp.mp.dps = 40
    E = mp.e
    md_set = sp.sympify(md_answer(2, "1"))

    def roots_f_eq(alpha, c):
        """f(x)=c, f(x)=ln(x+α)/(x+α) 의 실근 개수. 변수 w=x+α>0, h(w)=ln w / w."""
        h = lambda w: mp.log(w) / w - c
        ws = [mp.mpf(10) ** (k / 200) for k in range(-1200, 2401)]   # 1e-6 ~ 1e12 로그 격자
        cnt = 0
        for w1, w2 in zip(ws, ws[1:]):
            v1, v2 = h(w1), h(w2)
            if v1 == 0 or v1 * v2 < 0:
                cnt += 1
        # 최댓값 지점 접근 처리: c 가 최댓값(1/e)과 같으면 w=e 에서 접함
        if abs(c - 1 / E) < mp.mpf("1e-30"):
            cnt = 1
        return cnt

    def count_ff(alpha):
        # f(u) = 1/e ⟺ ln w / w = 1/e (w=u+α) — 최댓값이므로 미분 0 점 w=e 하나뿐임을 수치로 확인
        wstar = mp.findroot(lambda w: 1 - mp.log(w), 2)
        assert abs(mp.log(wstar) / wstar - 1 / E) < mp.mpf("1e-30")
        u = wstar - alpha
        return roots_f_eq(alpha, u)

    lo, hi = float(E - 1 / E), float(E)
    samples = [lo - 0.5, lo - 0.01, lo + 0.01, (lo + hi) / 2, hi - 0.01, hi + 0.01, hi + 1, 0.0, -2.0]
    ok1 = all((count_ff(mp.mpf(al)) == 2) == bool(md_set.contains(sp.Float(al))) for al in samples)
    # (2) a=f'(-1)>0, b=-f'(1)>0, a+b=23 인 여러 경우에 기하로 직접 계산
    S = sp.symbols("S")
    md2 = sp.sympify(md_answer(2, "2"), locals={"S": S})
    ok2 = True
    for av in (sp.Rational(1, 3), 2, 11, sp.Rational(45, 2)):
        bv = 23 - av
        X = sp.symbols("X")
        px = sp.solve(sp.Eq(av * (X + 1), -bv * (X - 1)), X)[0]
        P = sp.Matrix([px, av * (px + 1)])
        A_, B_ = sp.Matrix([-1, 0]), sp.Matrix([1, 0])
        u, v = A_ - P, B_ - P
        cosv = u.dot(v) / sp.sqrt(u.dot(u) * v.dot(v))
        cot = sp.simplify(cosv / sp.sqrt(1 - cosv**2))
        Sv = sp.Rational(1, 2) * 2 * P[1]
        ok2 &= sp.simplify(cot - md2.subs(S, Sv)) == 0
    # (3) 넓이 f(t) 를 직접 적분(쐐기 ∩ 원판)해 수치미분과 비교
    mp.mp.dps = 30
    s3 = mp.sqrt(3)

    def area(tt):
        # A(0,1), 쐐기: y ≤ 1 - √3|x| (A 아래, 두 변 사이), 원: (x-t)^2+(y-1)^2 ≤ 1
        hfun = lambda xx: max(mp.mpf(0), mp.sqrt(max(mp.mpf(0), 1 - (xx - tt) ** 2)) - s3 * abs(xx))
        xl = (tt - mp.sqrt(4 - 3 * tt**2)) / 4
        xr = (tt + mp.sqrt(4 - 3 * tt**2)) / 4
        return mp.quad(hfun, [xl, 0, xr])

    tsym = sp.symbols("t")
    md3 = sp.lambdify(tsym, sp.sympify(md_answer(2, "3"), locals={"t": tsym}), "mpmath")
    errs = [abs(mp.diff(area, tv) - md3(tv)) for tv in (mp.mpf("0.2"), mp.mpf("0.5"), mp.mpf("0.8"))]
    ok3 = max(errs) < 1e-9
    report(2, ok1 and ok2 and ok3, f"[{ok1},{ok2},{ok3}] (1)α 표본 {len(samples)}개 근 개수 일치 (2)a∈{{1/3,2,11,45/2}} (3)max|f'-md|={mp.nstr(max(errs), 3)}")


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
