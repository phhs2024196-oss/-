#!/usr/bin/env python3
"""HYU-G22D (한양대 2022학년도 논술 자연계열 의예과(문제 2만)) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "HYU-G22D"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HYU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def parts(s: str) -> dict:
    return dict(seg.split("=", 1) for seg in s.replace(" ", "").split(";"))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p02():
    import mpmath as mp
    # (1) p''(0) 을 n=1..10 에서 직접 계산
    N = sp.symbols("n", integer=True, positive=True)
    md1 = sp.sympify(md_answer(2, "1"), locals={"n": N})
    ok1 = all(sp.diff(sp.Mul(*[1 + k * x for k in range(1, m + 1)]), x, 2).subs(x, 0) == md1.subs(N, m)
              for m in range(1, 11))
    # (2) n∫x^n q dx = ∫_0^n (1-u/n)^n q(1-u/n) du 로 큰 n 수치계산 → c_n 이 md 값에 수렴
    mp.mp.dps = 30
    q = lambda X: mp.e**X + X * mp.log(X + 1) + X**2 * mp.cos(mp.pi * X)**2022
    target = float(sp.sympify(md_answer(2, "2")))
    errs = []
    for nn in (10**6, 10**7, 10**8):
        I = mp.quad(lambda u: (1 - u / nn)**nn * q(1 - u / nn), [0, 1, 5, 20, 60]) / nn
        cn = (nn - 2022) * I
        errs.append(abs(float(cn) - target))
    ok2 = errs[-1] < 1e-4 and errs[0] > errs[1] > errs[2]
    ok2 &= abs(float(q(mp.mpf(1))) - target) < 1e-12
    # (3) 원과 포물선 교점 수(서로 다른 실근 y 개수)를 k 격자에서 직접 세어 k0 판정
    k, y = sp.symbols("k y", real=True)
    circ = (y**2 / 4 - k**2 / 4)**2 + (y - k)**2 - (k**2 / 4 + 1)**2
    r = sp.expand(16 * circ)
    ok3 = sp.expand(r - (y**4 + 2 * (8 - k**2) * y**2 - 32 * k * y + 8 * (k**2 - 2))) == 0
    ok3 &= sp.factor(sp.diff(r, y)) == sp.factor(4 * (y - k) * (y**2 + k * y + 8))
    # B(-1,k), F(1,0) 이 원 위
    ok3 &= sp.simplify(((-1 - k**2 / 4)**2 + 0 - (k**2 / 4 + 1)**2)) == 0
    import numpy as np
    def nroots(kv):
        cs = [float(cf.subs(k, kv)) for cf in sp.Poly(r, y).all_coeffs()]
        rts = np.roots(cs)
        return len({round(z.real, 7) for z in rts if abs(z.imag) < 1e-7})
    k0 = float(sp.sympify(md_answer(2, "3")))
    below = [nroots(kv) for kv in np.linspace(0, k0 - 1e-3, 300)]
    above = [nroots(kv) for kv in np.linspace(k0 + 1e-3, 3 * k0, 300)]
    ok3 &= all(v == 2 for v in below) and all(v == 4 for v in above)
    # 공식 해설 핵심 등식 r(c) = -2 h(k/√2)  (k>4√2, c=(-k-sqrt(k^2-32))/2)
    X = sp.symbols("X", positive=True)
    hX = X * (X**2 - 16) * (X + sp.sqrt(X**2 - 16)) - 8 * (2 * X**2 - 5)
    for kv in (6, 7, 5 * sp.sqrt(2), 9, 15):
        cv = (-kv - sp.sqrt(kv**2 - 32)) / 2
        ok3 &= abs(float(r.subs({k: kv, y: cv}) + 2 * hX.subs(X, kv / sp.sqrt(2)))) < 1e-9
    # 판별식으로 경계값 독립 계산
    disc = sp.discriminant(r, y)
    kroots = [z for z in sp.solve(sp.factor(disc), k) if z.is_real and z > 0]
    ok3 &= any(abs(float(sp.re(z)) - k0) < 1e-9 for z in kroots)
    report(2, ok1 and ok2 and ok3, f"cn errs={['%.2e' % e for e in errs]} k-roots of disc={[round(float(sp.re(z)), 6) for z in kroots]}")


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
