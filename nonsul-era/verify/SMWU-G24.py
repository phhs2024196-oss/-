#!/usr/bin/env python3
"""SMWU-G24 (숙명여대 2024학년도 논술우수자전형 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import random
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "SMWU-G24"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)


def md_answer(no: int, sub: str):
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SMWU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    # [증명-수치확인] 위로 볼록(f''<0) 함수들에서 (1-1) f(a)=f(b)=0 → f>=0, (1-2) 사다리꼴 부등식
    a, b = sp.symbols("a b", real=True)
    fa, fb = sp.symbols("fa fb", real=True)
    h = (fb - fa) / (b - a) * (x - a) + fa
    ok_h = sp.simplify(sp.integrate(h, (x, a, b)) - sp.Rational(1, 2) * (b - a) * (fa + fb)) == 0
    ok_F = sp.diff(sp.Function("f")(x) - h, x, 2) == sp.diff(sp.Function("f")(x), x, 2)
    # 제시문 예시: sqrt(x) - x 의 이계도함수
    ok_ex = sp.simplify(sp.diff(sp.sqrt(x) - x, x, 2) + 1 / (4 * x * sp.sqrt(x))) == 0
    fams = [(-x**2 + 3 * x, -1, 5), (sp.sin(x), 0.3, 3.0), (sp.log(x), 0.5, 7), (sp.sqrt(x), 0.1, 4),
            (-sp.exp(x), -2, 1), (-x**4 + x, -1, 1.2)]
    ok1 = ok2 = True
    random.seed(1)
    for f, lo, hi in fams:
        fl = sp.lambdify(x, f, "mpmath")
        f2 = sp.lambdify(x, sp.diff(f, x, 2), "mpmath")
        for _ in range(20):
            A, B = sorted(random.uniform(lo, hi) for _ in range(2))
            if B - A < 1e-3 or not all(f2(A + (B - A) * i / 50) < 0 for i in range(1, 50)):
                continue
            # (1-1): F = f - chord, F(A)=F(B)=0, F''<0 → F>=0 (격자 1000점)
            chord = lambda t: (fl(B) - fl(A)) / (B - A) * (t - A) + fl(A)
            ok1 &= all(fl(A + (B - A) * i / 1000) - chord(A + (B - A) * i / 1000) >= -1e-12 for i in range(1001))
            # (1-2)
            ok2 &= (B - A) * (fl(A) + fl(B)) <= 2 * mp.quad(fl, [A, B]) + 1e-12
    ok_md = md_answer(1, "1") is None and md_answer(1, "2") is None
    report(1, ok_h and ok_F and ok_ex and ok1 and ok2 and ok_md,
           "[증명-수치확인] ∫chord=(b-a)(f(a)+f(b))/2, 볼록함수 6종×무작위 구간에서 f-chord>=0·사다리꼴 부등식")


def p02():
    # (2-1) 주기 1 인 여러 g 에 대해 A=∫_0^2 g h, ∫_{2n}^{2n+2} g h, ∫_0^1 g 를 직접 계산
    tri = lambda t: (t % 2) if (t % 2) < 1 else 2 - (t % 2)
    gs = [lambda t: 2 + mp.cos(2 * mp.pi * t), lambda t: mp.exp(mp.sin(2 * mp.pi * t)),
          lambda t: 1 + mp.sin(2 * mp.pi * t)**2 + 0.3 * mp.cos(6 * mp.pi * t), lambda t: 5 + 0 * t]
    md1 = dict(seg.split("=") for seg in md_answer(2, "1").replace(" ", "").split(";"))
    Asym = sp.Symbol("A")
    ok1 = True
    worst = 0
    for g in gs:
        A = mp.quad(lambda t: g(t) * tri(t), [0, 1, 2])
        for n in (1, 2, 5):
            I1 = mp.quad(lambda t: g(t) * tri(t), [2 * n, 2 * n + 1, 2 * n + 2])
            e = abs(I1 - float(sp.sympify(md1["I1"]).subs(Asym, A)))
            worst = max(worst, e)
            ok1 &= e < 1e-9
        I2 = mp.quad(g, [0, 1])
        e = abs(I2 - float(sp.sympify(md1["I2"]).subs(Asym, A)))
        worst = max(worst, e)
        ok1 &= e < 1e-9
    # (2-2) f>0, f(x)f(-x)=1 인 여러 f 로 수치 적분
    fs = [lambda t: mp.e**t, lambda t: mp.e**(t**3 + 2 * t), lambda t: mp.e**(mp.sin(t)),
          lambda t: (1 + t**2 + t) / (1 + t**2 - t)]
    vals = [mp.quad(lambda t: 1 / (f(t)**4 + 1), [-1, 0, 1]) for f in fs]
    ok_cond = all(abs(f(0.37) * f(-0.37) - 1) < 1e-12 for f in fs)
    ok2 = ok_cond and all(abs(v - float(sp.sympify(md_answer(2, "2")))) < 1e-12 for v in vals)
    # 제시문 <라> 값 확인
    ok_la = sp.simplify(sp.integrate(1 / (sp.exp(x) + 1), (x, -1, 1)) - 1) == 0
    report(2, ok1 and ok2 and ok_la, f"(2-1) max err={float(worst):.1e}, (2-2) values={[float(v) for v in vals]}")


def p03():
    # (3-1) b=1..3000 에서 g'(x)=3x^2-2(b+1)x+b 의 두 근이 모두 정수인 b 가 없음
    b = sp.symbols("b", positive=True, integer=True)
    gp = sp.expand(sp.diff(x * (x - 1) * (x - b), x))
    ok_gp = sp.expand(gp - (3 * x**2 - 2 * (b + 1) * x + b)) == 0
    bad = []
    for bv in range(1, 3001):
        D = (bv + 1)**2 - 3 * bv                       # 판별식/4
        r = sp.integer_nthroot(D, 2)
        if r[1] and ((bv + 1 + r[0]) % 3 == 0 and (bv + 1 - r[0]) % 3 == 0):
            bad.append(bv)
    ok1 = ok_gp and not bad and md_answer(3, "1") is None
    # (3-2) 명제 ②: F=k(x-r)(x-s) → F'=0 의 근 (r+s)/2
    k, r_, s_ = sp.symbols("k r s", nonzero=True)
    ok_prop = sp.solve(sp.diff(k * (x - r_) * (x - s_), x), x) == [(r_ + s_) / 2]
    # u=1 → b=1/2, v^2=(1-4a)/4 → a=1/4-v^2. v=1..4 에서 f, f', f'' 의 근이 모두 유리수인지 직접 판정
    found = {}
    for v in range(1, 5):
        bv = 1 - sp.Rational(1, 2)
        av = sp.Rational(1, 4) - v**2
        f = (x**2 + x + av) * (x - bv)
        def all_rational(e):
            rts = sp.Poly(sp.expand(e), x).all_roots()
            return all(rt.is_rational for rt in rts)
        if all(all_rational(sp.diff(f, x, d)) for d in (0, 1, 2)):
            found[v] = sp.expand(f)
    md = dict(seg.split(":", 1) for seg in md_answer(3, "2").replace(" ", "").split(";"))
    md = {int(key.split("=")[1]): sp.expand(sp.sympify(val, locals={"x": x})) for key, val in md.items()}
    ok2 = ok_prop and set(found) == set(md) and all(sp.expand(found[v] - md[v]) == 0 for v in found)
    report(3, ok1 and ok2, f"(3-1) b<=3000 반례 {len(bad)}개, (3-2) v={sorted(found)}")


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
