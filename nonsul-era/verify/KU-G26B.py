#!/usr/bin/env python3
"""KU-G26B (고려대 2026학년도 수시 논술 자연계열 오후) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건에서 독립 재계산 후 YAML 정답과 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import datetime
import itertools
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "KU-G26B"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)


def md_answer(no: int, sub: str) -> str:
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/KU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    a, b, c = sp.symbols("a b c", real=True)
    A = sp.symbols("A", positive=True)
    # (나) 두 점 대입
    sol = sp.solve([c - 1, a - b + c - 1], [b, c], dict=True)[0]
    q = (A * x**2 + sol[b].subs(a, A) * x + sol[c]) - x**2
    roots = sp.solve(q, x)
    d = [r for r in roots if sp.simplify(r + 1) != 0][0]                 # 다른 교점
    area = sp.integrate(q, (x, -1, d))                                    # 0<A<1 → q>=0
    yv = sp.simplify((A * x**2 + A * x + 1).subs(x, -sp.Rational(1, 2)))  # 꼭짓점 y
    # n→∞ ⇔ A→1-  (넓이 = (d+1)^3/(6d), d=1/(1-A) 증가)
    l1 = sp.limit(yv, A, 1, "-")
    l2 = sp.limit(d**2 / area, A, 1, "-")
    ok1 = sp.simplify(l1 - sp.sympify(md_answer(1, "1"))) == 0
    ok2 = sp.simplify(l2 - sp.sympify(md_answer(1, "2"))) == 0
    # d_n ≥ √(2n) 이고 1<d (0<A<1) 인 근 선택 확인 + 1-3 수치
    mp.mp.dps = 60
    dd = sp.symbols("dd", positive=True)
    areaD = sp.simplify(area.subs(A, 1 - 1 / dd))
    F = sp.lambdify(dd, areaD, "mpmath")
    def dn(n):
        return mp.findroot(lambda t: F(t) - n, mp.sqrt(6 * n) + 1)
    okc = all(dn(n) >= mp.sqrt(2 * n) and dn(n) > 1 for n in range(20, 60))
    okarea = abs(F(dn(20)) - 20) < mp.mpf(10)**-40
    vals = []
    for k in (6, 8, 10, 12):
        n = mp.mpf(10)**k
        d1, d0 = dn(n + 1), dn(n)
        vals.append(d1 * (d1 - d0))
    target = sp.sympify(md_answer(1, "3"))
    ok3 = abs(vals[-1] - float(target)) < 1e-4 and abs(vals[-1] - target) < abs(vals[0] - target) + 1e-30
    report(1, ok1 and ok2 and ok3 and okc and okarea,
           f"lim y_n={l1} lim d^2/n={l2} d(d+-d)~{mp.nstr(vals[-1], 10)} (n=1e12) area(A)={sp.factor(areaD)}")


def p02():
    b, c, d = sp.symbols("b c d", real=True)
    f = x**3 + b * x**2 + c * x + d
    # (나)
    lim_n = sp.limit((f + f.subs(x, -x) - 2 * f.subs(x, 0)) / x**2, x, 0)
    bsol = sp.solve(lim_n, b)
    f1 = f.subs(b, bsol[0])
    # (가) b=0 → f'=3x^2+c>0 ∀x ⇔ c>0
    # (다) 극한 존재 → f(0)=2g(0) ⇔ g(0)=d/2 ⇔ f(d/2)=0
    eq = sp.factor(f1.subs(x, d / 2))
    dsols = [s for s in sp.solve(eq, d) if s.is_real]                       # c>0 이면 d^2+4c+8>0
    c_pos = sp.symbols("cp", positive=True)
    dsols_pos = [s for s in sp.solve(eq.subs(c, c_pos), d) if s.is_real]
    dval = dsols_pos[0] if len(dsols_pos) == 1 else None
    v21 = sp.simplify(f1.subs(x, d / 2).subs(d, dval))
    ok1 = dval == 0 and sp.simplify(v21 - sp.sympify(md_answer(2, "1"))) == 0
    f2 = f1.subs(d, dval)
    # f'(0) - 2/f'(g(0)) = 1, g(0)=0
    csol = [s for s in sp.solve(sp.Eq(sp.diff(f2, x).subs(x, 0) - 2 / sp.diff(f2, x).subs(x, 0), 1), c) if s > 0]
    fx = sp.expand(f2.subs(c, csol[0]))
    ok2 = len(csol) == 1 and sp.simplify(fx - sp.sympify(md_answer(2, "2"), locals={"x": x})) == 0
    # (다) 수치 재확인
    mp.mp.dps = 40
    fl = sp.lambdify(x, fx, "mpmath")
    g = lambda y: mp.findroot(lambda t: fl(t) - y, y / 3)
    h = mp.mpf("1e-12")
    okd = abs((fl(h) - 2 * g(h)) / h - 1) < 1e-9
    # 2-3: 원 적분을 직접 수치 적분
    I = mp.quad(lambda t: t * (fl(t) + 1) * g(t), [-3, 0, 3])
    ok3 = abs(I - mp.mpf(sp.Rational(md_answer(2, "3")).p) / sp.Rational(md_answer(2, "3")).q) < 1e-25
    report(2, ok1 and ok2 and ok3 and okd, f"b={bsol} d={dval} f={fx} I={mp.nstr(I, 15)}")


def p03():
    mp.mp.dps = 30
    s32 = mp.sqrt(2)
    def extremes(bv, cv, av):
        A, B = bv / 2, mp.sqrt(bv**2 / 4 - cv**2)
        dist = lambda th: (A * mp.cos(th) / s32 - B * mp.sin(th) + av) / mp.sqrt(1.5)   # 부호 거리
        ths = [2 * mp.pi * k / 4000 for k in range(4000)]
        ds = [dist(t) for t in ths]
        assert all(v > 0 for v in ds) or all(v < 0 for v in ds)                         # 만나지 않음
        ad = [abs(v) for v in ds]
        out = []
        for idx in (ad.index(min(ad)), ad.index(max(ad))):
            th = mp.findroot(lambda t: mp.diff(dist, t), ths[idx])
            out.append(A * mp.cos(th))
        return out
    ok1 = True
    tgt1 = float(sp.sympify(md_answer(3, "1")))
    for bv, cv, av in [(4, 1, 5), (6, 2.5, -7), (3, 0, 4), (10, 4.9, 12)]:
        x1, x2 = extremes(mp.mpf(bv), mp.mpf(cv), mp.mpf(av))
        ok1 &= abs(x1 + x2 - tgt1) < 1e-15
    # 3-2: x1*x2 = -b^2/12 을 만족하는 c 를 수치로 찾고 길이 계산
    ok2 = True
    tgt2 = float(sp.sympify(md_answer(3, "2")))
    for bv in (2, 5, 9):
        bv = mp.mpf(bv)
        prod = lambda cv: (lambda r: r[0] * r[1])(extremes(bv, cv, 3 * bv)) + bv**2 / 12
        cs = [bv / 2 * k / 50 for k in range(0, 50)]
        ok2 &= min(abs(prod(cv)) for cv in cs) < 1e-15 and abs(prod(cs[0])) < 1e-15 \
            and all(abs(prod(cv)) > 1e-6 for cv in cs[1:])          # c=0 만 해
        A, B = bv / 2, bv / 2
        ell = mp.quad(lambda t: mp.sqrt((A * mp.sin(t))**2 + (B * mp.cos(t))**2), [0, 2 * mp.pi])
        ok2 &= abs(ell / bv - tgt2) < 1e-12
    report(3, ok1 and ok2, "x1+x2=0 (4가지 b,c,a), x1x2=-b^2/12 ⇔ c=0, l/b=π (수치)")


def p04():
    mdays = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    good = []
    for y in range(1, 2026):
        for m in range(1, 13):
            for dday in range(1, mdays[m] + 1):
                s = f"{y:04d}{m:02d}{dday:02d}"
                if len(set(s)) == 8:
                    good.append(s)
    firsts = sorted({s[0] for s in good})
    ok1 = "{" + ",".join(firsts) + "}" == md_answer(4, "1").replace(" ", "")
    cnt, first = len(good), min(good)
    exp = md_answer(4, "2").replace(" ", "").split(";")
    ok2 = str(cnt) == exp[0] and f"{first[:4]}-{first[4:6]}-{first[6:]}" == exp[1]
    report(4, ok1 and ok2, f"①∈{firsts} count={cnt} earliest={first} (전수 열거)")


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
