#!/usr/bin/env python3
"""HIU-G26S (홍익대 2026학년도 수시 논술 자연계열 세종) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "HIU-G26S"
WIDTH = 2
RESULTS = []
t = sp.symbols("t", real=True)
x, y = sp.symbols("x y", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/HIU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def kv(s: str) -> dict:
    """'a = 1/2; b = sqrt(3)/2' 형식을 dict 로."""
    out = {}
    for seg in s.split(";"):
        k, v = seg.split("=", 1) if seg.count("=") == 1 or ":" not in seg else seg.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    al = sp.symbols("alpha")
    a = [6, 4, 3]
    sol = sp.solve((a[1] + al)**2 - (a[0] + al) * (a[2] + al), al)
    res = []
    for s in sol:
        b1, b2 = a[0] + s, a[1] + s
        r = b2 / b1
        res.append(b1 * r**6 - s)
    ok = len(res) == 1 and sp.Rational(md_answer(1, "1")) == res[0]
    report(1, ok, f"alpha={sol} a7={res}")


def p02():
    p, q, r = sp.symbols("p q r")
    f = -x**2 + q * x + r                       # (가): 차수 2, 최고차항 -1 (x^2 f(1/x) -> 최고차 계수)
    ok_a = sp.limit(x**2 * f.subs(x, 1 / x), x, 0) == -1
    s1 = sp.solve(f.subs(x, -1), r)[0]          # (나): 분모 -> 0 이므로 f(-1)=0
    f1 = f.subs(r, s1)
    sq = sp.solve(sp.limit(f1 / (x + 1), x, -1) - f1.subs(x, 1), q)[0]
    fx = sp.expand(f1.subs(q, sq))
    ok_b = sp.limit(fx / (x + 1), x, -1) == fx.subs(x, 1)
    ok = ok_a and ok_b and fx.subs(x, 2) == sp.sympify(md_answer(2, "1"))
    report(2, ok, f"f={fx} f(2)={fx.subs(x, 2)}")


def p03():
    sols = sp.solve(3 * sp.tan(x)**2 - 8 * sp.tan(x) - 3, x)
    vals = {sp.nsimplify(sp.simplify(sp.cos(s))) for s in sols if -sp.pi / 2 < s < sp.pi / 2}
    md = {sp.nsimplify(sp.sympify(v)) for v in md_answer(3, "1").strip("{} ").split(",")}
    ok = len(vals) == 2 and all(any(sp.simplify(u - w) == 0 for w in md) for u in vals)
    report(3, ok, f"cos x = {vals}")


def p04():
    good = []
    for a in range(-200, 201):
        fm = lambda v: -v + abs(a)
        fp = lambda v: v**2 + 3 * v + a
        if fm(0) == fp(0):                      # 0 에서 연속이면 제외 (다른 점은 다항식이라 연속)
            continue
        roots = []
        if abs(a) < 0:                          # x<0 부분: -x+|a|=0 의 근 x=|a| 가 x<0 이어야 함(불가능)
            roots.append(abs(a))
        for rt in sp.solve(x**2 + 3 * x + a, x):
            if rt.is_real and rt >= 0:
                roots.append(rt)
        if all(rt < 5 for rt in roots):
            good.append(a)
    d = {k.strip(): int(v) for k, v in (s.split("=") for s in md_answer(4, "1").split(";"))}
    ok = max(good) == d["max"] and min(good) == d["min"]
    report(4, ok, f"integer a in [{min(good)},{max(good)}], count={len(good)}")


def p05():
    a = sp.symbols("a", real=True)
    import mpmath as mp
    # x2 = x1 + 1 이 두 교점: 2^{x1-a}-6a^2=x1, 2^{x1+1-a}-6a^2=x1+1
    x1 = sp.symbols("x1", real=True)
    u = sp.symbols("u", real=True)                 # u = x1 - a
    diff_eq = sp.expand((2**(u + 1) - 6 * a**2 - (u + a) - 1) - (2**u - 6 * a**2 - (u + a)))   # ㉡-㉠
    us = sp.solve(sp.Eq(diff_eq, 0), u)            # 2^u = 1
    avals = sorted({r for uu in us for r in sp.solve(sp.Eq(2**uu - 6 * a**2, uu + a), a)})
    # 교점이 정확히 2개인지 수치 확인
    ok_cnt = True
    for av in avals:
        g = lambda t: 2**(t - float(av)) - 6 * float(av)**2 - t
        grid = [i / 100 for i in range(-2000, 2000)]
        cnt = sum(1 for u, v in zip(grid, grid[1:]) if g(u) == 0 or g(u) * g(v) < 0)
        ok_cnt &= cnt == 2
    md = sorted(sp.Rational(v) for v in md_answer(5, "1").strip("{} ").split(","))
    ok = ok_cnt and avals == md
    report(5, ok, f"a={avals} two-intersections={ok_cnt}")


def p06():
    a, b, c, al = sp.symbols("a b c alpha")
    f = x**3 + a * x**2 + b * x + c
    g = 3 * x + 1
    alp = sp.symbols("alp", positive=True)
    co = sp.solve(sp.Poly(sp.expand(f - g - x * (x - alp) * (x + alp)), x).coeffs(), [a, b, c], dict=True)[0]
    h = sp.expand((f - g).subs(co))
    area = sp.integrate(h, (x, -alp, 0)) + sp.integrate(-h, (x, 0, alp))
    av = [s for s in sp.solve(area - 8, alp) if s > 0]
    fx = f.subs(co).subs(alp, av[0])
    ok = len(av) == 1 and fx.subs(x, 3) == sp.Integer(md_answer(6, "1")) and h.subs(alp, av[0]).subs(x, 1) < 0
    report(6, ok, f"area={sp.simplify(area)} alpha={av} f={sp.expand(fx)} f(3)={fx.subs(x, 3)}")


def p07():
    # |f| 의 극값(서로 다른 값) 개수를 a 의 후보(경계값·구간 표본)에 대해 직접 계산
    import numpy as np
    def nvals(a):
        F = lambda v: abs(v**3 - 3 * v**2 + a)
        xs = np.linspace(-6, 8, 280001)
        ys = F(xs)
        vals = []
        for i in range(1, len(xs) - 1):
            if (ys[i] > ys[i - 1] and ys[i] >= ys[i + 1]) or (ys[i] < ys[i - 1] and ys[i] <= ys[i + 1]):
                vals.append(round(float(ys[i]), 3))
        # 극값 후보는 f'=0 인 점과 f=0 인 점 → 정확값으로 보정
        exact = {abs(a), abs(a - 4)}
        roots = [r for r in np.roots([1, -3, 0, a]) if abs(r.imag) < 1e-9]
        if roots:
            exact.add(0)
        approx = set(vals)
        return len({min(exact, key=lambda e: abs(e - v)) for v in approx})
    cands = [-3, -1, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 7]
    sel = [c for c in cands if nvals(c) == 2]
    md = sorted(int(v) for v in md_answer(7, "1").strip("{} ").split(","))
    ok = sel == md
    # 참고: 극값을 갖는 점의 개수로 세면
    report(7, ok, f"distinct-extreme-value count==2 at a={sel}")


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
