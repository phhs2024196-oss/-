#!/usr/bin/env python3
"""SKKU-G23B (성균관대 2023학년도 수시 논술우수전형 자연계 2교시) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산(전수 열거 포함).
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import math
import re
import sys

import sympy as sp

PREFIX = "SKKU-G23B"
WIDTH = 2
RESULTS = []


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SKKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p01():
    tri = lambda x, y, z: x + y > z  # x<=y<=z
    # (i) 정렬된 세 변 x<=y<=z<=100, 등차(y-x==z-y), 삼각형
    c1 = sum(1 for x in range(1, 101) for y in range(x, 101) for z in range(y, 101) if y - x == z - y and tri(x, y, z))
    # (ii) 등비(y^2==xz)
    c2 = sum(1 for x in range(1, 101) for y in range(x, 101) for z in range(y, 101) if y * y == x * z and tri(x, y, z))
    # (iii) M 의 원소 전수: a,c 정하면 b 결정
    vals = [v for v in range(-100, 101) if v != 0]
    c3 = 0
    for a in vals:
        for c in vals:
            if (a + c) % 2:
                continue
            b = (a + c) // 2
            if b == 0 or abs(b) > 100:
                continue
            if any(u * w != 0 and v * v == u * w for (u, v, w) in itertools.permutations((a, b, c))):
                c3 += 1
    ok = (c1, c2, c3) == tuple(int(md_answer(1, s)) for s in "123")
    report(1, ok, f"AP={c1} GP={c2} M-count={c3}")


def p02():
    import numpy as np
    xs = np.linspace(1e-6, math.pi / 4 - 1e-6, 200001)
    def has_root(F):
        v = F(xs)
        return bool(np.any(np.sign(v[:-1]) != np.sign(v[1:])) or np.any(np.abs(v) < 1e-12))
    C = lambda k: (lambda x: np.cos(x + k * math.pi / 2))
    n1 = sum(1 for n in range(1, 2024) if has_root(lambda x: 4 * C(n % 4)(x) - 1))
    pairs = 0
    for m in range(1, 24):
        for n in range(1, 24):
            if has_root(lambda x: 6 * C(m % 4)(x)**2 + C(n % 4)(x) - 5):
                pairs += 1
    n3 = sum(1 for n in range(1, 2024) if has_root(lambda x: 8 * C(n % 4)(x)**4 - 7 * C(n % 4)(x)**2 + 3 * C(n % 4)(x) - 1))
    # 기호 확인: (ii) 근 cos x=5/6, sin x=1/3, 1/2 이 구간 안
    ok_sym = all(0 < float(sp.asin(sp.Rational(1, 3))) < math.pi / 4 for _ in [0]) and float(sp.acos(sp.Rational(5, 6))) < math.pi / 4
    ok = ok_sym and (n1, pairs, n3) == tuple(int(md_answer(2, s)) for s in "123")
    report(2, ok, f"(i)={n1} (ii)={pairs} (iii)={n3}")


def p03():
    a, b, c, d, t, h, x = sp.symbols("a b c d t h x", real=True)
    f = a * x**3 + b * x**2 + c * x + d
    ok1 = sp.simplify(sp.integrate(f, (x, 0, 1)) - sp.sympify(md_answer(3, "1"), locals={"a": a, "b": b, "c": c, "d": d})) == 0
    # (ii) 부호 있는 넓이 S(h) 의 h->0+ 극한
    S = sp.integrate(f, (x, t, t + h)) + t * f.subs(x, t) / 2 - (t + h) * f.subs(x, t + h) / 2
    lim = sp.limit(S / h, h, 0, "+")
    md2 = sp.sympify(md_answer(3, "2"), locals={"a": a, "b": b, "d": d, "t": t})
    ok2 = sp.simplify(sp.Abs(lim) - md2) == 0 or all(
        abs(float(abs(lim.subs(vv)) - md2.subs(vv))) < 1e-12
        for vv in [{a: 1, b: 2, c: 3, d: 4, t: 0.7}, {a: 2, b: -1, c: 0, d: 1, t: -1.3}, {a: 0.5, b: 3, c: -2, d: 2, t: 2.1}])
    # (iii)(iv) 조건 (가)~(라)를 만족하는 (a,b,d) 를 직접 풀기: g=2at^3+bt^2-d
    A_, B_, D_, al, be = sp.symbols("A B D alpha beta", real=True)
    g = 2 * A_ * t**3 + B_ * t**2 - D_
    eqs = sp.Poly(g - 2 * A_ * (t - al)**2 * (t - be), t).all_coeffs()
    sol = sp.solve(eqs + [D_ / 2 - 16], [B_, D_, al, be], dict=True)
    found = None
    for s_ in sol:
        gs = g.subs(s_)
        bev = s_[be]
        # A 의 극대점이 t=0 이고 극댓값 16 인지 확인은 아래 수치로
        for Av in sp.solve(sp.Eq(sp.integrate(-gs / 2, (t, 0, bev)) + sp.integrate(gs / 2, (t, bev, 2 * bev)), 38), A_):
            if Av.is_real and Av > 0:
                bv = sp.simplify(bev.subs(A_, Av)); alv = sp.simplify(s_[al].subs(A_, Av))
                if bv.is_real and alv.is_real and alv < bv:
                    found = (alv, bv, Av, sp.simplify(s_[B_].subs(A_, Av)), sp.simplify(s_[D_].subs(A_, Av)))
    alv, bv, Av, Bv, Dv = found
    gnum = sp.lambdify(t, (2 * Av * t**3 + Bv * t**2 - Dv))
    # (가)(나)(다) 수치 확인
    roots = sorted(set(round(float(r), 9) for r in sp.real_roots(sp.Poly(2 * Av * t**3 + Bv * t**2 - Dv, t))))
    import numpy as np
    ts = np.linspace(-6, 6, 120001); As = np.abs(gnum(ts)) / 2
    locmax = [As[i] for i in range(1, len(ts) - 1) if As[i] > As[i - 1] and As[i] >= As[i + 1] and As[i] > 1e-9]
    cond = roots == [float(alv), float(bv)] and len(locmax) == 1 and abs(locmax[0] - 16) < 1e-6
    md3 = dict(seg.split("=") for seg in md_answer(3, "3").replace(" ", "").split(";"))
    ok3 = cond and sp.sympify(md3["alpha"]) == alv and sp.sympify(md3["beta"]) == bv and sp.sympify(md3["a"]) == Av
    cv = sp.solve(sp.integrate(f, (x, 0, 1)).subs({a: Av, b: Bv, d: Dv}) - 23, c)[0]
    md4 = dict(seg.split("=") for seg in md_answer(3, "4").replace(" ", "").split(";"))
    ok4 = all(sp.sympify(md4[k]) == v for k, v in zip("abcd", (Av, Bv, cv, Dv)))
    report(3, ok1 and ok2 and ok3 and ok4, f"A(t)=|{sp.simplify(-2*lim)}|/2 alpha={alv} beta={bv} a={Av} b={Bv} c={cv} d={Dv}")


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
