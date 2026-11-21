#!/usr/bin/env python3
"""숭실대 2025학년도 모의논술고사(자연) 1~4 (SSU-M25-01~04) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import sympy as sp

PREFIX = "SSU-M25"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
a, b, k = sp.symbols("a b k", real=True)
SYM = {"x": x, "t": t, "n": n, "a": a, "b": b, "k": k}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")




def load_answer(seq, sub=0, raw=False):
    """yaml 의 subparts[sub].answer.value 를 읽는다(raw=True 면 문자열 그대로)."""
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/SSU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    v = d["subparts"][sub]["answer"]["value"]
    return v if raw else sp.sympify(v, locals=SYM)


def p01():
    """y(t)=0 에서 착지 시각, 속력을 직접 계산해 a 를 풀고, 거리 적분을 독립 계산."""
    tt = sp.symbols("tt", positive=True)
    aa = sp.symbols("aa", positive=True)
    X = 2 * aa * tt
    Y = -(tt**2 + 4 * aa) ** sp.Rational(3, 2) / 3 + 9
    speed = sp.sqrt(sp.factor(sp.expand(sp.diff(X, tt) ** 2 + sp.diff(Y, tt) ** 2)))   # 근호 안을 인수분해(완전제곱 여부를 sympy 가 판단)
    T = sp.sqrt(9 - 4 * aa)                                   # Y(T)=0 의 양근
    assert sp.simplify(Y.subs(tt, T)) == 0
    sols = [s for s in sp.solve(sp.Eq(sp.simplify(speed.subs(tt, T)), 5), aa) if 0 < s < sp.Rational(9, 4)]
    a0 = sols[0]
    y0 = Y.subs({tt: 0, aa: a0})
    T0 = T.subs(aa, a0)
    dist = sp.integrate(sp.simplify(speed.subs(aa, a0)), (tt, 0, T0))
    import mpmath as mp
    dnum = mp.quad(lambda s: mp.sqrt((2 * a0) ** 2 + (s * mp.sqrt(s**2 + 4 * a0)) ** 2), [0, float(T0)])
    ans1, ans2 = load_answer(1, 0), load_answer(1, 1)
    ok = len(sols) == 1 and a0 == ans1 and y0 > 0 and sp.simplify(dist - ans2) == 0 and abs(dnum - float(ans2)) < 1e-12
    report(1, ok, f"(1) a={sols} (발사 높이 {sp.N(y0,6)}>0), md={ans1}; (2) 거리={dist} (수치 {mp.nstr(dnum,12)}), md={ans2}")


def p02():
    """(1) sympy limit, (2) 큰 x 에서 정적분 수치값의 수렴."""
    u = sp.symbols("u", positive=True)
    deriv = sp.diff(sp.Integral(u * sp.tan(1 / u), (u, 1, x)), x)          # 미분과 적분의 관계(sympy)
    L1 = sp.limit(deriv, x, sp.oo)
    import mpmath as mp
    mp.mp.dps = 40
    F = lambda X: mp.quad(lambda s: s * mp.tan(1 / s), [X, X + 1])
    vals = [F(mp.mpf(10) ** e) for e in (2, 4, 6, 8)]
    ans1, ans2 = load_answer(2, 0), load_answer(2, 1)
    conv = abs(vals[-1] - float(ans2)) < 1e-12 and all(abs(vals[i + 1] - float(ans2)) < abs(vals[i] - float(ans2)) for i in range(3))
    report(2, L1 == ans1 and conv, f"(1) lim x tan(1/x)={L1}, md={ans1}; (2) ∫_x^(x+1) 값 {[mp.nstr(v,15) for v in vals]} → md={ans2}")


def p03():
    """(a,b) 유리 격자에서 (1) 해 존재, (2) 모든 x 에서 성립을 x 격자(+꼭짓점·큰 x)로 직접 판정해 md 조건과 비교."""
    av_grid = [sp.Rational(i, 4) for i in range(-12, 13)]
    bv_grid = [sp.Rational(j, 4) for j in range(-16, 17)]
    xs = [sp.Rational(i, 8) for i in range(-80, 81)] + [sp.Integer(10)**6, -sp.Integer(10)**6]
    c1, c2 = load_answer(3, 0), load_answer(3, 1)
    bad1 = bad2 = 0
    for av in av_grid:
        for bv in bv_grid:
            F1 = lambda X: av * (X - 1) * (X - 3) + bv - 1
            F2 = lambda X: av * (X - 1) * (X - 3) + bv - bv * X
            ex = list(xs) + ([sp.Integer(2)] if av != 0 else [])
            ex2 = list(xs) + ([(4 * av + bv) / (2 * av)] if av != 0 else [])
            exists = any(F1(X) >= 0 for X in ex)
            forall = all(F2(X) >= 0 for X in ex2)
            bad1 += exists != bool(c1.subs({a: av, b: bv}))
            bad2 += forall != bool(c2.subs({a: av, b: bv}))
    report(3, bad1 == 0 and bad2 == 0, f"격자 {len(av_grid)*len(bv_grid)}쌍: (1) 불일치 {bad1}, md={c1}; (2) 불일치 {bad2}, md={c2}")


def p04():
    """k=j/4, φ=πi/12 격자에서 f·g 의 연속(경계 좌우극한)과 ∫_0^π g 가 양의 정수인지 직접 판정."""
    fL = {0: (sp.Rational(-1, 3), 2), sp.pi: (2, sp.Rational(1, 2))}   # 경계점의 (왼쪽 f, 오른쪽 f) — 오른쪽 값이 함숫값
    found = set()
    for j in range(1, 41):
        kv = sp.Rational(j, 4)
        for i in range(24):
            ph = sp.pi * i / 12
            gv = lambda X: 3 * sp.sin(kv * X + ph)
            cont = all(sp.simplify(l * gv(p) - r * gv(p)) == 0 for p, (l, r) in fL.items())
            if not cont:
                continue
            I = sp.simplify(sp.integrate(3 * sp.sin(kv * x + ph), (x, 0, sp.pi)))
            if I.is_integer and I > 0:
                found.add((kv, ph))
    ans = load_answer(4)
    ok = found == set(tuple(p) for p in ans)
    report(4, ok, f"찾은 (k,φ)={sorted(found, key=str)}, md={ans}")


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
