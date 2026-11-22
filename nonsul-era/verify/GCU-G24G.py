#!/usr/bin/env python3
"""가천대 2024학년도 논술전형 자연G 수학(문제 7~15) — sympy 검산. 정답 하드코딩 비교 금지: 문제 조건에서 독립 재계산 후 YAML answer 와 대조."""
import pathlib
import re
import sys

import sympy as sp
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
PREFIX = "GCU-G24G"
WIDTH = 2
RESULTS = []
x, t, a, b, k, n, r = sp.symbols("x t a b k n r", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def ans(seq: int) -> str:
    """YAML 의 answer.value (문자열)."""
    it = yaml.safe_load((ROOT / "items/GCU/PAST" / f"{PREFIX}-{seq:02d}.yaml").read_text(encoding="utf-8"))
    return str(it["subparts"][0]["answer"]["value"])


def num(s: str):
    return sp.nsimplify(sp.sympify(s))


def nums(s: str):
    """답 문자열에서 동그라미 번호를 뺀 수들을 차례로."""
    s = re.sub(r"[①②③④]", " ", s)
    return [sp.Rational(v) for v in re.findall(r"-?\d+(?:/\d+)?", s)]


def p07():
    xs = [v for v in range(-100, 101) if -v > 0 and -v != 1 and -2 * v**2 - 7 * v + 15 > 0]
    report(7, sum(xs) == num(ans(7)), f"정수 x={xs}, 합={sum(xs)}")


def p08():
    f = 2 * sp.sin(3 * sp.pi / 2 + x) * sp.cos(x + sp.pi) + sp.sin(sp.pi - x) + 1
    g = -2 * sp.sin(x)**2 + sp.sin(x) + 3
    same = sp.simplify(sp.expand_trig(f - g)) == 0
    q = -2 * t**2 + t + 3
    cands = [q.subs(t, c) for c in [-1, 1] + [s for s in sp.solve(sp.diff(q, t), t) if -1 <= s <= 1]]
    M, m = max(cands), min(cands)
    report(8, same and M + m == num(ans(8)), f"f=-2sin²x+sinx+3 확인={same}, M={M}, m={m}, M+m={M + m}")


def p09():
    def seq(a2, N=30):
        s = [None, 4, a2]
        for i in range(1, N - 1):
            s.append(s[i + 1] + s[i] if s[i + 1] % 3 else s[i + 1] // 3)
        return s
    hits = [a2 for a2 in range(1, 5000) if (lambda s: s[5] == 5 and s[6] == 9)(seq(a2, 7))]
    s = seq(hits[0])
    cnt = sum(1 for kk in range(1, 31) if s[kk] > s[2])
    got = nums(ans(9))
    ok = len(hits) == 1 and got == [s[7], s[4], s[3], s[2]] + sorted({v for v in s[3:8] if v > s[2]}) + [cnt]
    report(9, ok, f"a2 후보={hits}, a2..a8={s[2:9]}, 개수={cnt}")


def p10():
    fp = sp.diff(x**3 - 2 * a * x**2 + x + 3, x)
    disc = sp.discriminant(fp, x)                              # ≤0 ⇔ f'≥0
    amax = max(sp.solve(sp.Eq(disc, 0), a))
    eps = sp.Rational(1, 1000)
    ok_edge = sp.discriminant(fp.subs(a, amax + eps), x) > 0
    report(10, ok_edge and sp.simplify(amax - num(ans(10))) == 0, f"판별식={sp.factor(disc)}, a 최댓값={amax}")


def p11():
    def pow2(q):
        return all(v > 0 and (v & (v - 1)) == 0 for v in (q.p, q.q))
    ns = [m for m in range(1, 101) if pow2(sp.Rational(7, 3 * m + 17) ** 2)]
    ks = [(m + 1) // 7 for m in ns]
    kmax = max(j for j in range(1, 100) if 7 * j - 1 <= 100)
    got = nums(ans(11))
    report(11, got == [kmax] + ks + [sum(ns)], f"n={ns}, k={ks}, k≤{kmax}, 합={sum(ns)}")


def p12():
    A = sp.Symbol("A", positive=True)
    g = A**2 * t**3 - 4 * A * t**2 + 5 * t - sp.Rational(1, 18)
    crit = [c for c in sp.solve(sp.discriminant(g, t), A) if c.is_positive]

    def count(av):
        P = sp.Poly(sp.expand(x**2 + (av * x**3 - 2 * x)**2 - sp.Rational(1, 18)), x)
        return len(set(P.real_roots()))
    pts = sorted(crit)
    tests = [pts[0] / 2] + [(pts[i] + pts[i + 1]) / 2 for i in range(len(pts) - 1)] + [pts[-1] * 2]
    good = [c for c in pts if count(c) == 4] + [s for s in tests if count(s) == 4]
    got = sorted(nums(ans(12)))
    report(12, sorted(good) == got, f"판별식 0 인 a={pts}, 교점 4개인 a={good}, 구간 표본 교점수={[count(s) for s in tests]}")


def p13():
    v = 3 * t**2 - 18 * t + k
    kv = sp.solve(sp.Eq(1 + sp.integrate(v, (t, 0, 1)), 17), k)[0]
    V = v.subs(k, kv)
    cuts = sorted([0, 3] + [c for c in sp.solve(V, t) if 0 < c < 3])
    dist = sum(abs(sp.integrate(V, (t, cuts[i], cuts[i + 1]))) for i in range(len(cuts) - 1))
    report(13, dist == num(ans(13)), f"k={kv}, 부호 전환 {cuts}, 거리={dist} (변위 {sp.integrate(V, (t, 0, 3))})")


def p14():
    found = set()
    for a1 in range(-30, 31):
        if a1 == 0:
            continue
        for q in range(1, 13):
            for p in range(-24, 25):
                rr = sp.Rational(p, q)
                if rr in (0, 1, -1) or sp.gcd(p, q) != 1:
                    continue
                terms = [a1 * rr**i for i in range(10)]
                Aset = sorted({u**2 for u in terms}, reverse=True)
                Bset = sorted({(-1)**i * u for i, u in enumerate(terms)}, reverse=True)
                if len(Aset) < 2 or len(Bset) < 3:
                    continue
                al1, al2 = Aset[0], Aset[1]
                b1, b2, b3 = Bset[0], Bset[1], Bset[2]
                if b2 == 2 and b1 != b2 and al1 / al2 == (b1 / b2)**2 and (al1 - al2) / (b1 - b2) == 5:
                    found.add((a1, rr, al2 * b3))
    vals = {f[2] for f in found}
    report(14, len(vals) == 1 and vals == {num(ans(14))}, f"(a1, r, α2β3) = {sorted(found)}  [a1∈[-30,30] 정수, r=p/q 전수 탐색]")


def p15():
    T = sp.Symbol("T", positive=True)
    xa = sp.solve(sp.Eq(5 / (x - 3), -T), x)[0]
    xb = sp.solve(sp.Eq(-2 / (x - 3), -T), x)[0]
    xc = sp.solve(sp.Eq(1 / x - 2, T), x)[0]
    xd = sp.solve(sp.Eq(-3 / x - 2, T), x)[0]
    P = [(xa, -T), (xb, -T), (xc, T), (xd, T)]
    area = sp.simplify(sp.Rational(1, 2) * sp.Abs(sum(P[i][0] * P[(i + 1) % 4][1] - P[(i + 1) % 4][0] * P[i][1] for i in range(4))))
    L = sp.limit(area, T, sp.oo)
    report(15, L == num(ans(15)), f"xA={xa}, xB={xb}, xC={xc}, xD={xd}, f(t)={sp.simplify(area)}, 극한={L}")


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
