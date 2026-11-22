#!/usr/bin/env python3
"""서울과학기술대 2021학년도 수시모집 논술전형(자연계열 2차) 문제 1~3 (SNUT-G21B-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산해 YAML answer.value 와 비교."""
import sys

import sympy as sp

PREFIX = "SNUT-G21B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
a, b, k = sp.symbols("a b k", real=True)
SYM = {"x": x, "t": t, "n": n, "a": a, "b": b, "k": k, "pi": sp.pi, "oo": sp.oo}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0, raw=False):
    """yaml 의 subparts[sub].answer.value 를 읽는다(raw=True 면 문자열 그대로)."""
    import pathlib
    import yaml
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/SNUT/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text(encoding="utf-8"))
    v = d["subparts"][sub]["answer"]["value"]
    return v if raw else sp.sympify(v, locals=SYM)


A = sp.Symbol("A")
SYM["A"] = A


def p01():
    """1.1 합을 정의대로 정확히 계산, 1.2 원탁 7좌석 전수, 1.3 θ 격자에서 OPQ 외접원 지름을 좌표로 직접 계산."""
    # 1.1
    S = sum(sp.Rational(-2, 3) ** m * sp.simplify(sp.sin(m * sp.pi / 4) ** 2) for m in range(1, 101))
    ans1 = load_answer(1, 0).subs(A, sp.Rational(2, 3) ** 100)
    ok1 = sp.simplify(S - ans1) == 0
    # 1.2 (A 를 0 번 자리에 고정, 회전 동일시)
    import itertools
    from fractions import Fraction
    dist = lambda i, j: min((i - j) % 7, (j - i) % 7)
    good = tot = 0
    for bi, ci in itertools.permutations(range(1, 7), 2):
        tot += 1
        good += dist(0, bi) < dist(0, ci)
    p12 = Fraction(good, tot)
    ans2 = load_answer(1, 1)
    ok2 = sp.Rational(p12.numerator, p12.denominator) == ans2
    # 1.3
    import numpy as np
    ths = np.linspace(np.pi / 6, np.pi / 2 - 1e-9, 400001)
    th_half = np.tan(ths / 2)
    m2 = (1 + th_half) / (1 - th_half)
    q = 1 / np.cos(ths)
    m1 = np.tan(ths)
    px = m2 * q / (m2 - m1)          # m1 x = m2 (x - q)
    py = m1 * px
    # 삼각형 OPQ: O(0,0), Q(q,0), P(px,py). 지름 = |OQ| / sin(∠OPQ)
    vo = np.stack([-px, -py]); vq = np.stack([q - px, -py])
    cosP = (vo * vq).sum(0) / (np.hypot(*vo) * np.hypot(*vq))
    cosP = np.clip(cosP, -1 + 1e-15, 1 - 1e-15)
    diam = q / np.sqrt(1 - cosP ** 2)
    mn = diam.min()
    ans3 = float(load_answer(1, 2))
    ok3 = abs(mn - ans3) < 1e-6 and ths[diam.argmin()] < np.pi / 6 + 1e-3
    report(1, ok1 and ok2 and ok3, f"[1.1] Σ=3/65(A-1)과 일치 {ok1}, [1.2] 확률={p12}, [1.3] 최소 지름≈{mn:.9f} (θ={ths[diam.argmin()]:.4f}), yaml≈{ans3:.9f}")


def p02():
    """2.1 삼각형 넓이 최대 Q 좌표 → u v w, 2.2 넓이 적분, 2.3 점-직선 거리 최대."""
    be = sp.symbols("be", positive=True)
    bb = -be ** 2 + 4 * be                      # 직선 y=b 가 지나는 점 Q(be, b)
    pxv = 4 - be                                # P 의 x좌표(대칭축 x=2)
    area = sp.Rational(1, 2) * (be - pxv) * bb
    crit = [c for c in sp.solve(sp.diff(area, be), be) if 2 < sp.N(c) < 4]
    qx = sp.radsimp(crit[0])
    u, rest = qx.as_independent(sp.sqrt(3), as_Add=True)
    s2 = sp.nsimplify(rest ** 2)                 # = v^2 w
    m = s2.p * s2.q
    w = 1
    for pr, e in sp.factorint(m).items():
        if e % 2:
            w *= pr
    v = sp.sqrt(s2 / w)
    uvw = sp.simplify(u * v * w)
    # 최대 확인(양 끝에서 0)
    okmax = area.subs(be, crit[0]) > area.subs(be, sp.Rational(5, 2)) and area.subs(be, crit[0]) > area.subs(be, sp.Rational(7, 2))
    ans1 = load_answer(2, 0)
    # 2.2
    xx = sp.symbols("xx", real=True)
    S22 = sp.integrate(-xx ** 2 + 4 * xx - xx, (xx, 0, 3))
    # 2.3 거리 d(x)=|x - y|/sqrt2, 점 (x, -x^2+4x), 0<x<3
    d = sp.Abs(xx - (-xx ** 2 + 4 * xx)) / sp.sqrt(2)
    dmax = sp.maximum(d.subs(sp.Abs(xx - (-xx ** 2 + 4 * xx)), -(xx - (-xx ** 2 + 4 * xx))), xx, sp.Interval.open(0, 3))
    import numpy as np
    xs = np.linspace(0, 3, 3000001)
    dn = np.abs(xs - (-xs ** 2 + 4 * xs)) / np.sqrt(2)
    ok = okmax and u == 2 and v == sp.Rational(2, 3) and w == 3 and sp.simplify(uvw - ans1) == 0 \
        and S22 == load_answer(2, 1) and sp.simplify(dmax - load_answer(2, 2)) == 0 and abs(dn.max() - float(load_answer(2, 2))) < 1e-9
    report(2, ok, f"[2.1] Q_x={qx}→(u,v,w)=({u},{v},{w}), uvw={uvw}; [2.2] 넓이={S22}; [2.3] 최대거리={dmax}")


def p03():
    """3.1 x≥3 자연수 해 탐색, 3.2 실수 해 개수(f 단조성+수치), 3.3 (r,s) 격자의 극한, 3.4 (r,s) 전수."""
    sols = [m for m in range(3, 300) if m ** 2 == 2 ** m]
    ok1 = sols == [4] and sp.Integer(load_answer(3, 0)) in sols
    # 3.2: f(x)=2 ln x - x ln 2 on [3,∞) , ln2=0.7 로도 f'<0 확인, 실수 해 수치 탐색
    import numpy as np
    xs = np.linspace(3, 400, 4000001)
    f = 2 * np.log(xs) - xs * np.log(2)
    sign_changes = int(((f[:-1] > 0) & (f[1:] <= 0)).sum() + ((f[:-1] < 0) & (f[1:] >= 0)).sum())
    fprime_max_07 = max(2 / 3 - 0.7, 2 / 400 - 0.7)       # 2/x - 0.7 는 감소함수, x=3 에서 최대
    ok2 = sign_changes == 1 and fprime_max_07 < 0 and abs(2 * np.log(4) - 4 * np.log(2)) < 1e-12 and load_answer(3, 1) == 1
    # 3.3: 극한이 유한·비영인 (r,s)
    nn = sp.symbols("nn", positive=True)
    pairs = {}
    for r in range(1, 9):
        for s in range(1, 9):
            L = sp.limit((nn * (nn + 1)) ** s / nn ** (r + 1), nn, sp.oo)
            if L.is_finite and L != 0:
                pairs[(r, s)] = L
    rel = load_answer(3, 2, raw=True)
    ok3 = all(2 * s == r + 1 for r, s in pairs) and all(L == 1 for L in pairs.values()) and set(pairs) == {(r, (r + 1) // 2) for r in range(1, 9, 2)} and "2*s = r + 1" in rel and rel.strip().endswith("1")
    # 3.4: r≥2, s 자연수, 모든 n=1..40 에서 성립하는 (r,s) 전수; 그 중 2s=r+1 인 것
    found = []
    for r in range(2, 120):
        for s in range(1, 120):
            if all(sum(k ** r for k in range(1, N + 1)) == (N * (N + 1) // 2) ** s for N in range(1, 41)):
                found.append((r, s))
    rs = sorted({r for r, s in found if 2 * s == r + 1})
    ok4 = found == [(3, 2)] and len(rs) == load_answer(3, 3)
    report(3, ok1 and ok2 and ok3 and ok4, f"[3.1] 해 {sols}; [3.2] 부호변화 {sign_changes}회; [3.3] 극한 유한비영 (r,s) {len(pairs)}쌍 모두 2s=r+1, 극한 1; [3.4] 전수 {found}, r 개수 {len(rs)}")


if __name__ == "__main__":
    for name in sorted(k for k in dict(globals()) if k.startswith("p") and k[1:].isdigit()):
        globals()[name]()
    passed = sum(RESULTS)
    print(f"SUMMARY G21B: {passed}/{len(RESULTS)} PASS")
    sys.exit(0 if passed == len(RESULTS) else 1)
