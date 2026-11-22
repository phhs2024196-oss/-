#!/usr/bin/env python3
"""SGU-G26A (서강대 2026학년도 수시 논술 자연계열 1차) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SGU-G26A"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SGU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def split_top(s: str, sep: str = ","):
    depth, cur, out = 0, "", []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == sep and depth == 0:
            out.append(cur); cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


def parse_intervals(ans: str):
    """'(-oo,a]:0; {b}:1; ...' → [(lo, hi, lo_closed, hi_closed, count)]"""
    loc = {"oo": sp.oo, "pi": sp.pi}
    res = []
    for seg in ans.split(";"):
        iv, cnt = seg.strip().rsplit(":", 1)
        iv = iv.strip()
        if iv.startswith("{"):
            v = sp.sympify(iv[1:-1], locals=loc)
            res.append((v, v, True, True, int(cnt)))
            continue
        lo, hi = split_top(iv[1:-1])
        res.append((sp.sympify(lo, locals=loc), sp.sympify(hi, locals=loc), iv[0] == "[", iv[-1] == "]", int(cnt)))
    return res


def sample_points(lo, hi, lc, hc):
    """구간의 대표점: 닫힌 끝점(정확값), 내부 여러 점(유리수)."""
    pts = []
    if lo == hi:
        return [lo]
    if lc:
        pts.append(lo)
    if hc:
        pts.append(hi)
    L = sp.Rational(-10) if lo == -sp.oo else lo
    H = sp.Rational(10) if hi == sp.oo else hi
    for r in (sp.Rational(1, 1000), sp.Rational(1, 3), sp.Rational(1, 2), sp.Rational(2, 3), sp.Rational(999, 1000)):
        pts.append(sp.nsimplify(sp.N(L + (H - L) * r, 30), rational=True))
    return pts


def covers_line(ivs):
    """구간들이 서로 겹치지 않고 실수 전체를 덮는지(정렬 후 이웃 끝점 일치·닫힘 보완)."""
    ivs = sorted(ivs, key=lambda r: (float(r[0]) if r[0] != -sp.oo else -1e18, 0 if r[2] else 1))
    if ivs[0][0] != -sp.oo or ivs[-1][1] != sp.oo:
        return False
    for a, b in zip(ivs, ivs[1:]):
        if sp.simplify(a[1] - b[0]) != 0 or a[3] == b[2]:
            return False
    return True


# ── 문제 1 ─────────────────────────────────────────────
def p01():
    # (1-1) 전수 세기: 거리 5|b|/sqrt(a^2+4) <= 5  ⇔  b^2 <= a^2+4 (정수 비교)
    cnt = sum(1 for a in range(-100, 101) for b in range(-100, 101) if 25 * b * b <= 25 * (a * a + 4))
    ok1 = cnt == int(md_answer(1, "1"))

    # (1-2) 연속·미분가능 조건을 극한으로 세워 풀기
    a, b, c, d = sp.symbols("a b c d", real=True)
    mid = (1 - sp.sqrt(1 + 3 * x**2)) / x
    b0 = sp.limit(mid, x, 0, "+")                                      # 연속: f(0)=b
    eqs = [
        sp.Eq(b, b0),
        sp.Eq(a, sp.limit((mid - b0) / x, x, 0, "+")),                 # 좌미분계수 a = 우미분계수
        sp.Eq(c + d, sp.limit(mid, x, 1, "-")),
        sp.Eq(c, sp.limit((mid - mid.subs(x, 1)) / (x - 1), x, 1, "-")),
    ]
    sol = sp.solve(eqs, [a, b, c, d], dict=True)[0]
    md2 = sp.sympify(md_answer(1, "2"))
    ok2 = all(sp.simplify(sol[v] - w) == 0 for v, w in zip((a, b, c, d), md2))
    # 실제로 미분가능한지 수치 확인
    fpw = sp.Piecewise((sol[a] * x + sol[b], x <= 0), (mid, x < 1), (sol[c] * x + sol[d], True))
    ok2 = ok2 and all(abs(float(sp.diff(fpw, x).subs(x, p + e) - sp.diff(fpw, x).subs(x, p - e))) < 1e-4
                      for p in (0, 1) for e in (1e-6,))

    # (1-3) F(x)=∫_0^x t f(x-t)dt = ∫_0^x (x-s) f(s) ds ⇒ F''=f
    G = x**2 * sp.exp(-x**2)
    f = sp.simplify(sp.diff(G, x, 2))
    s_, tt = sp.symbols("s_ tt", real=True)
    for xv in (sp.Rational(7, 10), sp.Rational(-3, 2)):                    # 원 방정식 수치 대입 확인
        lhs = sp.Integral(tt * f.subs(x, xv - tt), (tt, 0, xv)).evalf(30)
        assert abs(lhs - G.subs(x, xv).evalf(30)) < 1e-20
    val3 = f.subs(x, 0) * sp.integrate(f, (x, -2, 3))
    ok3 = sp.simplify(val3 - sp.sympify(md_answer(1, "3"))) == 0

    # (1-4) f'' 분자 6kx^2+2x-1 의 양의 근 중 홀수 중복도(부호 변화) 개수
    k = sp.symbols("k", real=True)
    def n_inflect(kv):
        num = sp.Poly(sp.expand(6 * kv * x**2 + 2 * x - 1), x)
        roots = sp.roots(num)
        return sum(1 for r, m in roots.items() if r.is_real and r > 0 and m % 2 == 1)
    ffull = k * x**3 + x**2 - x * sp.log(x)
    assert sp.simplify(sp.diff(ffull, x, 2) - (6 * k * x**2 + 2 * x - 1) / x) == 0
    ivs = parse_intervals(md_answer(1, "4"))
    ok4 = covers_line(ivs) and all(n_inflect(p) == cnt_ for lo, hi, lc, hc, cnt_ in ivs
                                   for p in sample_points(lo, hi, lc, hc))
    report(1, ok1 and ok2 and ok3 and ok4,
           f"count={cnt} (a,b,c,d)={tuple(sol[v] for v in (a, b, c, d))} value={sp.simplify(val3)} inflect-intervals={len(ivs)}")


# ── 문제 2 ─────────────────────────────────────────────
def acute_tan(tv):
    """두 직선 P(-1,0), P(-2,0) 의 방향벡터가 이루는 예각의 tan (수치, 정의대로)."""
    import mpmath as mp
    px, py = 2 * mp.cos(tv), 2 * mp.sin(tv)
    u = (px + 1, py); v = (px + 2, py)
    cr = abs(u[0] * v[1] - u[1] * v[0]); dt = abs(u[0] * v[0] + u[1] * v[1])
    return cr / dt


def p02():
    import mpmath as mp
    mp.mp.dps = 30
    # (2-1) tan(α-β) 로 f 를 구하고 정의(예각)와 수치 비교, t=2π/3 에서 연속
    m1 = 2 * sp.sin(t) / (2 * sp.cos(t) + 1); m2 = 2 * sp.sin(t) / (2 * sp.cos(t) + 2)
    f = sp.simplify((m1 - m2) / (1 + m1 * m2))
    ok_f = all(abs(mp.mpf(str(sp.N(f.subs(t, sp.Rational(kk, 10)), 30))) - acute_tan(mp.mpf(kk) / 10)) < 1e-20
               for kk in (1, 7, 13, 19, 25, 30))
    t0 = 2 * sp.pi / 3
    f_at = acute_tan(mp.mpf(2) * mp.pi / 3)                              # 정의(직각삼각형)로 함숫값
    lim_l, lim_r = sp.limit(f, t, t0, "-"), sp.limit(f, t, t0, "+")
    ok1 = ok_f and sp.simplify(lim_l - lim_r) == 0 and abs(mp.mpf(str(sp.N(lim_l, 30))) - f_at) < 1e-20 \
        and md_answer(2, "1") is None          # [증명-수치확인]

    # (2-2) α: sinα=√2/4, 0<α<π/2
    al = sp.asin(sp.sqrt(2) / 4)
    I = sp.integrate(f, (t, 2 * al, t0))
    ok2 = abs(sp.N(I - sp.sympify(md_answer(2, "2")), 40)) < 1e-30

    # (2-3) g(x)=f(x)/f(2x)^3, lim_{x→0+} g(π/2-x)/x^n
    g = f.subs(t, x) / f.subs(t, 2 * x)**3
    xp = sp.symbols("xp", positive=True)
    gh = g.subs(x, sp.pi / 2 - xp)
    lims = {}
    for nn in range(1, 7):
        L = sp.limit(gh / xp**nn, xp, 0, "+")
        lims[nn] = L
    conv = {nn: L for nn, L in lims.items() if L.is_finite}
    md3 = {int(a.split("=")[1]): sp.sympify(b) for a, b in (seg.strip().split(":") for seg in md_answer(2, "3").split(";"))}
    ok3 = set(conv) == set(md3) and all(sp.simplify(conv[k_] - md3[k_]) == 0 for k_ in conv)

    # (2-4) h(x)=f(x)-x/3 의 단조 구간 분해로 k 별 실근 개수 계산
    h = f.subs(t, x) - x / 3
    hp = sp.simplify(sp.diff(h, x))
    crit = sorted(sp.solveset(sp.numer(sp.together(hp)), x, sp.Interval.open(0, t0)), key=float)
    pts = [sp.Integer(0)] + crit + [t0]
    vals = [sp.limit(h, x, 0, "+")] + [sp.simplify(h.subs(x, cpt)) for cpt in crit] + [sp.limit(h, x, t0, "-")]
    def count(kv):
        c_ = 0
        for i in range(len(pts) - 1):
            lo_, hi_ = sorted([vals[i], vals[i + 1]], key=float)
            if sp.N(lo_ - kv, 50) < 0 < sp.N(hi_ - kv, 50):
                c_ += 1
        c_ += sum(1 for j, cpt in enumerate(crit) if sp.simplify(vals[j + 1] - kv) == 0)
        return c_
    ivs = parse_intervals(md_answer(2, "4"))
    ok4 = covers_line(ivs) and all(count(p) == cnt_ for lo, hi, lc, hc, cnt_ in ivs for p in sample_points(lo, hi, lc, hc))
    report(2, ok1 and ok2 and ok3 and ok4,
           f"f={f} [증명-수치확인] f(2pi/3)={float(lim_l):.6f} I={sp.N(I, 12)} conv={conv} crit={crit} vals={[sp.N(v, 6) for v in vals]}")


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
