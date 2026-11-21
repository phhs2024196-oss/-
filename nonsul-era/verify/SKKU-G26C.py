#!/usr/bin/env python3
"""SKKU-G26C (성균관대 2026학년도 수시 논술위주 수리형 3교시) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import math
import sys

import numpy as np
import sympy as sp

PREFIX = "SKKU-G26C"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SKKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def md_kv(no, sub, loc=None):
    return {k.strip(): sp.sympify(v, locals=loc or {}) for k, v in (s.split("=", 1) for s in md_answer(no, sub).split(";"))}


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def count_level(fn, m, lo=-50, hi=50, N=400001):
    xs = np.linspace(lo, hi, N)
    v = fn(xs) - m
    z = np.abs(v) < 1e-12
    v = np.where(z, 0.0, v)
    # 격자점이 정확히 근인 경우(z)는 한 번만, 그 외에는 부호 변화만 센다
    return int(np.sum(v[:-1] * v[1:] < 0)) + int(np.sum(z))


def p01():
    A, B, P, Q = sp.symbols("A B P Q", real=True)
    L, R = -x**2 + A * x + B, x**2 + P * x + Q
    eqs = [sp.Eq(sp.diff(L, x).subs(x, 1), sp.diff(R, x).subs(x, 1)), sp.Eq(L.subs(x, 1), R.subs(x, 1))]
    # 수평선과 정확히 두 점에서 만나는 높이 = 두 꼭짓점의 높이 → 두 가지 배정 모두 시도 후 수치로 검증
    found = []
    for top, bot in [(2, -6), (-6, 2)]:
        sols = sp.solve(eqs + [sp.Eq(B + A**2 / 4, top), sp.Eq(Q - P**2 / 4, bot)], [A, B, P, Q], dict=True)
        for s in sols:
            if not all(v.is_real for v in s.values()):
                continue
            fA, fB, fP, fQ = (float(s[k]) for k in (A, B, P, Q))
            fn = lambda u: np.where(u < 1, -u**2 + fA * u + fB, u**2 + fP * u + fQ)
            ms = [mv for mv in np.arange(-12, 8, 0.25)] + [-6, 2]
            two = sorted(set(round(mv, 6) for mv in ms if count_level(fn, mv) == 2))
            if two == [-6.0, 2.0]:
                found.append(s)
    kv = md_kv(1, "i")
    ok1 = len(found) == 1 and all(sp.simplify(found[0][sp.Symbol(k, real=True)] - kv[k]) == 0 for k in "ABPQ")
    # (ii) 조건 (1)(2)(4)(5) + (3)의 하한 1 = 극솟값
    a, b, c, p, q = sp.symbols("a b c p q", real=True)
    gl = -x**3 + a * x**2 + b * x + c
    gr = x**2 + p * x + q
    sol = sp.solve([sp.Eq(c, q), sp.Eq(sp.diff(gl, x).subs(x, -2), 0), sp.Eq(gl.subs(x, -2), 1),
                    sp.Eq(p + c, a + b), sp.Eq(gr.subs(x, 1), sp.Rational(-19, 2))], [a, b, c, p, q], dict=True)
    kv2 = md_kv(1, "ii")
    s2 = sol[0]
    ok2 = len(sol) == 1 and all(sp.simplify(s2[sp.Symbol(k, real=True)] - kv2[k]) == 0 for k in "abcpq")
    # 조건 (2)(3) 수치 확인 + (iii) β = 네 점 교차하는 m 의 상한
    glf = sp.lambdify(x, gl.subs(s2)); grf = sp.lambdify(x, gr.subs(s2))
    fn = lambda u: np.where(u <= 0, glf(u), grf(u))
    ms = np.linspace(-3, 4, 1401)
    four = [mv for mv in ms if count_level(fn, mv) == 4]
    is_min = float(sp.diff(gl.subs(s2), x, 2).subs(x, -2)) > 0
    beta_md = float(sp.sympify(md_answer(1, "iii")))
    ok3 = is_min and abs(min(four) - 1) <= 0.006 and abs(max(four) - beta_md) <= 0.006 \
        and count_level(fn, beta_md) != 4 and count_level(fn, beta_md - 1e-4) == 4
    report(1, ok1 and ok2 and ok3, f"(i) {found} (ii) {s2} (iii) four-range≈[{min(four):.3f},{max(four):.3f}]")


def construct(AB, AD1, th, nmax):
    """A 를 원점, 이등분선을 -y 방향으로 두고 B, D_1 을 배치해 C_n, D_n 을 귀납적으로 작도."""
    A = np.array([0.0, 0.0])
    uB = np.array([-math.sin(th), -math.cos(th)])
    uD = np.array([math.sin(th), -math.cos(th)])
    B = AB * uB
    D = [None, AD1 * uD]
    C = [None]

    def circum(P1, P2, P3):
        ax, ay = P1; bx, by = P2; cx, cy = P3
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
        return np.array([ux, uy])

    def bis_point(Dn):
        # ∠BAD_n 의 이등분선 방향(단위벡터 합), 외접원과의 A 아닌 교점
        v = B / np.linalg.norm(B) + Dn / np.linalg.norm(Dn)
        v = v / np.linalg.norm(v)
        O = circum(A, B, Dn)
        return 2 * np.dot(O - A, v) * v
    C.append(bis_point(D[1]))
    for n in range(2, nmax + 1):
        d = C[n - 1] - D[n - 1]
        # B + s d = u uD
        M = np.array([[d[0], -uD[0]], [d[1], -uD[1]]])
        s, u = np.linalg.solve(M, -B)
        D.append(u * uD)
        C.append(bis_point(D[n]))
    return A, B, C, D, uD


def p02():
    th = sp.symbols("theta", real=True)
    m_md = sp.sympify(md_answer(2, "i"), locals={"theta": th})
    ok1 = True
    for tv in (0.1, 0.3, 0.5, 0.7):
        mv = float(m_md.subs(th, tv))
        A, B, C, D, _ = construct(mv, 1.0, tv, 1)
        v1, v2 = B - A, D[1] - C[1]
        ok1 &= abs(v1[0] * v2[1] - v1[1] * v2[0]) / (np.linalg.norm(v1) * np.linalg.norm(v2)) < 1e-9
    # (ii) AB=AD1=2: P = 직선 AB ∩ 직선 C1D1
    ap_md = sp.sympify(md_answer(2, "ii"), locals={"theta": th})
    ok2 = True
    for tv in (0.1, 0.3, 0.5, 0.7):
        A, B, C, D, _ = construct(2.0, 2.0, tv, 1)
        d1, d2 = B - A, D[1] - C[1]
        s, u = np.linalg.solve(np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]]), C[1] - A)
        AP = np.linalg.norm(s * d1)
        ok2 &= abs(AP - float(ap_md.subs(th, tv))) < 1e-9
    # (iii) AB=2, AD1=3, θ=π/6: D_n 이 A 의 반대편(∠BAD_n = π-2θ = 2π/3)이 되는 최소 n
    A, B, C, D, uD = construct(2.0, 3.0, math.pi / 6, 8)
    nmin = None
    for n in range(1, 9):
        u = float(np.dot(D[n], uD))
        ang = math.acos(np.dot(B, D[n]) / (np.linalg.norm(B) * np.linalg.norm(D[n])))
        if abs(ang - 2 * math.pi / 3) < 1e-9:
            nmin = n
            break
    kv = md_kv(2, "iii")
    ok3 = nmin == int(kv["n"]) and abs(np.linalg.norm(D[nmin - 1]) - float(kv["AD"])) < 1e-9
    report(2, ok1 and ok2 and ok3, f"(i)(ii) 작도 일치, (iii) n={nmin} AD_(n-1)={np.linalg.norm(D[nmin - 1]):.9f} "
           f"AD_2={np.linalg.norm(D[2]):.6f}")


def gfun_vals(xs):
    best = None
    for a, b, c in itertools.permutations([1, 2, 3]):
        v = (xs**2 - a * xs + b) * (xs - c)
        best = v if best is None else np.maximum(best, v)
    return best


def p03():
    # (i) 6개 순열의 최댓값 — 기호적으로: 각 구간에서 최대인 다항식을 판정해 정확 적분
    polys = [sp.expand((x**2 - a * x + b) * (x - c)) for a, b, c in itertools.permutations([1, 2, 3])]
    brk = sorted(set([sp.Integer(-1), sp.Integer(1)] + [r for i, j in itertools.combinations(range(6), 2)
                       for r in sp.solve(polys[i] - polys[j], x) if r.is_real and -1 < r < 1]), key=float)
    I = 0
    pieces = []
    for lo, hi in zip(brk[:-1], brk[1:]):
        mid = (lo + hi) / 2
        k = max(range(6), key=lambda i: float(polys[i].subs(x, mid)))
        pieces.append((lo, hi, polys[k]))
        I += sp.integrate(polys[k], (x, lo, hi))
    ok1 = sp.simplify(I - sp.sympify(md_answer(3, "i"))) == 0
    # (ii) 전 실수에서 최대 다항식이 바뀌는 점(꺾이는 점)과 도함수 점프
    allbrk = sorted(set(r for i, j in itertools.combinations(range(6), 2)
                        for r in sp.solve(polys[i] - polys[j], x) if r.is_real), key=float)
    def top(xv):
        vals = [float(pp.subs(x, xv)) for pp in polys]
        mval = max(vals)
        return [polys[i] for i, v in enumerate(vals) if abs(v - mval) < 1e-12]
    jumps = {sp.Integer(0)}
    for r in allbrk:
        L = top(r - sp.Rational(1, 10**6))[0]; R = top(r + sp.Rational(1, 10**6))[0]
        jv = sp.nsimplify(sp.simplify(sp.Abs(sp.diff(L, x).subs(x, r) - sp.diff(R, x).subs(x, r))))
        if jv != 0:
            jumps.add(jv)
    # 이웃 꺾이는 점 간 거리 ≥ 1/2 확인(열린구간 조건이 각 꺾이는 점에서 성립)
    kinks = [r for r in allbrk if any(sp.simplify(j) != 0 for j in [sp.Abs(sp.diff(top(r - sp.Rational(1, 10**6))[0], x).subs(x, r)
                                                                      - sp.diff(top(r + sp.Rational(1, 10**6))[0], x).subs(x, r))])]
    sep = all(float(b - a) >= 0.5 for a, b in zip(kinks[:-1], kinks[1:]))
    md2 = sp.sympify(md_answer(3, "ii"))
    ok2 = sep and set(sp.nsimplify(v) for v in jumps) == set(md2)
    # (iii) k(r): 각 r 에서 R 이 놓일 수 있는 둘레 길이(수치), 최댓값
    x1 = np.linspace(0, 1, 20001)
    y1 = gfun_vals(x1)

    def k_of(r):
        tot = 0.0
        # 변: x2 고정(−2, −14) 세로, y2 고정(r, r+12) 가로. 조건: x2(y1-2) + y2(x1+1) = y1 + 2x1
        for kind, const in (("x", -2.0), ("x", -14.0), ("y", r), ("y", r + 12.0)):
            if kind == "x":
                with np.errstate(divide="ignore", invalid="ignore"):
                    s = (y1 + 2 * x1 - const * (y1 - 2)) / (x1 + 1)          # y2
                lo, hi = r, r + 12
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    s = (y1 + 2 * x1 - const * (x1 + 1)) / (y1 - 2)          # x2
                lo, hi = -14, -2
            valid = np.isfinite(s) & (s >= lo) & (s <= hi)
            ivs = []
            for i in range(len(s) - 1):
                if valid[i] and valid[i + 1]:
                    ivs.append((min(s[i], s[i + 1]), max(s[i], s[i + 1])))
            if not ivs:
                continue
            ivs.sort()
            cur_lo, cur_hi = ivs[0]
            for a_, b_ in ivs[1:]:
                if a_ <= cur_hi + 1e-12:
                    cur_hi = max(cur_hi, b_)
                else:
                    tot += cur_hi - cur_lo
                    cur_lo, cur_hi = a_, b_
            tot += cur_hi - cur_lo
        return tot
    rs = np.arange(-30, 5, 0.25)
    ks = [k_of(rv) for rv in rs]
    i0 = int(np.argmax(ks))
    lo, hi = rs[max(i0 - 1, 0)], rs[min(i0 + 1, len(rs) - 1)]
    for _ in range(60):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if k_of(m1) < k_of(m2):
            lo = m1
        else:
            hi = m2
    kmax = k_of((lo + hi) / 2)
    ok3 = abs(kmax - float(sp.sympify(md_answer(3, "iii")))) < 5e-3
    report(3, ok1 and ok2 and ok3, f"(i)={I} (ii) jumps={sorted(jumps, key=float)} (iii) kmax≈{kmax:.5f} at r≈{(lo + hi) / 2:.4f}")


if __name__ == "__main__":
    for name in sorted(k for k in dict(globals()) if k.startswith("p") and k[1:].isdigit()):
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(int(name[1:]), False, f"예외 {type(e).__name__}: {e}")
    print(f"SUMMARY {PREFIX}: {sum(RESULTS)}/{len(RESULTS)} PASS")
    sys.exit(0 if all(RESULTS) else 1)
