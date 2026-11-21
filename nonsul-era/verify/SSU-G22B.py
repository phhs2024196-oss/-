#!/usr/bin/env python3
"""숭실대 2022학년도 논술고사(자연2) (SSU-G22B-01~04) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
yaml 의 answer.value(sympy 문법, 'a=.., b=..' 꼴은 이름별 dict, 'L=R' 꼴은 L-R 식)를 읽어 독립 재계산 값과 비교한다.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "SSU-G22B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)
mp.mp.dps = 30


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def _split_top(s):
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def parse_answer(s):
    """'1/4' → 식, '[..]' → 리스트, 'a=1, b=2' → dict, 'L=R' → L-R."""
    s = str(s).strip()
    loc = {"E": sp.E, "pi": sp.pi}
    if s.startswith("["):
        return list(sp.sympify(s, locals=loc))
    parts = _split_top(s)
    if all(re.fullmatch(r"[A-Za-z]\w*\s*=[^=].*", p) for p in parts):
        if len(parts) > 1 or re.fullmatch(r"[A-Za-z]\w*", parts[0].split("=")[0].strip()):
            return {p.split("=", 1)[0].strip(): sp.sympify(p.split("=", 1)[1], locals=loc) for p in parts}
    if "=" in s:
        l, r = s.split("=", 1)
        return sp.sympify(l, locals=loc) - sp.sympify(r, locals=loc)
    return sp.sympify(s, locals=loc)


def load_answer(seq, sub="1"):
    import pathlib
    import yaml
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/SSU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    sp_ = next(s for s in d["subparts"] if str(s["id"]) == str(sub))
    return parse_answer(sp_["answer"]["value"])


def close(a, b, tol=1e-12):
    return abs(complex(sp.N(a, 40)) - complex(sp.N(b, 40))) < tol


def ginv(X):
    """g(X) = (u ln u = X^2 의 해 u>=1) — 수치로 구성."""
    X = mp.mpf(X)
    if X == 0:
        return mp.mpf(1)
    h = lambda u: u * mp.log(u) - X**2
    lo, hi = mp.mpf(1), mp.mpf(1) + X**2 + 2
    for _ in range(60):          # 이분법으로 좁힌 뒤 뉴턴법으로 고정밀화
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if h(mid) < 0 else (lo, mid)
    return mp.findroot(h, (lo + hi) / 2, solver="newton")


def p01():
    """g 를 수치로 구성해 (1) 접선의 기울기·절편, (2) ∫0^√e 2x g(x)^2 dx 를 직접 계산."""
    x0 = mp.sqrt(mp.e)
    y0 = ginv(x0)
    slope = mp.diff(ginv, x0)
    ans1 = load_answer(1, "1")
    line = ans1["y"]
    X = sp.Symbol("x")
    ok1 = abs(y0 - mp.e) < 1e-20 and all(
        abs(mp.mpf(str(sp.N(line.subs(X, sp.Float(str(xx), 30)), 30))) - (y0 + slope * (xx - x0))) < 1e-15
        for xx in (mp.mpf(0), mp.mpf(1), mp.mpf(3)))
    I = mp.quad(lambda xx: 2 * xx * ginv(xx) ** 2, [0, 0.5, 1, x0])
    ans2 = load_answer(1, "2")
    ok2 = abs(I - mp.mpf(str(sp.N(ans2, 30)))) < 1e-15
    report(1, ok1 and ok2, f"g(√e)={mp.nstr(y0, 12)}, g'(√e)={mp.nstr(slope, 12)}, 적분={mp.nstr(I, 15)}, md=({ans1}, {ans2})")


def _f(X):
    X = mp.mpf(X)
    return 1 - X**2 if X <= 0 else mp.sqrt(1 - X**2)


def _area(pts):
    s = 0
    for i in range(len(pts)):
        (x1, y1), (x2, y2) = pts[i], pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def _ang(P, Q, R):   # ∠PQR
    v, w = (P[0] - Q[0], P[1] - Q[1]), (R[0] - Q[0], R[1] - Q[1])
    return mp.acos((v[0] * w[0] + v[1] * w[1]) / (mp.sqrt(v[0]**2 + v[1]**2) * mp.sqrt(w[0]**2 + w[1]**2)))


def p02():
    """B, C 를 반원 위에서 각 ∠ABC=150° 가 되도록 수치로 찾고(여러 B 에 대해 C 가 고정되는지),
    오각형 넓이를 B·D 에 대해 수치 최대화."""
    A, E = (mp.mpf(1), mp.mpf(0)), (mp.mpf(-1), mp.mpf(0))
    Cs = []
    for xb in (mp.mpf("0.95"), mp.mpf("0.9"), mp.mpf("0.8")):
        B = (xb, _f(xb))
        xc = mp.findroot(lambda xc: _ang(A, B, (xc, _f(xc))) - 5 * mp.pi / 6, (mp.mpf("0.0001"), xb - mp.mpf("1e-9")), solver="bisect")
        Cs.append(xc)
    C = (Cs[0], _f(Cs[0]))
    ace = _area([A, C, E])
    ans1 = load_answer(2, "1")
    ok1 = max(Cs) - min(Cs) < 1e-20 and abs(ace - mp.mpf(str(sp.N(ans1, 30)))) < 1e-20
    # (2) B∈(xc,1), D∈(-1,xc) 에서 넓이 최대(두 변수 분리: ABC 와 CDE)
    fb = lambda xb: _area([A, (xb, _f(xb)), C])
    fd = lambda xd: _area([C, (xd, _f(xd)), E])
    gb = max((fb(C[0] + (1 - C[0]) * k / 4000), C[0] + (1 - C[0]) * k / 4000) for k in range(1, 4000))[1]
    gd = max((fd(-1 + (C[0] + 1) * k / 4000), -1 + (C[0] + 1) * k / 4000) for k in range(1, 4000))[1]
    xb = mp.findroot(lambda z: mp.diff(fb, z), gb)
    xd = mp.findroot(lambda z: mp.diff(fd, z), gd)
    md = load_answer(2, "2")
    pts = {"B": (xb, _f(xb)), "C": C, "D": (xd, _f(xd))}
    ok2 = all(all(abs(pts[k][i] - mp.mpf(str(sp.N(md[k][i], 30)))) < 1e-15 for i in (0, 1)) for k in "BCD")
    report(2, ok1 and ok2, f"C_x(B 3개)={[mp.nstr(c, 10) for c in Cs]}, ACE={mp.nstr(ace, 12)}, "
                           f"B=({mp.nstr(xb,10)},{mp.nstr(_f(xb),10)}) D=({mp.nstr(xd,10)},{mp.nstr(_f(xd),10)}), md=({ans1}; {md})")


def p03():
    """접선 xcosθ+ysinθ=1 에 내린 수선의 발을 정사영 공식으로 구하고, 넓이를 sympy 로 최대화."""
    th = sp.Symbol("theta", real=True)
    u = sp.Matrix([sp.cos(th), sp.sin(th)])
    A = sp.Matrix([-1, 0])
    H = A - (A.dot(u) - 1) * u
    md = [v.subs(sp.Symbol("theta"), th) for v in load_answer(3, "1")["H"]]
    ok1 = all(sp.simplify(sp.expand_trig(H[i] - md[i])) == 0 for i in (0, 1))
    P = u
    area = sp.simplify(sp.Abs((H - A)[0] * (P - A)[1] - (H - A)[1] * (P - A)[0]) / 2)
    area_f = lambda v: float(area.subs(th, v))
    gridmax = max(area_f(k * mp.pi / 20000) for k in range(1, 20000))
    ans2 = load_answer(3, "2")
    ok2 = abs(gridmax - float(ans2)) < 1e-8 and gridmax <= float(ans2) + 1e-12
    report(3, ok1 and ok2, f"H={tuple(sp.simplify(h) for h in H)}, 넓이 격자최대={gridmax:.12f}, md=({md}; {ans2})")


def p04():
    """x=2 에서 연속·미분가능 조건으로 a, r 을 sympy 로 풀고, ∫0^{a+r} f 를 수치 적분."""
    a, r = sp.symbols("a r", positive=True)
    s = sp.Symbol("s", real=True)
    F02 = sp.integrate(sp.Abs(sp.cos(sp.pi * s / 2)), (s, 0, 1)) - sp.integrate(sp.cos(sp.pi * s / 2), (s, 1, 2))
    dF2 = sp.Abs(sp.cos(sp.pi * 2 / 2))                      # 정적분으로 정의된 함수의 미분
    circ = sp.sqrt(r**2 - (x - a) ** 2)
    sols = sp.solve([circ.subs(x, 2) - F02, sp.diff(circ, x).subs(x, 2) - dF2], [a, r], dict=True)
    sols = [q for q in sols if (q[a] + q[r]) > 2]
    ans1 = load_answer(4, "1")
    ok1 = len(sols) == 1 and sp.simplify(sols[0][a] - ans1["a"]) == 0 and sp.simplify(sols[0][r] - ans1["r"]) == 0
    av, rv = (mp.mpf(str(sp.N(sols[0][k], 30))) for k in (a, r))
    f = lambda X: mp.quad(lambda z: abs(mp.cos(mp.pi * z / 2)), [0, min(X, 1), X]) if X <= 2 else mp.sqrt(max(rv**2 - (X - av) ** 2, 0))
    I = mp.quad(f, [0, 1, 2, av, av + rv])
    ans2 = load_answer(4, "2")
    ok2 = abs(I - mp.mpf(str(sp.N(ans2, 30)))) < 1e-15
    report(4, ok1 and ok2, f"a={sols[0][a]}, r={sols[0][r]}, 적분={mp.nstr(I, 15)}, md=({ans1}; {ans2})")


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
