#!/usr/bin/env python3
"""서울과기대 2025학년도 논술전형고사(오전) 문제 1~3 (SNUT-G25A-01~03) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건으로부터 독립 재계산해 YAML answer.value 와 비교.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SNUT-G25A"
WIDTH = 2
RESULTS = []
x, t, n, k = sp.symbols("x t n k", real=True)
SYM = {"x": x, "t": t, "n": n, "k": k, "pi": sp.pi, "e": sp.E}


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


def same(a, b, tol=1e-9):
    """기호 동일 또는 수치 근사 동일."""
    try:
        if sp.simplify(a - b) == 0:
            return True
    except Exception:  # noqa: BLE001
        pass
    return abs(complex(sp.N(a - b, 30))) < tol


def p01():
    """[1.1] 삼차방정식 근→식 계산, [1.2] 역함수·대칭합, [1.3] 대우 카드 전수."""
    # 1.1
    roots = sp.Poly(x**3 - 18*x**2 - 20*x + 40, x).all_roots()
    nonint = sorted([r for r in roots if not r.is_integer], key=lambda r: float(sp.N(r)))
    be, al = nonint[0], nonint[1]
    val = sp.radsimp((sp.sqrt(al) + sp.sqrt(be)) / (sp.sqrt(al) - sp.sqrt(be)))
    val = sp.nsimplify(sp.N(val, 40), [sp.sqrt(5)])
    pq = sp.Poly(sp.expand(val), sp.sqrt(5))
    q = val.coeff(sp.sqrt(5)); pp = sp.simplify(val - q*sp.sqrt(5))
    s1 = pp + q + 5
    ok1 = same(s1, load_answer(1, 0))
    report(1, ok1, f"[1.1] α={sp.N(al,6)}, β={sp.N(be,6)}, 값={val}, p+q+r={s1}, yaml={load_answer(1,0)}")
    # 1.2
    X, Y = sp.symbols("X Y", positive=True)
    f = lambda z: sp.exp(2*z) / (sp.exp(2*z) + sp.E)
    sol = sp.solve(sp.Eq(Y, f(X)), X)
    sol = [z for z in sol if abs(sp.im(sp.N(z.subs(Y, sp.Rational(3, 10))))) < 1e-12]
    assert len(sol) == 1
    g = lambda z: sol[0].subs(Y, z)
    pts = [sp.Rational(i, 17) for i in range(1, 17)]
    a_ok = all(abs(sp.N(f(p) + f(1 - p) - 1, 30)) < 1e-25 for p in pts)
    b_ok = all(abs(sp.N(g(p) + g(1 - p) - 1, 30)) < 1e-25 for p in pts)
    inv_ok = all(abs(sp.N(g(f(p)) - p, 30)) < 1e-25 for p in pts)
    c = sum(f(sp.Rational(j, 10)) + g(sp.Rational(j, 10)) for j in range(1, 10))
    tot = 1 + 1 + sp.N(c, 30)
    ok2 = a_ok and b_ok and inv_ok and abs(tot - sp.N(load_answer(1, 1), 30)) < 1e-20
    report(1, ok2, f"[1.2] a=1,b=1 확인={a_ok and b_ok}, c(수치)={sp.N(c,15)}, a+b+c={sp.N(tot,12)}, yaml={load_answer(1,1)}")
    # 1.3 카드 전수: 앞면 가능한 보이는 값, 반례 가능성(뒷면 미지수) 판정
    from itertools import product
    nums = {"2": 2, "7": 7, "10": 10, "3": 3}
    shapes = ["해", "달", "별", "구름"]
    need = []
    cards = ["2", "해", "달", "7", "별", "10", "3", "구름"]
    def violates(front, back):
        a = front if front in nums else back
        s = back if front in nums else front
        v = nums.get(a, a) if isinstance(a, str) and a in nums else a
        c1 = (v % 2 == 0) and s not in ("해", "달")
        c2 = (s == "별") and v < 7
        return c1 or c2
    for c in cards:
        if c in nums:
            backs = shapes
        else:
            backs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        possible = []
        for b in backs:
            fr, bk = (c, b) if c in nums else (c, b)
            if c in nums:
                vv, ss = nums[c], b
            else:
                vv, ss = b, c
            bad = ((vv % 2 == 0) and ss not in ("해", "달")) or (ss == "별" and vv < 7)
            possible.append(bad)
        if any(possible):
            need.append(c)
    ans = [s.strip() for s in load_answer(1, 2, raw=True).split(",")]
    ok3 = sorted(need) == sorted(ans)
    report(1, ok3, f"[1.3] 반례 가능한 카드={need}, yaml={ans}")


def p02():
    """외분점 → k, l1; 접선 대칭이동 → 거리; 곡선 길이."""
    kk = sp.symbols("kk", positive=True)
    d = sp.symbols("d", real=True)
    # A 가 DC 를 1:3 으로 외분: A = (3*D... ) 외분 m:n → (m*C - n*D)/(m-n)  (m=1,n=3)
    Ax = (1*kk - 3*d) / (1 - 3); Ay = (1*2*sp.log(kk) - 3*0) / (1 - 3)
    s = sp.solve([sp.Eq(Ax, sp.Rational(1, 2)), sp.Eq(Ay, -2*sp.log(2))], [kk, d], dict=True)
    assert len(s) == 1
    kv, dv = s[0][kk], s[0][d]
    A = sp.Point(sp.Rational(1, 2), -2*sp.log(2)); C = sp.Point(kv, 2*sp.log(kv))
    # A 가 곡선 위 점인지, 외분 비 확인: A - D = (1/3)... DA:AC? 외분 1:3 → |AD|:|AC| = 1:3 이고 A 는 D 와 C 바깥쪽
    D = sp.Point(dv, 0)
    ratio_ok = sp.simplify(A.distance(D) * 3 - A.distance(C)) == 0 and sp.simplify(sp.Matrix(A - D).dot(sp.Matrix(A - C))) > 0
    l1 = sp.Line(A, C)
    slope = sp.simplify(-l1.coefficients[0] / l1.coefficients[1])
    icpt = sp.simplify(-l1.coefficients[2] / l1.coefficients[1])
    raw = load_answer(2, 0, raw=True)
    kstr, lstr = [z.strip() for z in raw.split(";")]
    ky = sp.sympify(kstr.split("=")[1], locals=SYM)
    ly = sp.sympify(lstr.split("=")[1], locals=SYM)
    ok1 = ratio_ok and same(kv, ky) and same(slope * x + icpt, ly)
    report(2, ok1, f"[2.1] k={kv}, d={dv}, l1: y={sp.simplify(slope)}x+{sp.simplify(icpt)}")
    # 2.2
    X = sp.symbols("X", positive=True)
    ycurve = 2*sp.log(X)
    m = sp.diff(ycurve, X).subs(X, kv)
    # 접선 y - yc = m (x - kv). 대칭이동 (x,y)->(y,x): x - yc = m (y - kv)
    xx, yy = sp.symbols("xx yy", real=True)
    tang = yy - ycurve.subs(X, kv) - m*(xx - kv)
    l2 = tang.subs({xx: yy, yy: xx}, simultaneous=True)
    a_, b_ = sp.Poly(l2, xx, yy).coeff_monomial(xx), sp.Poly(l2, xx, yy).coeff_monomial(yy)
    c_ = l2.subs({xx: 0, yy: 0})
    dist = sp.Abs(a_*C[0] + b_*C[1] + c_) / sp.sqrt(a_**2 + b_**2)
    # 수치 확인: 대칭이동한 접선 위의 임의의 점의 역반사가 접선 위
    ok2 = same(dist, load_answer(2, 1))
    report(2, ok2, f"[2.2] 거리={sp.simplify(dist)}≈{sp.N(dist,10)}, yaml={load_answer(2,1)}")
    # 2.3
    L = sp.Integral(sp.sqrt(1 + sp.diff(ycurve, X)**2), (X, 1, kv))
    Ln = sp.N(L, 30)
    ok3 = abs(Ln - sp.N(load_answer(2, 2), 30)) < 1e-20
    report(2, ok3, f"[2.3] 길이(수치)={sp.N(Ln,15)}, yaml={sp.N(load_answer(2,2),15)}")


def p03():
    """f=cos x/x^2: 영점 x_n, 접선 y절편, 삼각형 넓이, 급수."""
    f = sp.cos(x) / x**2
    xn = load_answer(3, 0)  # (2n-1)pi/2 (사용자 답과 독립 재계산 비교)
    zeros = sorted(set(float(sp.N(z)) for m in range(-3, 40) for z in [sp.pi/2 + m*sp.pi] if z > 0))
    ok1 = all(abs(zeros[i] - float(sp.N(xn.subs(n, i+1)))) < 1e-9 for i in range(10)) and \
        all(abs(float(sp.N(f.subs(x, z)))) < 1e-9 for z in zeros[:10])
    report(3, ok1, f"[3.1] 영점 앞 3개={[round(z,6) for z in zeros[:3]]}, yaml x_n={xn}")
    fp = sp.diff(f, x)
    def yint(xv):
        return sp.simplify(f.subs(x, xv) - fp.subs(x, xv) * xv)  # 접선의 y절편
    xs = [sp.pi/2 + i*sp.pi for i in range(2025)]
    s2 = sum(sp.nsimplify(sp.N(xv * yint(xv), 25), rational=True) for xv in xs[:60])
    # 60항까지 직접 + 규칙(교대 ±1)으로 2025항
    vals = [sp.N(xv*yint(xv), 20) for xv in xs[:60]]
    pattern = all(abs(vals[i] - (1 if i % 2 == 0 else -1)) < 1e-12 for i in range(60))
    total = sum(1 if i % 2 == 0 else -1 for i in range(2025))
    ok2 = pattern and same(total, load_answer(3, 1))
    report(3, ok2, f"[3.2] x_n y_n 교대 ±1 확인={pattern}, 2025항 합={total}, yaml={load_answer(3,1)}")
    # 3.3
    def A(i):
        xi, xj = xs[i], xs[i+1]
        P = sp.Point(xi, 0); P2 = sp.Point(xj, 0); Q = sp.Point(0, yint(xi))
        return sp.simplify(sp.Abs(sp.Triangle(P, P2, Q).area))
    As = [A(i) for i in range(40)]
    formula = all(sp.simplify(As[i] - sp.Rational(1, 2*(i+1)-1)) == 0 for i in range(40))
    s25 = sum(As[i]*As[i+1] for i in range(25))
    ser = sp.summation(1/((2*n-1)*(2*n+1)), (n, 1, sp.oo))
    a1, a2 = [sp.sympify(z, locals=SYM) for z in load_answer(3, 2, raw=True).split(";")]
    ok3 = formula and same(s25, a1) and same(ser, a2)
    report(3, ok3, f"[3.3] A_n=1/(2n-1) 확인={formula}, S25={s25}, 급수={ser}, yaml={a1};{a2}")


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
