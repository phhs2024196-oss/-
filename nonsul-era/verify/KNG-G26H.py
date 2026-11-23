import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
from vlib import *
import numpy as np
from sympy.parsing.sympy_parser import parse_expr
V = Verifier("KNG-G26H", "items/KNG/PAST")
P = lambda s: parse_expr(s.replace("^", "**"), local_dict={"x": x})

def p09():
    # M-m 은 k 에 무관한 상수. 격자 수치(끝점 가정 없이 구간 전체 샘플)와 기호 계산 둘 다.
    kk = sp.symbols("k", positive=True)
    f = lambda xx, kv: np.log2(xx + kv)
    for kv in (0.1, 0.5, 1, 2, 7.3, 40):
        xs = np.linspace(0, kv, 20001)
        V.check(9, f"k={kv} M-m", float(f(xs, kv).max() - f(xs, kv).min()), P(V.ans(9, "1")), 1e-9)
    M = sp.log(2 * kk, 2); m = sp.log(kk, 2)
    V.check(9, "기호 M-m", sp.simplify(sp.expand_log(M - m, force=True)), P(V.ans(9, "1")))
    V.check(9, "증가(도함수>0)", sp.simplify(sp.diff(sp.log(x + kk, 2), x) * sp.log(2)), 1 / (x + kk))

def p10():
    f = x**3 / 3 - x
    V.check(10, "1 f'", sp.diff(f, x), P(V.ans(10, "1")))
    cp = sorted(sp.solve(sp.diff(f, x), x))
    f2 = sp.diff(f, x, 2)
    mx = [f.subs(x, c) for c in cp if f2.subs(x, c) < 0]
    mn = [f.subs(x, c) for c in cp if f2.subs(x, c) > 0]
    V.check(10, "2 극댓값", mx[0], P(V.ans(10, "2")))
    V.check(10, "3 극솟값", mn[0], P(V.ans(10, "3")))
    V.check(10, "4 정적분", sp.integrate(f, (x, 0, sp.sqrt(3))), P(V.ans(10, "4")))
    # 부호 변화로도 확인(극점 양쪽 증감)
    fp = sp.diff(f, x)
    V.report(10, fp.subs(x, -2) > 0 and fp.subs(x, 0) < 0 and fp.subs(x, 2) > 0, "증감 + - + 로 x=-1 극대, x=1 극소")

if __name__ == "__main__": V.run(globals())
