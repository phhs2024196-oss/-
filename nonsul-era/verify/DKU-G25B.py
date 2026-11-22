#!/usr/bin/env python3
"""DKU-G25B (단국대 2025학년도 논술 자연계열 오후) — 검산. 규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건에서 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import mpmath as mp
import numpy as np
import sympy as sp

PREFIX = "DKU-G25B"
WIDTH = 2
RESULTS = []
x, t, k, a = sp.symbols("x t k a")
NS = {"pi": sp.pi, "E": sp.E, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt}


def md_answer(no: int, sub: str) -> str:
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/DKU/PAST" / f"DKU-G25B-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def ans_num(no, sub):
    return float(sp.N(sp.sympify(md_answer(no, sub), locals=NS)))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"DKU-G25B-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def close(u, v, tol=1e-9):
    return abs(float(u) - float(v)) < tol



def gmin(tv):
    """f(x)=x^2+|x-t| 는 볼록 → 최솟값은 꺾임점 x=t 또는 각 가지의 꼭짓점 x=±1/2 중 하나."""
    cands = [tv, 0.5, -0.5]
    return min(c * c + abs(c - tv) for c in cands)


def p01():
    mp.mp.dps = 25
    # 1. g(t) (격자 최솟값과 후보식이 같음을 확인한 뒤) -1/2<t<1/2 에서 YAML 식과 비교
    xs = np.linspace(-3, 3, 600001)
    ts = np.linspace(-2, 2, 81)
    okg = all(abs(np.min(xs**2 + np.abs(xs - tv)) - gmin(tv)) < 1e-9 for tv in ts)
    ge = sp.lambdify(t, sp.sympify(md_answer(1, "1"), locals=NS), "numpy")
    ok1 = okg and all(abs(gmin(tv) - ge(tv)) < 1e-12 for tv in np.linspace(-0.499, 0.499, 41))
    # 2. g' 를 수치미분하여 적분
    gprime = lambda tv: (mp.mpf(gmin(float(tv + mp.mpf("1e-7")))) - mp.mpf(gmin(float(tv - mp.mpf("1e-7"))))) / mp.mpf("2e-7")
    integrand = lambda tv: mp.log(abs(gprime(tv)) + 1)**2
    val = mp.quad(integrand, [-1, -0.5, 0, 0.5, 1])
    ok2 = abs(val - mp.mpf(str(sp.N(sp.sympify(md_answer(1, "2"), locals=NS), 20)))) < 1e-5
    ok3, msg3 = q3()
    report(1, ok1 and ok2 and ok3, f"g=t^2 확인, ∫={mp.nstr(val, 10)} (수치미분 오차 허용 1e-5); " + msg3)


def q3():
    # n(k) 의 불연속점 = h 의 극값들의 값. 극솟값 2개(같음)=0, 극댓값=1, h(1)=0 인 h(x)=(x-p)^2(x-q)^2 를 풀어 h(4) 계산
    P, Q = sp.symbols("P Q", real=True)
    h = (x - P)**2 * (x - Q)**2
    # 극대점은 (P+Q)/2, 극댓값 ((P-Q)/2)^4 = 1, h(1)=0 ⇒ 1∈{P,Q}
    sols = sp.solve([((P - Q) / 2)**4 - 1, (1 - P) * (1 - Q)], [P, Q], dict=True)
    cand = {tuple(sorted((float(s_[P]), float(s_[Q])))) for s_ in sols}          # {(−1,1),(1,3)}
    g0 = 0.0                                                                      # g(0)=min(x^2+|x|)=0
    good = [c for c in cand if abs(float(h.subs({P: c[0], Q: c[1], x: -1})) - g0) > 1e-12]
    ok_c = good == [(1.0, 3.0)]
    hf = sp.lambdify(x, h.subs({P: 1, Q: 3}), "numpy")
    # n(k) 의 불연속점을 다항식 실근 개수로 직접 확인
    def nk(kv):
        co = np.poly1d(np.poly([1, 1, 3, 3])) - kv
        r = np.roots(co)
        return len(set(np.round([z.real for z in r if abs(z.imag) < 1e-6], 5)))
    ks = np.concatenate([np.linspace(-1, 3, 4001), [0.0, 1.0]])
    ns = [nk(kv) for kv in np.linspace(-1, 3, 40001)]
    jumps = [round(float(v), 3) for v, a1, b1 in zip(np.linspace(-1, 3, 40001)[1:], ns[:-1], ns[1:]) if a1 != b1]
    ok_j = len(jumps) <= 4 and jumps and all(min(abs(j - 0), abs(j - 1)) < 1e-3 for j in jumps)
    ok = ok_c and ok_j and close(hf(4.0), ans_num(1, "3"))
    return ok, f"극대 1·극소 0 조건 해 {sorted(cand)} → h(-1)≠g(0)=0 으로 (1,3); n(k) 변화 지점 {sorted(set(jumps))}; h(4)={hf(4.0):.1f}"


def Fint(kv):
    """F(k)=∫_0^k f. 구간 [2(n-1),2n) 에서 f=sin(nπx); 각 구간의 적분은 0 이므로 F(k)=(1-cos(nπk))/(nπ)."""
    n = np.floor(np.asarray(kv, dtype=float) / 2 + 1e-12) + 1
    return (1 - np.cos(n * np.pi * kv)) / (n * np.pi)


def p02():
    mp.mp.dps = 25
    # 1. F(t)=1/(6π) 의 [0,6] 최대근 (격자 + 이분법). F 를 구간 적분 수치로도 교차 확인
    chk = all(abs(Fint(v) - float(mp.quad(lambda z: mp.sin((int(float(z) // 2) + 1) * mp.pi * z), [2 * (int(v // 2)) if False else 0] + [2 * j for j in range(1, int(v // 2) + 1)] + [v]))) < 1e-9 for v in (0.7, 3.3, 5.1, 5.9))
    ts = np.linspace(0, 6, 6000001)
    G = np.array(Fint(ts)) - 1 / (6 * np.pi)
    idx = np.where(G[:-1] * G[1:] < 0)[0]
    tmax = ts[idx[-1]]
    r = mp.findroot(lambda z: Fint(float(z)) - 1 / (6 * mp.pi), tmax)
    ok1 = chk and close(r, ans_num(2, "1"), 1e-9)
    # 2. F(k)=0 인 k 를 [0,12) 에서 모두 구해(구간별 cos(nπk)=1) 18번째·19번째 근으로 m 의 범위 결정
    roots = []
    for n in range(1, 7):
        for j in range(n * (n - 1), n * n + 1):
            kv = 2 * j / n
            if 2 * (n - 1) <= kv < 2 * n and abs(Fint(kv)) < 1e-12:
                roots.append(round(kv, 10))
    roots = sorted(set(roots))
    r18, r19 = roots[17], roots[18]
    parts = re.findall(r"[-\d./]+", md_answer(2, "2").replace("<=", " ").replace("<", " ").replace("m", " "))
    lo_md, hi_md = [float(sp.Rational(q)) for q in parts[:2]]
    ok2 = abs(lo_md - r18) < 1e-9 and abs(hi_md - r19) < 1e-9 and len(roots) >= 19
    report(2, ok1 and ok2, f"최대 t={mp.nstr(r, 10)}; k 근 {len(roots)}개, 18번째={r18:.4f}, 19번째={r19:.4f}")


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
