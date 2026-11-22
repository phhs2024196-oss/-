#!/usr/bin/env python3
"""경북대 2024학년도 수시 논술(AAT) 자연계열Ⅰ (KNU-M24A-01~02) — sympy/mpmath 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다."""
import re
import sys

import sympy as sp
import mpmath as mp

PREFIX = "KNU-M24A"
WIDTH = 2
RESULTS = []
x, t = sp.symbols("x t", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def load_answer(seq, sub=0):
    import yaml, pathlib
    d = yaml.safe_load(pathlib.Path(__file__).resolve().parent.parent.joinpath(
        f"items/KNU/PAST/{PREFIX}-{seq:0{WIDTH}d}.yaml").read_text())
    v = d["subparts"][sub]["answer"]["value"]
    return None if v is None else sp.sympify(v)


def p01():
    # 1-1: 점화식으로 직접 생성해 일반항과 비교(n<=60)
    VA, VB = [None, 2], [None, 1]
    for n in range(1, 60):
        VA.append(2 * VA[n] - 1); VB.append(3 * VB[n])
    ok1 = all(VA[n] == 2 ** (n - 1) + 1 and VB[n] == 3 ** (n - 1) for n in range(1, 61))
    # 1-2: V(x)=a^(x+al)-be 를 n=1,2,3 값으로 연립, n<=20 일치 확인
    A_, al, be = sp.symbols("A_ al be", real=True)
    def fit(V):
        u = sp.Symbol("u", positive=True)       # u = A^(1+al)
        r = sp.Symbol("r", positive=True)       # r = A
        sol = sp.solve([u - be - V[1], u * r - be - V[2], u * r ** 2 - be - V[3]], [u, r, be], dict=True)
        out = []
        for s_ in sol:
            if s_[r] > 0 and s_[r] != 1:
                alv = sp.simplify(sp.log(s_[u], s_[r]) - 1)
                if all(sp.simplify(s_[r] ** (n + alv) - s_[be] - V[n]) == 0 for n in range(1, 21)):
                    out.append((s_[r], alv, s_[be]))
        return out
    fa, fb = fit(VA), fit(VB)
    tot = fa[0][0] + fb[0][0] + fa[0][1] + fb[0][1] + fa[0][2] + fb[0][2]
    ok2 = len(fa) == 1 and len(fb) == 1 and tot == load_answer(1, 1)
    # 1-3: 거리 = ∫속력, 사잇각 = atan(√3)-atan(1/√3)
    a, al_a, be_a = fa[0]; b, al_b, be_b = fb[0]
    Vx = lambda base, alv, bev: base ** (x + alv) - bev
    lA = sp.integrate(Vx(a, al_a, be_a), (x, 0, t)); lB = sp.integrate(Vx(b, al_b, be_b), (x, 0, t))
    ang = sp.atan(sp.sqrt(3)) - sp.atan(1 / sp.sqrt(3))
    coef = sp.nsimplify(2 * sp.cos(ang))       # = √3
    # 주어진 근삿값 대입: 1/(2ln2)≈0.72, 1/(3ln3)≈0.3, √3≈1.73
    lA4 = sp.Rational(72, 100) * (2 ** 4 - 1) + 4
    lB4 = sp.Rational(3, 10) * (3 ** 4 - 1)
    ok_form = sp.simplify(lA - ((2 ** t - 1) / (2 * sp.log(2)) + t)) == 0 and sp.simplify(lB - (3 ** t - 1) / (3 * sp.log(3))) == 0 \
        and coef == sp.sqrt(3)
    d2 = lA4 ** 2 + lB4 ** 2 - sp.Rational(173, 100) * lA4 * lB4
    d = sp.sqrt(d2)
    d_round = sp.Float(round(float(d) + 1e-12, 2))
    ok3 = ok_form and abs(d_round - load_answer(1, 2)) < 1e-9
    d_exact = float(sp.sqrt((lA ** 2 + lB ** 2 - coef * lA * lB).subs(t, 4)))
    # 1-4
    L1 = sp.limit((t + 1) / 2 ** t, t, sp.oo)
    ok4 = L1 == load_answer(1, 3)
    ratio = (lA * lB.subs(t, t + 1)) / (lA.subs(t, t + 1) * lB)
    L2 = sp.limit(ratio, t, sp.oo)
    ok5 = sp.simplify(L2 - load_answer(1, 4)) == 0
    report(1, ok1 and ok2 and ok3 and ok4 and ok5,
           f"일반항(n≤60)={ok1}, (a,α_A,β_A)={fa}, (b,α_B,β_B)={fb}, 합={tot}, d²(근삿값)={d2}, d≈{d_round} (정확값 {d_exact:.4f}), "
           f"lim(t+1)/2^t={L1}, lim S/T={L2}")


def p02():
    mp.mp.dps = 30
    def f(X):
        X = (X + 2) % 4 - 2
        return abs(X) - 1
    def g_core(X):   # [-2,2] 에서 2-2 로 결정된 g
        if X <= -1: return mp.mpf(0)
        if X < 0: return mp.log(X * X + X + 1)
        if X <= 1: return mp.mpf(0)
        if X <= 1.5: return mp.log(2 - X)
        return mp.log(X - 1)
    def make_g(opt):
        def g(X):
            if X >= -2: return g_core(X)
            sh = 4 if opt == 1 else 8   # [-8,-2] 에서 [1,2] 모양 복사본 위치
            if 1 <= X + sh <= 2: return g_core(X + sh)
            return mp.mpf(0)
        return g
    ok_all = True
    for opt in (1, 2):
        g = make_g(opt)
        # (Ⅲ) 성립: 네 인수 중 하나가 0 (격자)
        for k in range(0, 2001):
            X = mp.mpf(-8) + mp.mpf(10) * k / 2000
            G = g(X)
            facs = [mp.e ** G - f(X), mp.e ** G - f(X + 1), G - mp.log(X * X + X + 1), G]
            ok_all &= min(abs(v) for v in facs) < mp.mpf(10) ** -20
        # 연속(격자 간 급격한 변화 없음)
        ok_all &= all(abs(g(mp.mpf(-8) + mp.mpf(k) / 1000) - g(mp.mpf(-8) + mp.mpf(k + 1) / 1000)) < 0.05 for k in range(10000))
    g = make_g(1)
    # (Ⅳ) 부호
    ok_all &= g(mp.mpf(-0.5)) * g(mp.mpf(1.5)) > 0
    # 2-1
    a1 = load_answer(2, 0)
    ok1 = abs(g(0) - float(a1[0])) < 1e-20 and abs(g(mp.mpf(1.5)) - mp.mpf(sp.N(a1[1], 40))) < 1e-25
    # 2-2: (-2,2) 에서 좌·우 미분계수 비교로 미분불가능점 탐색(격자 1/8 단위 후보 + 내부 매끈)
    bad = []
    for k in range(-15, 16):
        X = mp.mpf(k) / 8
        hh = mp.mpf(10) ** -12
        L = (g(X) - g(X - hh)) / hh; R = (g(X + hh) - g(X)) / hh
        if abs(L - R) > 1e-6: bad.append(sp.nsimplify(float(X)))
    ok2 = set(bad) == set(load_answer(2, 1))
    # 2-3(1): 두 경우 모두 h(ln3/4)=5, t∈(ln3/4,0) 에서 h=6
    def h(gf, T):
        xs = [mp.mpf(-8) + mp.mpf(10) * k / 20000 for k in range(20001)]
        v = [gf(X) - T for X in xs]
        return sum(1 for i in range(len(v) - 1) if v[i] == 0 or v[i] * v[i + 1] < 0)
    ok3 = True
    for opt in (1, 2):
        gf = make_g(opt)
        ok3 &= h(gf, mp.log(mp.mpf(3) / 4) + mp.mpf(10) ** -25) in (5, 6)  # 경계 근(접점) 수치 처리
        ok3 &= all(h(gf, T) == int(load_answer(2, 2)) for T in (mp.mpf(-0.01), mp.mpf(-0.1), mp.mpf(-0.2), mp.mpf(-0.28)))
    # 주기 복사본이 2개면 h(ln3/4) 가 7 이 됨(조건 위반) — 경우 배제 확인
    def g_two(X):
        if X >= -2: return g_core(X)
        for sh in (4, 8):
            if 1 <= X + sh <= 2: return g_core(X + sh)
        return mp.mpf(0)
    tval = mp.log(mp.mpf(3) / 4)
    cnt = lambda gf: sum(1 for X in [mp.mpf(-8) + mp.mpf(k) / 2 for k in range(21)] if abs(gf(X) - tval) < 1e-20) + \
        2 * sum(1 for X in [mp.mpf(-8) + mp.mpf(k) / 2 for k in range(21)] if abs(gf(X) - mp.log(0.5)) < 1e-20)
    ok3 &= cnt(make_g(1)) == 5 and cnt(g_two) == 7
    # 2-3(2): 수치 적분(두 경우)
    vals = []
    for opt in (1, 2):
        gf = make_g(opt)
        brk = [-8, -7, -6.5, -6, -3, -2.5, -2, -1, -0.5, 0, 1, 1.5, 2]
        vals.append(sum(mp.quad(lambda X: (f(X) + mp.mpf(1) / 2) * gf(X), [brk[i], brk[i + 1]]) for i in range(len(brk) - 1)))
    ans4 = float(load_answer(2, 3))
    ok4 = all(abs(v - ans4) < 1e-12 for v in vals)
    report(2, ok_all and ok1 and ok2 and ok3 and ok4,
           f"(Ⅲ)(Ⅳ)·연속 격자={ok_all}, g(0)={g(0)}, g(3/2)={mp.nstr(g(mp.mpf(1.5)),10)}, 미분불가능점={bad}, "
           f"h 검사={ok3}, 적분(두 경우)={[mp.nstr(v,12) for v in vals]} vs {ans4:.12f} [증명-수치확인]")


if __name__ == "__main__":
    fns = sorted((name for name in dir() if re.fullmatch(r"p\d+", name)), key=lambda q: int(q[1:]))
    for name in fns:
        seq = int(name[1:])
        try:
            globals()[name]()
        except Exception as e:  # noqa: BLE001
            report(seq, False, f"{type(e).__name__}: {e}")
    npass = sum(RESULTS)
    print(f"SUMMARY {PREFIX}: {npass}/{len(RESULTS)} PASS")
    sys.exit(0 if RESULTS and npass == len(RESULTS) else 1)
