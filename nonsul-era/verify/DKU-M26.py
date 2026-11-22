#!/usr/bin/env python3
"""DKU-M26 (단국대 2026 모의논술 자연계열) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 문제 조건에서 독립 재계산 후 YAML answer.value 와 비교.
p<번호>() 함수를 추가하면 자동 실행된다."""
import itertools
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "DKU-M26"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/DKU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return str(next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub))


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


import numpy as np


def make_G(al, lam=0.0):
    """연속·기함수 g 의 한 가족: G(t)=t g(t)=(αk/2)(cos kt - cos(2πt/α)) + λ(cos(4πt/α)-cos(2πt/α)), kα=(2j+1)π."""
    def Gf(tv, k):
        return (al * k / 2) * (mp.cos(k * tv) - mp.cos(2 * mp.pi * tv / al)) + lam * (mp.cos(4 * mp.pi * tv / al) - mp.cos(2 * mp.pi * tv / al))
    return Gf


def p01():
    mp.mp.dps = 30
    # 1. 식 ① : d/dx ∫_{x-α}^{x} G(t)dt = G(x)-G(x-α) (G=t g(t)) = f'(x)=αk cos kx.  x=0 → α g(-α)=αk, g 기함수 → g(α)=-k
    al, k = sp.symbols("alpha k", positive=True)
    Gs = sp.Function("G")
    dI = sp.diff(sp.Integral(Gs(t), (t, x - al, x)), x).doit()
    ok_leib = sp.simplify(dI - (Gs(x) - Gs(x - al))) == 0
    kval = 3 * sp.pi                                           # g(α)=-3π ⇒ k=3π
    sol = sp.solveset(sp.cos(kval * al) + 1, al, sp.Interval.open(0, 2))   # x=α: α g(α)=αk cos(kα), g(α)=-k ⇒ cos(kα)=-1
    cand = sorted(sol, key=float)
    # 존재 확인: α=1/3,1,5/3, k=3π 이면 G(t)=(αk/2)(cos kt - cos(2πt/α)) 로 g 가 조건을 만족
    okex = True
    for a in cand:
        av, kv = mp.mpf(sp.N(a, 30)), mp.mpf(sp.N(kval, 30))
        Gf = make_G(av)
        gfun = lambda tv: Gf(tv, kv) / tv if abs(tv) > mp.mpf("1e-12") else mp.mpf(0)
        for xv in (mp.mpf("0.37"), mp.mpf("1.91")):
            lhs = mp.quad(lambda tv: Gf(tv, kv), [xv - av, xv])
            okex = okex and abs(lhs - av * mp.sin(kv * xv)) < mp.mpf("1e-18")
        okex = okex and abs(gfun(av) + 3 * mp.pi) < mp.mpf("1e-18") and abs(gfun(mp.mpf("0.4")) + gfun(-mp.mpf("0.4"))) < mp.mpf("1e-18")
    md = sorted(sp.sympify(md_answer(1, "1").replace("{", "[").replace("}", "]")), key=float)
    ok1 = ok_leib and okex and [sp.simplify(a - b) for a, b in zip(cand, md)] == [0] * 3 and len(cand) == len(md)
    # 2. αk=π, a_n=G(x_n), x_n=2π/k+nα. 임의 α(0.7) 와 가족 g 로 a_11-a_2 를 직접 계산
    av = mp.mpf("0.7")
    kv = mp.pi / av
    Gf = make_G(av)
    xn = lambda nn: 2 * mp.pi / kv + nn * av
    an = lambda nn: Gf(xn(nn), kv)                             # x g(x)=G(x)
    okfam = abs(mp.quad(lambda tv: Gf(tv, kv), [mp.mpf("0.9") - av, mp.mpf("0.9")]) - av * mp.sin(kv * mp.mpf("0.9"))) < mp.mpf("1e-18")
    v2 = an(11) - an(2)
    ok2 = okfam and abs(v2 - mp.mpf(sp.N(sp.sympify(md_answer(1, "2"), locals={"pi": sp.pi}), 30))) < mp.mpf("1e-18")
    # 3. α=1, k=π (x=0 ⇒ g(α)=-k=-π/α). h(x)=-f(x)+∫_0^x (t+x) g(x-t) dt. h'(2α) 를 수치미분하여 2∫_0^{2α}g - π 와 비교
    al1 = mp.mpf(1)
    k1 = mp.pi / al1

    def gam(lam):
        Gf = make_G(al1, lam)
        return lambda tv: Gf(tv, k1) / tv if abs(tv) > mp.mpf("1e-10") else mp.mpf(0)

    def intg(lam):
        gg = gam(lam)
        return mp.quad(gg, [0, 1, 2])

    def hprime2(lam):
        gg = gam(lam)
        hh = lambda xv: -al1 * mp.sin(k1 * xv) + mp.quad(lambda tv: (tv + xv) * gg(xv - tv), [0, xv / 2, xv])
        d = mp.mpf("1e-6")
        return (hh(2 + d) - hh(2 - d)) / (2 * d)
    mp.mp.dps = 20
    i0, i1 = intg(0), intg(1)
    lam_star = (mp.pi / 2 - i0) / (i1 - i0)                       # ∫_0^{2α} g = π/2 가 되도록 가족을 맞춤
    chk = []
    for lam in (mp.mpf(0), lam_star):
        chk.append((hprime2(lam), 2 * intg(lam) - mp.pi))
    rel = all(abs(a - b) < mp.mpf("1e-6") for a, b in chk)          # h'(2α)=2∫g-π 확인
    ok3 = rel and abs(chk[1][0]) < mp.mpf("1e-6") and abs(intg(lam_star) - mp.mpf(sp.N(sp.sympify(md_answer(1, "3"), locals={"pi": sp.pi}), 20))) < mp.mpf("1e-10")
    report(1, ok1 and ok2 and ok3, f"α={cand} a11-a2={mp.nstr(v2, 8)} h'(2α)=2∫g-π: {[ (mp.nstr(a, 6), mp.nstr(b, 6)) for a, b in chk]} ∫g={mp.nstr(intg(lam_star), 8)}")


E = float(np.e)


def gnum(tv, N=200001):
    xs = np.linspace(0.0, 1.0, N)
    fv = xs * np.exp(2 * xs) - np.abs(np.exp(2 * xs) - np.exp(tv))
    return fv.min()


def gform(tv):
    """분석식(기울기 변화를 보이기 위한 구간별 표현) — 수치 최소값과의 일치를 p02 에서 확인한다."""
    return np.exp(tv) - E / 2 if tv < np.log((2 + E) / 4) else 1 - np.exp(tv)


def p02():
    # 1. g(t)=min_{[0,1]} f 를 격자로 직접 계산 → 꺾임점 탐색
    ts = np.linspace(-4, 3, 1401)
    gv = np.array([gnum(v) for v in ts])
    d2 = np.abs(gv[2:] - 2 * gv[1:-1] + gv[:-2]) / (ts[1] - ts[0])**2 * (ts[1] - ts[0])      # 기울기 변화량
    i = int(np.argmax(d2)) + 1
    kink_num = ts[i]
    err = max(abs(gnum(v) - gform(v)) for v in np.linspace(-6, 3, 121))
    alpha_sym = sp.log((2 + sp.E) / 4)
    sol_eq = sp.solve(sp.exp(t) - sp.E / 2 - (1 - sp.exp(t)), t)
    ok1 = err < 1e-6 and abs(kink_num - float(alpha_sym)) < 0.01 and sp.simplify(sol_eq[0] - alpha_sym) == 0 and \
        sp.simplify(sp.sympify(md_answer(2, "1"), locals={"E": sp.E}) - alpha_sym) == 0
    # 2. h 의 불연속점 = |g(x)-k|=1 인 x (그 점에서 h=1/2 이고 주변은 0 또는 1). 개수 = #{g=k+1}+#{g=k-1}
    M = (2 - E) / 4                                             # g 의 최댓값 g(α)
    al = float(alpha_sym)

    def npre(c):
        left = 1 if (c + E / 2 > 1e-15 and c < M - 1e-15) else 0
        right = 1 if (1 - c > 0 and c <= M + 1e-15) else 0
        return left + right

    def cnt(kv):
        return npre(kv + 1) + npre(kv - 1)
    xs = np.concatenate([np.linspace(-40, al, 400001), np.linspace(al, 3, 400001)])
    gx = np.array([gform(v) for v in xs[::40]])
    xs2 = xs[::40]

    def cnt_grid(kv):
        n_ = 0
        for c in (kv + 1, kv - 1):
            s_ = np.sign(gx - c)
            n_ += int(np.sum(s_[:-1] * s_[1:] < 0))
        return n_
    rng = np.random.default_rng(0)
    agree = all(cnt(kv) == cnt_grid(kv) for kv in rng.uniform(-4, 1, 60))
    ks = np.linspace(-4, 1, 500001)
    good = [kv for kv in ks if cnt(kv) == 3]
    lo, hi = min(good), max(good)
    parts = [sp.sympify(p_.strip().replace("k", "").strip() or "0", locals={"E": sp.E}) for p_ in md_answer(2, "2").split("<")]
    lo_md, hi_md = float(parts[0]), float(parts[2])
    ok2 = agree and abs(lo - lo_md) < 1e-4 and abs(hi - hi_md) < 1e-4 and cnt(lo_md) != 3 and cnt(hi_md) != 3 \
        and cnt((lo_md + hi_md) / 2) == 3
    report(2, ok1 and ok2, f"g 꺾임점≈{kink_num:.3f} (α={al:.4f}) max|g_num-g_form|={err:.1e}; k∈({lo:.4f},{hi:.4f}) 공식 ({lo_md:.4f},{hi_md:.4f}) 끝점 개수 {cnt(lo_md)},{cnt(hi_md)}")


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
