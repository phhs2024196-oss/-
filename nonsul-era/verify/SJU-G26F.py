#!/usr/bin/env python3
"""SJU-G26F (세종대 2026학년도 논술고사 자유전공학부-수학문항) — sympy/mpmath 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import mpmath as mp
import sympy as sp

PREFIX = "SJU-G26F"
WIDTH = 2
RESULTS = []
x = sp.symbols("x", real=True)
mp.mp.dps = 25


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def md_parts(no, sub):
    out = {}
    for seg in md_answer(no, sub).split(";"):
        k, v = seg.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def p02():
    loc = {"x": x}
    f = x * (x - 1)**2 * sp.exp(2 * x)
    g = -x / sp.sqrt(1 + x**2)
    # (2-1)
    m1 = md_parts(2, "1")
    gp = sp.diff(g, x)
    lims = (sp.limit(g, x, -sp.oo), sp.limit(g, x, sp.oo))
    rng = [sp.sympify(v) for v in m1["range"].strip("()").split(",")]
    ok1 = sp.simplify(gp - sp.sympify(m1["gp"], locals=loc)) == 0 and \
        all(float(gp.subs(x, v)) < 0 for v in (-50, -1, 0, 2, 40)) and \
        sorted(lims) == rng and m1["range"].startswith("(") and m1["range"].endswith(")")
    # (2-2) 역함수: g(ginv(y))=y 수치 확인, h=0 의 해는 격자 부호변화 + 정밀화
    m2 = md_parts(2, "2")
    ginv = sp.sympify(m2["ginv"], locals=loc)
    gl, fl, gil = (sp.lambdify(x, e, "mpmath") for e in (g, f, ginv))
    ok_inv = all(abs(gl(gil(v)) - v) < 1e-20 and abs(gil(gl(7 * v)) - 7 * v) < 1e-15 for v in map(mp.mpf, ("-0.9", "-0.3", "0.2", "0.77")))
    h = lambda v: gl(fl(gil(v)))
    xs = [-0.999 + 1.998 * i / 4000 for i in range(4001)]
    roots = []
    for u, w in zip(xs, xs[1:]):
        if h(u) == 0:
            roots.append(mp.mpf(u))
        elif h(u) * h(w) < 0:
            roots.append(mp.findroot(h, (u, w), solver="bisect", verify=False))
    # 접하는 근(부호 변화 없음): h' 의 부호가 바뀌는 점 중 h=0 인 곳
    hp = lambda v: mp.diff(h, v)
    for u, w in zip(xs, xs[1:]):
        if hp(u) * hp(w) < 0:
            r = mp.findroot(hp, (u, w), solver="bisect", verify=False)
            if abs(h(r)) < 1e-15:
                roots.append(r)
    md_roots = sorted(float(sp.sympify(v)) for v in m2["roots"].strip("{}").split(","))
    uniq = sorted({round(float(r), 9) for r in roots})
    ok2 = ok_inv and len(uniq) == len(md_roots) and all(abs(a - b) < 1e-9 for a, b in zip(uniq, md_roots))
    # (2-3) h 의 극값을 h' 부호변화로 찾고, 양 끝 극한·치역 확인
    ext = []
    ys = [-0.995 + 1.99 * i / 2000 for i in range(2001)]
    for u, w in zip(ys, ys[1:]):
        if hp(u) * hp(w) < 0:
            r = mp.findroot(hp, (u, w), solver="bisect", verify=False)
            ext.append((r, "max" if hp(u) > 0 else "min", h(r)))
    m3 = md_parts(2, "3")
    md_max = sorted(float(sp.sympify(v)) for v in m3["max"].strip("{}").split(","))
    md_min = float(sp.sympify(m3["min"]))
    got_max = sorted(float(v) for _, k, v in ext if k == "max")
    got_min = [float(v) for _, k, v in ext if k == "min"]
    lim_l = h(mp.mpf(-1) + mp.mpf(10)**-12); lim_r = h(mp.mpf(1) - mp.mpf(10)**-12)
    rr = m3["range"].replace(" ", "")
    lo, hi = [float(sp.sympify(v)) for v in rr.strip("(]").split(",")]
    sup = max(max(got_max), float(lim_r))
    ok3 = len(got_max) == len(md_max) and all(abs(a - b) < 1e-9 for a, b in zip(got_max, md_max)) and \
        len(got_min) == 1 and abs(got_min[0] - md_min) < 1e-9 and abs(float(lim_l) - lo) < 1e-6 and \
        abs(sup - hi) < 1e-9 and rr.startswith("(") and rr.endswith("]") and all(h(v) > lo for v in (-0.99, -0.9, 0.99))
    report(2, ok1 and ok2 and ok3, f"g'={sp.simplify(gp)} roots={uniq} max={[round(v, 10) for v in got_max]} min={[round(v, 10) for v in got_min]} lim(-1+)={float(lim_l):.6f}")


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
