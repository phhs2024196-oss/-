#!/usr/bin/env python3
"""SJU-G24B (세종대 2024학년도 수시 논술 B형) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<번호>() 함수를 추가하면 자동 실행된다."""
import re
import sys

import sympy as sp

PREFIX = "SJU-G24B"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def md_answer(no: int, sub: str) -> str:
    """문항 YAML 의 answer.value 를 읽는다(검산 스크립트에 정답을 적지 않기 위해)."""
    import pathlib
    import yaml
    f = pathlib.Path(__file__).resolve().parents[1] / "items/SJU/PAST" / f"{PREFIX}-{no:0{WIDTH}d}.yaml"
    it = yaml.safe_load(f.read_text(encoding="utf-8"))
    return next(s["answer"]["value"] for s in it["subparts"] if str(s["id"]) == sub)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")


def _seq(a1, N=6):
    s = [a1]
    for k in range(1, N):
        a = s[-1]
        if a < 10:
            s.append(2 * a)
        elif a % 2 == 0:
            s.append(a // 2 + k + 1)
        else:
            s.append((a + 1) // 2)
    return s


def p01():
    # 전수 열거: a1 을 충분히 크게(1..200000) 훑는다. (다)의 역으로 a1 은 a4 에 대해 8배 정도면 충분
    seqs = [_seq(a) for a in range(1, 200001)]
    odd = [(s[1], s[0]) for s in seqs if s[1] % 2 == 1]
    m = min(x for x, _ in odd)
    a1s = sorted(a for x, a in odd if x == m)
    parts = dict(seg.split("=", 1) for seg in md_answer(1, "1").replace(" ", "").split(";"))
    ok1 = int(parts["a2_min"]) == m and sorted(eval(parts["a1"])) == a1s
    ok_a = [s[0] for s in seqs if s[1] % 2 == 0 and s[2] % 2 == 0 and s[1] <= 10 and s[2] <= 10]
    ok2 = max(ok_a) == int(md_answer(1, "2"))
    a4 = [s[3] for s in seqs if s[5] == 12]
    p3 = dict(seg.split("=", 1) for seg in md_answer(1, "3").replace(" ", "").split(";"))
    ok3 = min(a4) == int(p3["min"]) and max(a4) == int(p3["max"])
    report(1, ok1 and ok2 and ok3, f"a2min={m} a1={a1s} (1-2) cases={sorted(ok_a)} (1-3) a4 in [{min(a4)},{max(a4)}] #a1={len(a4)}")


def p02():
    import numpy as np
    X = sp.symbols("X", real=True)
    f = (1 + sp.Abs(X)**3)**-2
    fp, fn = (1 + X**3)**-2, (1 - X**3)**-2
    d1p, d1n = sp.diff(fp, X), sp.diff(fn, X)
    d2p, d2n = sp.diff(fp, X, 2), sp.diff(fn, X, 2)
    # 원점 미분계수(정의)
    okd0 = sp.limit((fp - 1) / X, X, 0, "+") == 0 and sp.limit((fn - 1) / X, X, 0, "-") == 0 and \
        sp.limit(d1p / X, X, 0, "+") == 0 and sp.limit(d1n / X, X, 0, "-") == 0
    c = float(sp.Rational(2, 7)**sp.Rational(1, 3))
    md = md_answer(2, "1").replace(" ", "")
    ok_str = "f1<0:x>0" in md and md.count("(2/7)**(1/3)") == 2
    xs = np.concatenate([np.linspace(-3, -1e-6, 30001), np.linspace(1e-6, 3, 30001)])
    D1 = np.where(xs > 0, sp.lambdify(X, d1p, "numpy")(xs), sp.lambdify(X, d1n, "numpy")(xs))
    D2 = np.where(xs > 0, sp.lambdify(X, d2p, "numpy")(xs), sp.lambdify(X, d2n, "numpy")(xs))
    ok_f1 = np.all((D1 < 0) == (xs > 0))
    pred = ((xs > -c) & (xs < 0)) | ((xs > 0) & (xs < c))
    ok_f2 = np.all(((D2 < 0) == pred) | (np.abs(np.abs(xs) - c) < 1e-3))
    ok1 = okd0 and ok_str and ok_f1 and ok_f2
    # (2) g(x)=∫_x^{x+4} f 의 최대점 수치 탐색
    import mpmath as mp
    F = lambda t: (1 + abs(t)**3)**-2
    g = lambda x: float(mp.quad(F, sorted({x, x + 4} | ({0.0} if x < 0 < x + 4 else set()))))
    grid = np.linspace(-4, 0, 2001)
    gv = np.array([g(x) for x in grid])
    a = grid[np.argmax(gv)]
    ok2 = abs(a - float(sp.sympify(md_answer(2, "2")))) < 1e-3
    # (3) S, g(a) 고정밀 수치
    import mpmath as mp
    mp.mp.dps = 30
    av = int(sp.sympify(md_answer(2, "2")))
    Fm = lambda t: (1 + abs(t)**3)**-2
    S = mp.quad(lambda t: t**3 * Fm(t), [av, 0]) * (-1)
    ga = mp.quad(Fm, [av, 0, av + 4])
    val = ga - 4 * S
    ok3 = abs(val - mp.mpf(str(sp.N(sp.sympify(md_answer(2, "3")), 40)))) < 1e-25
    report(2, ok1 and ok2 and ok3, f"f''<0 cut=±{c:.6f} argmax g~{a:.4f} g(a)-4S={mp.nstr(val, 20)}")


def p03():
    import numpy as np
    al = sp.symbols("alpha", positive=True)
    md = dict(map(str.strip, seg.split("=", 1)) for seg in md_answer(3, "1").split(";"))
    loc = {"alpha": al, "pi": sp.pi}
    md_ang = sp.sympify(md["angle"], locals=loc)
    md_f = sp.sympify(md["f"], locals=loc)
    # 독립 기하: O1=(0,0), O2=(0,1), P=(r sin al, r cos al). 각 O1PO2 가 5π/6 인 r 를 수치로 풀고 β, s, 8자 길이 계산
    def geom(a):
        def brentq(h, lo, hi):
            for _ in range(200):
                mid = (lo + hi) / 2
                if (h(lo) > 0) == (h(mid) > 0):
                    lo = mid
                else:
                    hi = mid
            return (lo + hi) / 2
        def angP(r):
            P = np.array([r * np.sin(a), r * np.cos(a)])
            u = -P; v = np.array([0, 1]) - P
            return np.arccos(u @ v / np.linalg.norm(u) / np.linalg.norm(v))
        r = brentq(lambda r: angP(r) - 5 * np.pi / 6, 1e-9, np.cos(a) - 1e-12)
        P = np.array([r * np.sin(a), r * np.cos(a)])
        s = np.linalg.norm(P - np.array([0, 1]))
        b = np.arccos((np.array([0, -1]) @ (P - np.array([0, 1]))) / s)
        # 접선 사이 예각 확인: 접선 방향은 반지름에 수직
        t1 = np.array([P[1], -P[0]]); t2 = np.array([P[1] - 1, -P[0]])
        acute = np.arccos(abs(t1 @ t2) / np.linalg.norm(t1) / np.linalg.norm(t2))
        return r, s, b, (2 * np.pi - 2 * a) * r + (2 * np.pi - 2 * b) * s, (2 * np.pi - 2 * b) * s, acute
    ok1 = True
    for a in (0.05, 0.2, 0.4, 0.5):
        r, s, b, L, _, acute = geom(a)
        ok1 &= abs(acute - np.pi / 6) < 1e-9 and r < 1 and s < 1
        ok1 &= abs(b - float(md_ang.subs(al, a))) < 1e-9 and abs(L - float(md_f.subs(al, a))) < 1e-9
    # 범위: s<1 ⇔ a<π/6 (경계 직전·직후 확인)
    ok1 &= geom(np.pi / 6 - 1e-4)[1] < 1 and "0<alpha<pi/6" in md_answer(3, "1").replace(" ", "")
    # (2) g' 범위: 수치 미분 + 극한
    gs = lambda a: geom(a)[4]
    A = np.linspace(1e-4, np.pi / 6 - 1e-4, 400)
    gd = np.array([(gs(a + 1e-6) - gs(a - 1e-6)) / 2e-6 for a in A])
    mono = np.all(np.diff(gd) > 0)
    p2 = dict(seg.split("=", 1) for seg in md_answer(3, "2").replace(" ", "").split(";"))
    amax, bmin = float(sp.sympify(p2["a_max"])), float(sp.sympify(p2["b_min"]))
    ok2 = mono and abs(gd[0] - amax) < 1e-2 and abs(gd[-1] - bmin) < 1e-2 and gd.min() > amax and gd.max() < bmin
    # (3) f 최소점
    A2 = np.linspace(1e-3, np.pi / 6 - 1e-3, 20001)
    fl = sp.lambdify(al, md_f, "numpy")
    fv = fl(A2)
    amin = A2[np.argmin(fv)]
    ok3 = abs(amin - float(sp.sympify(md_answer(3, "3")))) < 1e-4 and abs(geom(amin)[3] - fv.min()) < 1e-9
    report(3, ok1 and ok2 and ok3, f"g' in ({gd.min():.5f},{gd.max():.5f}) vs ({amax:.5f},{bmin:.5f}) argmin f={amin:.6f}")


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
