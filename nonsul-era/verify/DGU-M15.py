#!/usr/bin/env python3
"""DGU-M15 (2015학년도 동국대 모의논술) — sympy 검산.
규칙(db/제작지침.md §6): 정답 하드코딩 비교 금지, 설정으로부터 독립 재계산.
p<순번>() 함수를 추가하기만 하면 자동으로 실행된다(목록·개수 고정 금지)."""
import re
import sys

import numpy as np
import sympy as sp

PREFIX = "DGU-M15"
WIDTH = 2
RESULTS = []
x, t, n = sp.symbols("x t n", real=True)


def report(seq: int, ok: bool, msg: str) -> None:
    RESULTS.append(bool(ok))
    print(f"{PREFIX}-{seq:0{WIDTH}d} {'PASS' if ok else 'FAIL'}  {msg}")

def p01():
    """[증명-수치확인] 소득 오름차순 누적 → 로렌츠 곡선 기울기 = 소득/평균 이 증가(아래로 볼록)."""
    rng = np.random.default_rng(15)
    bad = 0
    for _ in range(200):
        inc = np.sort(rng.lognormal(0, rng.uniform(0.2, 1.5), 5000))
        L = np.concatenate([[0], np.cumsum(inc) / inc.sum()])
        xs = np.linspace(0, 1, len(L))
        slope = np.diff(L) / np.diff(xs)
        mean = inc.mean()
        if not (np.all(np.diff(slope) >= -1e-12) and np.allclose(slope, inc / mean) and np.all(L <= xs + 1e-12)):
            bad += 1
    # 연속 모형: 소득 분위함수 q(p)=e^{p} (증가) → L(p)=∫0^p q / ∫0^1 q, L''=q'/∫q > 0
    p = sp.symbols("p", positive=True)
    q = sp.exp(p)
    Lc = sp.integrate(q, (p, 0, p)) / sp.integrate(q, (p, 0, 1))
    ok_c = sp.simplify(sp.diff(Lc, p, 2) - sp.diff(q, p) / sp.integrate(q, (p, 0, 1))) == 0
    report(1, bad == 0 and ok_c, f"[증명-수치확인] 무작위 소득분포 200개 모두 기울기 단조증가·L≤x (위반 {bad}), 연속모형 L''=q'/평균>0")


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
