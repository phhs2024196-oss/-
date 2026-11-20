#!/usr/bin/env python3
"""문항 파일 검사기. 사용: python3 tools/validate.py [items 하위 경로 ...] [--quiet] [--strict]

ERROR(종료코드 1): 형식·ID·배점·필수 항목 위반 — 고치기 전에는 DB 빌드·교재 조판 금지
WARN           : 사람이 판단할 것(범위 추정 미확정, 유형 분포, 범위 밖 도구 의심 등)
--strict       : WARN 도 실패로 처리 (교재 출고 직전에 사용)
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib import (GEOM_PAT, ID_ORIGINAL, ID_PAST, KIT, REQUIRED_SECTIONS, STATUS_ORDER,  # noqa: E402
                 load_item, load_taxonomy, parse_rubric)

REQUIRED_FIELDS = ["id", "kind", "school", "level", "form", "points", "status", "version", "created", "updated"]


def check_latex(text: str) -> list[str]:
    errs = []
    if re.search(r"(?<!\\)\\\(|(?<!\\)\\\[", text):   # \\[2mm] (줄바꿈 간격)은 허용
        errs.append("\\( 또는 \\[ 수식 구분자 사용 금지 ($, $$ 사용)")
    if re.search(r"\\tag\b|\\label\b", text):
        errs.append("\\tag/\\label 사용 금지")
    no_display = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)
    if no_display.count("$") % 2:
        errs.append("인라인 $ 개수가 홀수 (닫히지 않은 수식)")
    for m in re.finditer(r"(?<!\\)\$([^$\n]+?)\$", no_display):
        inner = m.group(1)
        if inner != inner.strip():
            errs.append(f"$ 바로 안쪽 공백 금지: ${inner}$")
            break
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", type=pathlib.Path)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    tax = load_taxonomy()
    roots = args.paths or [KIT / "items"]
    files = sorted({f for r in roots for f in ([r] if r.is_file() else r.rglob("*.md"))})

    errors: dict[str, list[str]] = collections.defaultdict(list)
    warns: dict[str, list[str]] = collections.defaultdict(list)
    seen_ids: dict[str, pathlib.Path] = {}
    seen_legacy: dict[str, str] = {}
    metas: dict[str, dict] = {}
    by_type = collections.defaultdict(list)

    for f in files:
        key = str(f.relative_to(KIT)) if f.is_relative_to(KIT) else str(f)
        try:
            meta, sec, body = load_item(f)
        except Exception as e:  # noqa: BLE001
            errors[key].append(f"파싱 실패: {e}")
            continue
        E, W = errors[key], warns[key]
        for fld in REQUIRED_FIELDS:
            if meta.get(fld) in (None, ""):
                E.append(f"필수 필드 누락: {fld}")
        iid = str(meta.get("id", ""))
        kind, sch, typ = meta.get("kind"), meta.get("school"), meta.get("type")

        # ── ID ──
        if iid in seen_ids:
            E.append(f"ID 중복: {iid} ({seen_ids[iid]})")
        seen_ids[iid] = f
        if f.stem != iid:
            E.append(f"파일명({f.stem}) ≠ id({iid})")
        lid = meta.get("legacy_id")
        if lid:
            if lid in seen_legacy:
                E.append(f"legacy_id 중복: {lid} ({seen_legacy[lid]})")
            seen_legacy[lid] = iid
        if kind in ("original", "variant"):
            m = ID_ORIGINAL.match(iid)
            if not m:
                E.append(f"ID 형식 오류(학교-Txx-순번3자리): {iid}")
            elif (m["school"], m["type"]) != (sch, typ):
                E.append(f"ID 의 학교/유형({m['school']}/{m['type']}) ≠ 필드({sch}/{typ})")
        elif kind in ("past", "mock"):
            m = ID_PAST.match(iid)
            if not m:
                E.append(f"기출 ID 형식 오류(학교-G26-04 / 학교-M27-02 / 학교-G26A-01): {iid}")
            elif m["school"] != sch or (m["src"] == "G") != (kind == "past"):
                E.append("기출 ID 의 학교·출처구분(G=past, M=mock)이 필드와 다름")
            if not meta.get("source"):
                E.append("past/mock 은 source(학교·연도·번호·URL) 필수")
        else:
            E.append(f"kind 값 오류: {kind}")
        if kind == "variant" and not meta.get("parent"):
            E.append("variant 는 parent(원 문항 ID) 필수")

        # ── 분류값 ──
        if sch not in tax["schools"]:
            E.append(f"school 코드 미등록: {sch}")
        if kind in ("original", "variant") and typ not in (tax["types"].get(sch) or {}):
            E.append(f"유형 미등록: {sch}/{typ} (taxonomy.yaml types 에 먼저 추가)")
        if meta.get("level") not in tax["levels"]:
            E.append(f"level 값 오류: {meta.get('level')}")
        if meta.get("form") not in tax["forms"]:
            E.append(f"form 값 오류: {meta.get('form')}")
        if meta.get("status") not in tax["statuses"]:
            E.append(f"status 값 오류: {meta.get('status')}")
        if meta.get("scope_2028", "unknown") not in tax["scope_2028"]:
            E.append(f"scope_2028 값 오류: {meta.get('scope_2028')}")
        for s in meta.get("subjects") or []:
            if s not in tax["subjects_2015"]:
                E.append(f"subjects 값 오류(2015 개정 과목명): {s}")

        # ── 본문 ──
        for s in REQUIRED_SECTIONS:
            if not sec.get(s, "").strip():
                E.append(f"본문 절 누락/빈칸: ## {s}")
        if sch == "SJU" and sec.get("제시문", "").strip():
            W.append("세종대 문항인데 제시문이 있음")
        if sch == "DGU" and "【가】" not in sec.get("제시문", "") and kind == "original":
            W.append("동국대 문항인데 【가】 제시문 없음")
        E.extend(check_latex(body))

        # ── 배점 ──
        pts = meta.get("points") or 0
        rub = parse_rubric(sec.get("채점표", ""))
        if not rub:
            E.append("채점표 행을 읽지 못함")
        elif sum(r["points"] for r in rub) != pts:
            E.append(f"채점표 합 {sum(r['points'] for r in rub)} ≠ 배점 {pts}")
        sub = meta.get("sub_points") or []
        if sub:
            if sum(sub) != pts:
                E.append(f"sub_points 합 {sum(sub)} ≠ 배점 {pts}")
            if rub and all(r["sub_question"] for r in rub):
                for i, sp in enumerate(sub, 1):
                    got = sum(r["points"] for r in rub if r["sub_question"] == f"({i})")
                    if got != sp:
                        E.append(f"소문항 ({i}) 채점 소계 {got} ≠ {sp}")
            else:
                W.append("다소문항인데 채점표에 소문항 열이 없음(소계 검사 불가)")

        # ── 상태·검산 ──
        st = meta.get("status")
        if st in STATUS_ORDER and STATUS_ORDER.index(st) >= STATUS_ORDER.index("verified"):
            ref = str(meta.get("verify") or "")
            vfile = KIT / ref.split("::")[0] if ref else None
            if not vfile or not vfile.exists():
                E.append(f"status={st} 인데 검산 스크립트 없음: {ref}")
            stages = {(r.get("stage"), r.get("result")) for r in meta.get("reviews") or []}
            for need in ("r1", "r2a", "r2b"):
                if (need, "pass") not in stages and (need, "fixed") not in stages:
                    E.append(f"status={st} 인데 reviews 에 {need} 통과 기록 없음")

        # ── 사람 판단 필요 ──
        if not meta.get("scope_confirmed", False):
            W.append(f"scope_2028={meta.get('scope_2028')} 은 추정값(scope_confirmed: false)")
        excl = (tax["schools"].get(sch) or {}).get("exclude_tools") or []
        if "기하" in excl and GEOM_PAT.search(sec.get("모범답안", "")):
            W.append("모범답안에 기하 도구(벡터·내적 등) 의심 — 이 학교 출제범위 밖일 수 있음")

        metas[iid] = meta
        if kind in ("original", "variant"):
            by_type[(sch, typ)].append(meta)

    # ── 유형 단위 게이트(G2·G3): 10문항 단위 세트에만 적용 ──
    for (sch, typ), ms in sorted(by_type.items()):
        key = f"[유형 {sch}/{typ}]"
        live = [m for m in ms if m.get("status") != "retired"]
        if len(live) == 10:
            lv = collections.Counter(m.get("level") for m in live)
            if (lv["L4"], lv["L5"], lv["L6"]) != (2, 6, 2):
                warns[key].append(f"난이도 분포 L4·L5·L6 = {lv['L4']}·{lv['L5']}·{lv['L6']} (기준 2·6·2)")
        fm = collections.Counter(m.get("form") for m in live)
        if live and len(fm) < 4:
            warns[key].append(f"형식 종류 {len(fm)}개 (4종 이상 권장)")
        over = [f for f, c in fm.items() if c > 3]
        if over:
            warns[key].append(f"같은 형식 3회 초과: {over}")

    for iid, m in metas.items():
        if m.get("parent") and m["parent"] not in metas:
            errors[iid].append(f"parent {m['parent']} 가 DB 에 없음")

    n_err = sum(len(v) for v in errors.values())
    n_warn = sum(len(v) for v in warns.values())
    if not args.quiet:
        for k in sorted(set(errors) | set(warns)):
            for e in errors.get(k, []):
                print(f"ERROR {k}: {e}")
            for w in warns.get(k, []):
                print(f"WARN  {k}: {w}")
    wc = collections.Counter(re.sub(r"[:(].*", "", w) for v in warns.values() for w in v)
    print(f"SUMMARY files={len(files)} errors={n_err} warnings={n_warn}")
    for w, c in wc.most_common():
        print(f"  WARN x{c}: {w}")
    return 1 if n_err or (args.strict and n_warn) else 0


if __name__ == "__main__":
    sys.exit(main())
