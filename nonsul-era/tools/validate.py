#!/usr/bin/env python3
"""문항 파일 검사기.

사용:
  python3 tools/validate.py                      # 저장소 전체 + books/*.yaml
  python3 tools/validate.py items/SSU/T03        # 일부 폴더(부모·교재 참조는 저장소 전체 기준으로 확인)
  python3 tools/validate.py --book books/SSU-YUYU-2028.yaml --strict   # 교재 출고 직전
옵션: --quiet(요약만) --strict(WARN 도 실패)

ERROR(종료코드 1): 형식·ID·배점·필수 항목·검수 기록 위반 — 고치기 전에는 DB 빌드·교재 조판 금지
WARN           : 사람이 판단할 것(범위 추정 미확정, 세트 분포, 범위 밖 도구 의심 등)
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib import (GEOM_PAT, ID_ORIGINAL, ID_PAST, KIT, REQUIRED_SECTIONS, STAT_TOOL_PAT,  # noqa: E402
                 STATUS_ORDER, all_item_ids, item_seq, load_books, load_item, load_taxonomy, parse_rubric)

REQUIRED_FIELDS = ["id", "kind", "school", "level", "form", "points", "status", "version", "created", "updated"]
REQUIRED_REVIEWS = ("r1", "r2a", "r2b", "verify")     # verified 이상이 되려면 모두 result: pass
REVIEW_STAGES = {"r1", "r2a", "r2b", "verify", "sample", "errata", "legacy_verify"}
REVIEW_RESULTS = {"pass", "fail", "fixed"}
ERRATA_SEVERITY = {"typo", "wording", "answer", "rubric", "scope"}
ERRATA_STATUS = {"open", "confirmed", "rejected", "fixed"}


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


def status_at_least(st: str | None, floor: str) -> bool:
    return st in STATUS_ORDER and STATUS_ORDER.index(st) >= STATUS_ORDER.index(floor)


def check_item(f: pathlib.Path, tax: dict, E: list, W: list) -> dict | None:
    try:
        meta, sec, body = load_item(f)
    except Exception as e:  # noqa: BLE001
        E.append(f"파싱 실패: {e}")
        return None
    for fld in REQUIRED_FIELDS:
        if meta.get(fld) in (None, ""):
            E.append(f"필수 필드 누락: {fld}")
    iid = str(meta.get("id", ""))
    kind, sch, typ = meta.get("kind"), meta.get("school"), meta.get("type")

    # ── ID ──
    if f.stem != iid:
        E.append(f"파일명({f.stem}) ≠ id({iid})")
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
    if typ and kind in ("past", "mock") and typ not in (tax["types"].get(sch) or {}):
        E.append(f"유형 미등록: {sch}/{typ}")
    for fld, allowed in (("level", tax["levels"]), ("form", tax["forms"]), ("status", tax["statuses"])):
        if meta.get(fld) not in allowed:
            E.append(f"{fld} 값 오류: {meta.get(fld)}")
    if meta.get("scope_2028", "unknown") not in tax["scope_2028"]:
        E.append(f"scope_2028 값 오류: {meta.get('scope_2028')}")
    subjects = meta.get("subjects") or []
    for s in subjects:
        if s not in tax["subjects_2015"]:
            E.append(f"subjects 값 오류(2015 개정 과목명): {s}")
    if not subjects:
        W.append("subjects 비어 있음")

    # ── 본문 ──
    for s in REQUIRED_SECTIONS:
        if not sec.get(s, "").strip():
            E.append(f"본문 절 누락/빈칸: ## {s}")
    if sch == "SJU" and sec.get("제시문", "").strip():
        W.append("세종대 문항인데 제시문이 있음")
    if sch == "DGU" and kind == "original" and "【가】" not in sec.get("제시문", ""):
        W.append("동국대 문항인데 【가】 제시문 없음")
    E.extend(check_latex(body))

    # ── 배점 ──
    pts = meta.get("points") or 0
    rub, bad = parse_rubric(sec.get("채점표", ""))
    for b in bad:
        E.append(f"채점표 행을 읽지 못함(배점 칸은 정수): {b[:60]}")
    if not rub:
        E.append("채점표 행이 없음")
    elif sum(r["points"] for r in rub) != pts:
        E.append(f"채점표 합 {sum(r['points'] for r in rub)} ≠ 배점 {pts}")
    dup = [k for k, c in collections.Counter(r["step"] for r in rub).items() if c > 1]
    if dup:
        E.append(f"채점 단계 표시 중복 {dup} — 소문항이 바뀌어도 ①②③… 을 이어서 번호")
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

    # ── 검수 기록 형식 ──
    reviews = meta.get("reviews") or []
    for i, r in enumerate(reviews, 1):
        if not isinstance(r, dict):
            E.append(f"reviews[{i}] 형식 오류")
            continue
        if r.get("stage") not in REVIEW_STAGES:
            E.append(f"reviews[{i}] stage 값 오류: {r.get('stage')}")
        if r.get("result") not in REVIEW_RESULTS:
            E.append(f"reviews[{i}] result 값 오류(pass/fail/fixed): {r.get('result')}")
        if not r.get("reviewer") or not r.get("date"):
            E.append(f"reviews[{i}] reviewer·date 필수")

    # ── 상태·검수 기록 ──
    st = meta.get("status")
    if status_at_least(st, "verified"):
        ref = str(meta.get("verify") or "")
        path_part, _, fn = ref.partition("::")
        vfile = KIT / path_part if path_part else None
        if not vfile or not vfile.exists():
            E.append(f"status={st} 인데 검산 스크립트 없음: {ref}")
        elif not fn or not re.search(rf"^def {re.escape(fn)}\(", vfile.read_text(encoding="utf-8"), re.M):
            E.append(f"검산 함수 {fn or '(미지정)'} 가 {path_part} 에 없음")
        last, pos = {}, {}
        for i, r in enumerate(reviews):               # 단계별 마지막 기록이 기준
            if isinstance(r, dict):
                last[r.get("stage")], pos[r.get("stage")] = r.get("result"), i
        for need in REQUIRED_REVIEWS:
            if last.get(need) != "pass":
                E.append(f"status={st} 인데 reviews 의 {need} 마지막 기록이 pass 가 아님({last.get(need)})")
        if "verify" in pos and any(pos.get(s, -1) > pos["verify"] for s in ("r1", "r2a", "r2b")):
            E.append("verify 기록이 마지막 r1·r2 기록보다 앞섬 — 검토·수정 후 검산을 다시 돌릴 것")

    # ── 사람 판단 필요 ──
    if not meta.get("scope_confirmed", False):
        W.append(f"scope_2028={meta.get('scope_2028')} 은 추정값(scope_confirmed: false)")
    excl = (tax["schools"].get(sch) or {}).get("exclude_tools") or []
    sol = sec.get("모범답안", "")
    if "기하" in excl and GEOM_PAT.search(sol):
        W.append("모범답안에 기하 도구(벡터·내적 등) 의심 — 이 학교 출제범위 밖일 수 있음")
    if "확률과통계" in excl and STAT_TOOL_PAT.search(sec.get("논제", "") + sol):
        W.append("확률과통계 도구(확률·기댓값 등) 의심 — 이 학교 출제범위 밖일 수 있음")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", type=pathlib.Path)
    ap.add_argument("--book", type=pathlib.Path, help="이 교재에 실린 문항만 검사 + 전부 verified 이상인지 확인")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    tax = load_taxonomy()
    repo_ids = all_item_ids()
    errors: dict[str, list[str]] = collections.defaultdict(list)
    warns: dict[str, list[str]] = collections.defaultdict(list)

    id_to_file = {p.stem: p for p in (KIT / "items").rglob("*.md")}
    if args.book:
        book = yaml.safe_load(args.book.read_text(encoding="utf-8")) or {}
        ids = [it["id"] for it in book.get("items", [])]
        missing = [i for i in ids if i not in id_to_file]
        for i in missing:
            errors[f"[교재 {args.book.name}]"].append(f"없는 문항 {i}")
        files = [id_to_file[i] for i in ids if i in id_to_file]
    else:
        roots = args.paths or [KIT / "items"]
        files = sorted({f for r in roots for f in ([r] if r.is_file() else r.rglob("*.md"))})

    seen_legacy: dict[str, str] = {}
    metas: dict[str, dict] = {}
    by_set = collections.defaultdict(list)
    for f in files:
        key = str(f.relative_to(KIT)) if f.is_relative_to(KIT) else str(f)
        meta = check_item(f, tax, errors[key], warns[key])
        if meta is None:
            continue
        iid = str(meta.get("id"))
        if iid in metas:
            errors[key].append(f"ID 중복: {iid}")
        lid = meta.get("legacy_id")
        if lid:
            if lid in seen_legacy:
                errors[key].append(f"legacy_id 중복: {lid} ({seen_legacy[lid]})")
            seen_legacy[lid] = iid
        if meta.get("parent") and meta["parent"] not in repo_ids:
            errors[key].append(f"parent {meta['parent']} 가 저장소에 없음")
        if args.book and not status_at_least(meta.get("status"), "verified"):
            errors[key].append(f"교재 수록 문항인데 status={meta.get('status')} (verified 이상 필요)")
        metas[iid] = meta
        seq = item_seq(iid)
        rep = meta.get("replaces")                   # 폐기 문항을 대신하는 문항은 원래 세트로 계산
        if rep:
            if rep not in repo_ids:
                errors[key].append(f"replaces {rep} 가 저장소에 없음")
            elif item_seq(rep):
                seq = item_seq(rep)
        if meta.get("kind") in ("original", "variant") and seq:
            by_set[(meta["school"], meta["type"], (seq - 1) // 10)].append(meta)

    # ── 세트 단위 게이트(G2·G3): 순번 001~010, 011~020 … 을 한 세트로 ──
    for (sch, typ, k), ms in sorted(by_set.items()):
        key = f"[세트 {sch}-{typ} {k * 10 + 1:03d}~{k * 10 + 10:03d}]"
        live = [m for m in ms if m.get("status") != "retired"]
        if len(live) == 10:
            lv = collections.Counter(m.get("level") for m in live)
            if (lv["L4"], lv["L5"], lv["L6"]) != (2, 6, 2):
                warns[key].append(f"난이도 분포 L4·L5·L6 = {lv['L4']}·{lv['L5']}·{lv['L6']} (기준 2·6·2)")
            fm = collections.Counter(m.get("form") for m in live)
            if len(fm) < 4:
                warns[key].append(f"형식 종류 {len(fm)}개 (4종 이상)")
            over = [f for f, c in fm.items() if c > 3]
            if over:
                warns[key].append(f"같은 형식 3회 초과: {over}")

    # ── 정오표 파일 검사(전체 검사일 때) ──
    if not args.paths and not args.book and (KIT / "errata").exists():
        for ef in sorted((KIT / "errata").glob("*.yaml")):
            key = f"[정오표 {ef.name}]"
            try:
                e = yaml.safe_load(ef.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as ex:
                errors[key].append(f"YAML 오류: {ex}")
                continue
            for k in ("item_id", "reported", "description"):
                if not e.get(k):
                    errors[key].append(f"필수 키 누락: {k}")
            if e.get("item_id") and e["item_id"] not in repo_ids:
                errors[key].append(f"없는 문항 {e['item_id']}")
            if e.get("severity") is not None and e["severity"] not in ERRATA_SEVERITY:
                errors[key].append(f"severity 값 오류: {e['severity']} ({'/'.join(sorted(ERRATA_SEVERITY))})")
            if e.get("status", "open") not in ERRATA_STATUS:
                errors[key].append(f"status 값 오류: {e.get('status')}")

    # ── 교재 배치표 참조 검사(전체 검사일 때) ──
    if not args.paths and not args.book:
        for bf, b in load_books():
            key = f"[교재 {bf.name}]"
            ids = [it.get("id") for it in b.get("items", [])]
            for i in ids:
                if i not in repo_ids:
                    errors[key].append(f"없는 문항 {i}")
            d = [i for i, c in collections.Counter(ids).items() if c > 1]
            if d:
                errors[key].append(f"같은 문항 중복 수록 {d}")

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
