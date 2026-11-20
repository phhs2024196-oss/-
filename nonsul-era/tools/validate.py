#!/usr/bin/env python3
"""문항 검사기 (양식 v2: items/**/*.yaml).

사용:
  python3 tools/validate.py                      # 저장소 전체 + books/*.yaml + errata/*.yaml + 카탈로그 참조
  python3 tools/validate.py items/SSU/T03        # 일부(참조 확인은 저장소 전체 기준)
  python3 tools/validate.py --book books/SSU-YUYU-2028.yaml --strict   # 교재 출고 직전
옵션: --quiet(요약만) --strict(WARN 도 실패)

ERROR : 구조·ID·배점·참조·수식·저작권·검수 기록 위반. 고치기 전에는 빌드·조판 금지
GATE  : v2 완비 항목(마크 종류, 해설 A·B, 오개념, 처방, 출제의도 …) — draft 등에서는 WARN, verified 이상이면 ERROR
WARN  : 사람이 판단할 것(범위 추정, 세트 분포, 범위 밖 도구 의심 …)
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib import (GEOM_PAT, ID_ORIGINAL, ID_PAST, KIT, SCHEMA_VERSION, STAT_TOOL_PAT, STATUS_ORDER,  # noqa: E402
                 all_item_ids, item_files, item_seq, item_texts, load_books, load_catalog, load_item,
                 load_taxonomy)

ERRATA_SEVERITY = {"typo", "wording", "answer", "rubric", "scope"}
ERRATA_STATUS = {"open", "confirmed", "rejected", "fixed"}
GATE_REVIEWS = ("r1_scope", "r2_calc_1", "r2_calc_2", "verify")


def check_latex(text: str) -> list[str]:
    errs = []
    if re.search(r"(?<!\\)\\\(|(?<!\\)\\\[", text):     # \\[2mm] (줄바꿈 간격)은 허용
        errs.append("\\( 또는 \\[ 수식 구분자 사용 금지 ($, $$ 사용)")
    if re.search(r"\\tag\b|\\label\b", text):
        errs.append("\\tag/\\label 사용 금지")
    no_display = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)
    if no_display.replace("\\$", "").count("$") % 2:
        errs.append("인라인 $ 개수가 홀수 (닫히지 않은 수식)")
    for m in re.finditer(r"(?<!\\)\$([^$\n]+?)\$", no_display):
        if m.group(1) != m.group(1).strip():
            errs.append(f"$ 바로 안쪽 공백 금지: ${m.group(1)}$")
            break
    return errs


def at_least(st, floor) -> bool:
    return st in STATUS_ORDER and STATUS_ORDER.index(st) >= STATUS_ORDER.index(floor)


def all_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k not in ("reviews", "legacy"):
                yield from all_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from all_strings(v)


def check_item(f: pathlib.Path, tax: dict, cat: dict, repo_ids: set, E: list, W: list) -> dict | None:
    try:
        it = load_item(f)
    except Exception as e:  # noqa: BLE001
        E.append(f"YAML 파싱 실패(글 값은 | 블록이나 따옴표로): {str(e).splitlines()[0]}")
        return None
    st = it.get("status")
    gated = at_least(st, "verified")

    def gate(msg):                       # v2 완비 항목: verified 이상이면 오류
        (E if gated else W).append(("GATE " if not gated else "") + msg)

    def enum(val, allowed, name, required=True):
        if val is None and not required:
            return
        if val not in allowed:
            E.append(f"{name} 값 오류: {val!r}")

    # ── 식별 ──
    if str(it.get("schema_version")) != SCHEMA_VERSION:
        E.append(f"schema_version 은 '{SCHEMA_VERSION}'")
    iid, sch, typ, origin = str(it.get("id", "")), it.get("school"), it.get("type"), it.get("origin")
    if f.stem != iid:
        E.append(f"파일명({f.stem}) ≠ id({iid})")
    for fld in ("version", "created", "updated", "points", "level", "form", "stem"):
        if it.get(fld) in (None, "") and not (fld == "stem" and it.get("subparts")):
            E.append(f"필수 필드 누락: {fld}")
    enum(st, tax["statuses"], "status")
    enum(it.get("exam_family"), tax["exam_families"], "exam_family")
    enum(it.get("item_type"), tax["item_types"], "item_type")
    enum(origin, tax["origins"], "origin")
    enum(it.get("level"), tax["levels"], "level")
    enum(it.get("form"), tax["forms"], "form")
    enum(sch, tax["schools"], "school")
    if origin in ("original", "variant"):
        m = ID_ORIGINAL.match(iid)
        if not m:
            E.append(f"ID 형식 오류(학교-Txx-순번3): {iid}")
        elif (m["school"], m["type"]) != (sch, typ):
            E.append(f"ID 의 학교/유형({m['school']}/{m['type']}) ≠ 필드({sch}/{typ})")
        if typ not in (tax["types"].get(sch) or {}):
            E.append(f"유형 미등록: {sch}/{typ} (taxonomy.yaml types 에 먼저 추가)")
    elif origin in ("past_exam", "mock_exam"):
        m = ID_PAST.match(iid)
        if not m:
            E.append(f"기출 ID 형식 오류(학교-G26-04 / 학교-M27-02 / 학교-G26A-01): {iid}")
        elif m["school"] != sch or (m["src"] == "G") != (origin == "past_exam"):
            E.append("기출 ID 의 학교·G/M 구분이 school·origin 과 다름")
        if typ is not None and typ not in (tax["types"].get(sch) or {}):
            E.append(f"유형 미등록: {sch}/{typ}")
    if origin == "variant" and not it.get("variant_of"):
        E.append("origin=variant 이면 variant_of 필수")
    for ref_f in ("variant_of", "replaces"):
        if it.get(ref_f) and it[ref_f] not in repo_ids:
            E.append(f"{ref_f} {it[ref_f]} 가 저장소에 없음")

    # ── 교육과정 ──
    cur = it.get("curriculum") or {}
    enum(str(cur.get("version")), ("2015", "2022"), "curriculum.version")
    enum(cur.get("scope_2028", "unknown"), tax["scope_2028"], "curriculum.scope_2028")
    subj_allowed = tax["subjects_2015"] if str(cur.get("version")) == "2015" else tax["subjects_2022"]
    for s in cur.get("subjects") or []:
        if s not in subj_allowed:
            E.append(f"curriculum.subjects 값 오류: {s}")
    if not cur.get("subjects"):
        W.append("curriculum.subjects 비어 있음")
    if not cur.get("scope_confirmed"):
        (E if gated else W).append(f"scope_2028={cur.get('scope_2028')} 은 추정값(scope_confirmed: false)")
    if cur.get("behavior_domain") is None:
        gate("curriculum.behavior_domain 없음")
    else:
        enum(cur.get("behavior_domain"), tax["behavior_domains"], "behavior_domain")
    for s in cur.get("standards") or []:
        if not s.get("verified") and gated:
            W.append(f"성취기준 {s.get('code')} 원문 대조 전(verified: false)")

    # ── 출처·저작권 (내보내기 차단 규칙) ──
    src = it.get("source") or {}
    enum(src.get("license"), tax["licenses"], "source.license")
    enum(src.get("permission_status"), tax["permission_statuses"], "source.permission_status")
    enum(src.get("allowed_use"), tax["allowed_uses"], "source.allowed_use")
    enum(src.get("derivative_level"), tax["derivative_levels"], "source.derivative_level")
    if not src.get("rights_holder"):
        E.append("source.rights_holder 필수")
    if src.get("allowed_use") == "full_reproduction" and src.get("permission_status") not in ("granted", "not_needed"):
        E.append("허락 없이 allowed_use=full_reproduction 불가(permission_status 가 granted/not_needed 여야 함)")
    if src.get("permission_status") == "granted" and not src.get("permission_ref"):
        E.append("permission_status=granted 이면 permission_ref(허락서 파일) 필수")
    if origin in ("past_exam", "mock_exam"):
        for k in ("institution", "admission_year", "question_no", "official_url"):
            if not src.get(k):
                E.append(f"기출·모의는 source.{k} 필수")
        if src.get("license") == "own_original":
            E.append("기출·모의의 license 는 own_original 이 될 수 없음")

    # ── 소문항·채점표·해설 ──
    subs = it.get("subparts") or []
    if not subs:
        E.append("subparts 없음(단일 논제도 소문항 '1' 하나)")
    sub_ids = [str(s.get("id")) for s in subs]
    if len(set(sub_ids)) != len(sub_ids):
        E.append(f"소문항 id 중복 {sub_ids}")
    if sum(int(s.get("points") or 0) for s in subs) != (it.get("points") or 0):
        E.append(f"소문항 배점 합 {sum(int(s.get('points') or 0) for s in subs)} ≠ points {it.get('points')}")
    all_crit = {}
    for s in subs:
        for c in (s.get("rubric") or {}).get("criteria") or []:
            all_crit[str(c.get("id"))] = (str(s.get("id")), c)
    mis_ids_item = set()
    for s in subs:
        sid = str(s.get("id"))
        if len(subs) > 1 and not (s.get("stem") or "").strip():
            E.append(f"소문항 {sid} 발문 없음")
        for dep in s.get("depends_on") or []:
            if str(dep) not in sub_ids or sub_ids.index(str(dep)) >= sub_ids.index(sid):
                E.append(f"소문항 {sid} depends_on {dep} 오류(앞 소문항만)")
        ans = s.get("answer") or {}
        enum(ans.get("kind"), tax["answer_kinds"], f"소문항 {sid} answer.kind")
        if ans.get("kind") != "proof" and not ans.get("value"):
            gate(f"소문항 {sid} 정답(answer.value) 없음")
        # 채점표
        rub = s.get("rubric") or {}
        crit = rub.get("criteria") or []
        if not crit:
            E.append(f"소문항 {sid} 채점 마크 없음")
        if sum(int(c.get("points") or 0) for c in crit) != int(s.get("points") or 0):
            E.append(f"소문항 {sid} 마크 배점 합 {sum(int(c.get('points') or 0) for c in crit)} ≠ {s.get('points')}")
        step_ids_sub = {st.get("id") for so in s.get("solutions") or [] for st in so.get("steps") or []}
        for c in crit:
            cid = str(c.get("id"))
            if not cid.startswith(f"{sid}."):
                E.append(f"마크 id {cid} 는 '{sid}.' 로 시작해야 함")
            if not isinstance(c.get("points"), int) or c["points"] < 1:
                E.append(f"마크 {cid} 배점은 1 이상 정수")
            if not (c.get("evidence") or "").strip():
                E.append(f"마크 {cid} evidence(답안에 적혀 있어야 할 것) 없음")
            mt = c.get("mark_type")
            if mt is None:
                gate(f"마크 {cid} mark_type 미지정(M/A/B/R/C)")
            elif mt not in tax["mark_types"]:
                E.append(f"마크 {cid} mark_type 값 오류: {mt}")
            for d in c.get("depends_on") or []:
                if str(d) not in all_crit:
                    E.append(f"마크 {cid} depends_on {d} 없음")
            if mt == "A" and not c.get("implied_ok") and not any(
                    all_crit.get(str(d), (None, {}))[1].get("mark_type") in ("M", "R") for d in c.get("depends_on") or []):
                gate(f"A 마크 {cid} 는 M(또는 R) 마크에 depends_on 해야 함(a_requires_m)")
            ft = c.get("ft") or {}
            if ft.get("allowed"):
                for fr in ft.get("from") or []:
                    if str(fr) not in all_crit or all_crit[str(fr)][0] == sid:
                        E.append(f"마크 {cid} ft.from {fr} 오류(다른 소문항의 마크만)")
                if int(ft.get("max_points") or 0) > int(c.get("points") or 0):
                    E.append(f"마크 {cid} ft.max_points 가 배점보다 큼")
            for stp in c.get("steps") or []:
                if stp not in step_ids_sub:
                    E.append(f"마크 {cid} 의 steps {stp} 가 해설에 없음")
        for d in rub.get("deductions") or []:
            enum(d.get("effect"), tax["deduction_effects"], f"감점 {d.get('code')} effect")
        # 해설
        sols = s.get("solutions") or []
        auds = [so.get("audience") for so in sols]
        for a in auds:
            enum(a, tax["solution_audiences"], f"소문항 {sid} solutions.audience")
        for need in ("A", "B"):
            if need not in auds:
                gate(f"소문항 {sid} 해설 {need} 없음")
        seen_steps = set()
        for so in sols:
            for stp in so.get("steps") or []:
                if stp.get("id") in seen_steps:
                    E.append(f"소문항 {sid} 해설 단계 id 중복 {stp.get('id')}")
                seen_steps.add(stp.get("id"))
                for cr in stp.get("criteria") or []:
                    if str(cr) not in all_crit:
                        E.append(f"해설 단계 {stp.get('id')} 의 criteria {cr} 가 채점표에 없음")
        # 오개념
        mis = s.get("misconceptions") or []
        if not mis:
            gate(f"소문항 {sid} 흔한 오답(misconceptions) 없음")
        codes = []
        for mc in mis:
            if mc.get("id") not in cat["mis"]:
                E.append(f"오개념 {mc.get('id')} 가 db/misconceptions.yaml 에 없음")
            mis_ids_item.add(mc.get("id"))
            oc = str(mc.get("outcome_code", ""))
            if not re.fullmatch(r"7[0-8]", oc):
                E.append(f"오개념 {mc.get('id')} outcome_code 는 70~78")
            codes.append(oc)
            for k in ("wrong_work", "detect_cue", "feedback"):
                if not mc.get(k):
                    gate(f"오개념 {mc.get('id')} {k} 없음")
        if len(set(codes)) != len(codes):
            E.append(f"소문항 {sid} outcome_code 중복 {codes}")

    # ── 처방 링크 ──
    links = it.get("remedy_links") or []
    link_ids = {ln.get("id") for ln in links}
    for ln in links:
        tr, tg = ln.get("trigger") or {}, ln.get("target") or {}
        enum(tr.get("type"), tax["remedy_triggers"], f"처방 {ln.get('id')} trigger.type")
        enum(ln.get("relation"), tax["remedy_relations"], f"처방 {ln.get('id')} relation")
        if tr.get("type") == "criterion_missed" and str(tr.get("ref")) not in all_crit:
            E.append(f"처방 {ln.get('id')} trigger.ref {tr.get('ref')} 마크 없음")
        if tr.get("type") == "misconception" and tr.get("ref") not in mis_ids_item:
            E.append(f"처방 {ln.get('id')} trigger.ref {tr.get('ref')} 이 이 문항 오개념에 없음")
        if tg.get("kind") == "concept" and tg.get("ref") not in cat["con"]:
            E.append(f"처방 {ln.get('id')} target {tg.get('ref')} 가 db/concepts.yaml 에 없음")
        elif tg.get("kind") == "item" and tg.get("ref") not in repo_ids:
            E.append(f"처방 {ln.get('id')} target 문항 {tg.get('ref')} 없음")
        elif tg.get("kind") not in ("concept", "item"):
            E.append(f"처방 {ln.get('id')} target.kind 는 concept/item")
    for s in subs:
        for mc in s.get("misconceptions") or []:
            for r in mc.get("remedy") or []:
                if r not in link_ids:
                    E.append(f"오개념 {mc.get('id')} 의 remedy {r} 가 remedy_links 에 없음")
    if not links:
        gate("remedy_links(처방) 없음")
    if not (it.get("intent") or "").strip():
        gate("intent(출제의도) 없음")
    if not it.get("idea_tags"):
        gate("idea_tags(발상 태그) 없음")

    # ── 수식 표기 ──
    seen = set()
    for txt in all_strings(it):
        for e in check_latex(txt):
            if e not in seen:
                E.append(e)
                seen.add(e)

    # ── 검수 기록 ──
    revs = it.get("reviews") or []
    last, pos = {}, {}
    for i, r in enumerate(revs, 1):
        if not isinstance(r, dict):
            E.append(f"reviews[{i}] 형식 오류")
            continue
        enum(r.get("stage"), tax["review_stages"], f"reviews[{i}].stage")
        enum(r.get("verdict"), tax["verdicts"], f"reviews[{i}].verdict")
        enum(r.get("actor_type"), tax["actor_types"], f"reviews[{i}].actor_type")
        if not r.get("actor_id") or not r.get("date"):
            E.append(f"reviews[{i}] actor_id·date 필수")
        if r.get("actor_type") == "model" and not r.get("prompt_version"):
            E.append(f"reviews[{i}] 모델 단계는 prompt_version 필수")
        if r.get("verdict") == "fail":
            if not r.get("reason_codes"):
                E.append(f"reviews[{i}] fail 이면 reason_codes 필수")
            for rc in r.get("reason_codes") or []:
                enum(rc, tax["reason_codes"], f"reviews[{i}].reason_codes")
            if r.get("stage") in ("r1_scope", "r2_calc_1", "r2_calc_2"):
                enum(r.get("restart_from"), tax["restart_from"], f"reviews[{i}].restart_from")
        if r.get("stage") in ("r2_calc_1", "r2_calc_2") and r.get("verdict") == "pass" and not r.get("computed_answers"):
            E.append(f"reviews[{i}] r2 통과 기록에는 computed_answers(독립 계산 답) 필수")
        last[r.get("stage")], pos[r.get("stage")] = r, i
    if gated:
        ref = str(it.get("verify") or "")
        path_part, _, fn = ref.partition("::")
        vfile = KIT / path_part if path_part else None
        if not vfile or not vfile.exists():
            E.append(f"status={st} 인데 검산 스크립트 없음: {ref}")
        elif not fn or not re.search(rf"^def {re.escape(fn)}\(", vfile.read_text(encoding="utf-8"), re.M):
            E.append(f"검산 함수 {fn or '(미지정)'} 가 {path_part} 에 없음")
        for need in GATE_REVIEWS:
            if (last.get(need) or {}).get("verdict") != "pass":
                E.append(f"status={st} 인데 {need} 의 마지막 기록이 pass 가 아님")
        if "verify" in pos and any(pos.get(s, -1) > pos["verify"] for s in GATE_REVIEWS[:3]):
            E.append("verify 기록이 마지막 r1·r2 기록보다 앞섬 — 검토·수정 후 검산 재실행")
        a1 = (last.get("r2_calc_1") or {}).get("computed_answers")
        a2 = (last.get("r2_calc_2") or {}).get("computed_answers")
        if a1 and a2 and a1 != a2:
            E.append("r2_calc_1 과 r2_calc_2 의 computed_answers 불일치")
    if at_least(st, "sampled") and st != "published" and (last.get("human_sample") or {}).get("verdict") != "pass":
        E.append("status=sampled 인데 human_sample 통과 기록 없음")

    # ── 사람 판단 필요 ──
    excl = (tax["schools"].get(sch) or {}).get("exclude_tools") or []
    txt = item_texts(it)
    if "기하" in excl and GEOM_PAT.search(txt["solution"]):
        W.append("해설에 기하 도구(벡터·내적 등) 의심 — 이 학교 출제범위 밖일 수 있음")
    if "확률과통계" in excl and STAT_TOOL_PAT.search(txt["prompt"] + txt["solution"]):
        W.append("확률과통계 도구 의심 — 이 학교 출제범위 밖일 수 있음")
    if sch == "SJU" and it.get("stimuli"):
        W.append("세종대 문항인데 제시문이 있음")
    if sch == "DGU" and origin == "original" and "【가】" not in "".join(s.get("body", "") for s in it.get("stimuli") or []):
        W.append("동국대 문항인데 【가】 제시문 없음")
    return it


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", type=pathlib.Path)
    ap.add_argument("--book", type=pathlib.Path)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    tax = load_taxonomy()
    cat = {"mis": load_catalog("misconceptions"), "con": load_catalog("concepts")}
    repo_ids = all_item_ids()
    id_to_file = {p.stem: p for p in item_files()}
    errors: dict[str, list[str]] = collections.defaultdict(list)
    warns: dict[str, list[str]] = collections.defaultdict(list)

    if args.book:
        book = yaml.safe_load(args.book.read_text(encoding="utf-8")) or {}
        ids = [x["id"] for x in book.get("items", [])]
        for i in ids:
            if i not in id_to_file:
                errors[f"[교재 {args.book.name}]"].append(f"없는 문항 {i}")
        files = [id_to_file[i] for i in ids if i in id_to_file]
    else:
        roots = args.paths or [KIT / "items"]
        files = sorted({f for r in roots for f in ([r] if r.is_file() else r.rglob("*.yaml"))})

    metas: dict[str, dict] = {}
    seen_legacy: dict[str, str] = {}
    by_set = collections.defaultdict(list)
    for f in files:
        key = str(f.relative_to(KIT)) if f.is_relative_to(KIT) else str(f)
        E, W = [], []
        it = check_item(f, tax, cat, repo_ids, E, W)
        for w in W:
            (warns[key]).append(w)
        errors[key].extend(E)
        if it is None:
            continue
        iid = str(it.get("id"))
        if iid in metas:
            errors[key].append(f"ID 중복: {iid}")
        if it.get("legacy_id"):
            if it["legacy_id"] in seen_legacy:
                errors[key].append(f"legacy_id 중복: {it['legacy_id']}")
            seen_legacy[it["legacy_id"]] = iid
        if args.book:
            if not at_least(it.get("status"), "verified"):
                errors[key].append(f"교재 수록 문항인데 status={it.get('status')} (verified 이상 필요)")
            if (it.get("source") or {}).get("allowed_use") != "full_reproduction":
                errors[key].append("allowed_use≠full_reproduction — 원문을 교재에 실을 수 없음(출처+자체 해설로 바꿀 것)")
        metas[iid] = it
        seq = item_seq(iid)
        if it.get("replaces") and item_seq(it["replaces"]):
            seq = item_seq(it["replaces"])
        if it.get("origin") in ("original", "variant") and seq:
            by_set[(it["school"], it["type"], (seq - 1) // 10)].append(it)

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
            over = [x for x, c in fm.items() if c > 3]
            if over:
                warns[key].append(f"같은 형식 3회 초과: {over}")

    if not args.paths and not args.book:
        for bf, b in load_books():
            key = f"[교재 {bf.name}]"
            ids = [x.get("id") for x in b.get("items", [])]
            for i in ids:
                if i not in repo_ids:
                    errors[key].append(f"없는 문항 {i}")
            d = [i for i, c in collections.Counter(ids).items() if c > 1]
            if d:
                errors[key].append(f"같은 문항 중복 수록 {d}")
            if b.get("series") not in tax["series"]:
                errors[key].append(f"series 값 오류: {b.get('series')}")
        for ef in sorted((KIT / "errata").glob("*.yaml")) if (KIT / "errata").exists() else []:
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
                errors[key].append(f"severity 값 오류: {e['severity']}")
            if e.get("status", "open") not in ERRATA_STATUS:
                errors[key].append(f"status 값 오류: {e.get('status')}")
        for mid, m in cat["mis"].items():
            for c in m.get("concepts") or []:
                if c not in cat["con"]:
                    errors["[db/misconceptions.yaml]"].append(f"{mid} 의 개념 {c} 가 concepts.yaml 에 없음")
        for cid, c in cat["con"].items():
            for r in [c.get("parent")] + list(c.get("precedes") or []):
                if r and r not in cat["con"]:
                    errors["[db/concepts.yaml]"].append(f"{cid} 가 참조하는 {r} 없음")

    n_err = sum(len(v) for v in errors.values())
    n_warn = sum(len(v) for v in warns.values())
    if not args.quiet:
        for k in sorted(set(errors) | set(warns)):
            for e in errors.get(k, []):
                print(f"ERROR {k}: {e}")
            for w in warns.get(k, []):
                print(f"WARN  {k}: {w}")
    wc = collections.Counter(re.sub(r"\s\S*\d\S*|[:(=].*", "", w) for v in warns.values() for w in v)
    print(f"SUMMARY files={len(files)} errors={n_err} warnings={n_warn}")
    for w, c in wc.most_common(12):
        print(f"  WARN x{c}: {w}")
    return 1 if n_err or (args.strict and n_warn) else 0


if __name__ == "__main__":
    sys.exit(main())
