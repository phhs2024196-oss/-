#!/usr/bin/env python3
"""기출·모의 문항 YAML 뼈대 생성기 — 고정 필드는 스크립트가 쓰고, 모델은 TODO 만 채운다(출력 토큰 절약).

사용:
  python3 tools/scaffold.py --conf db/schools/AJOU.yaml --id AJOU-G24A-01 --year 2024 \
      --track '자연계열(오전)' --qno 1 --sub 1-1:8,1-2:12,1-3:10 \
      --url 'https://…' --from '아주대 입학처 공식 PDF「…」p.3' [--mock] [--force]
  → items/<학교>/PAST/<ID>.yaml (이미 있으면 --force 없이는 건드리지 않음)

conf(학교 공통값, 한 번만 작성): school, institution, admission_type(실전), admission_type_mock,
  retrieved_at, actor_id(기본 sonnet), prompt_version(기본 past-v2-lean), url_by_year({2024: …, M2024: …})
draft 단계 원칙(BRIEF_common.md): A 해설·hint_ladder·오개념·처방은 비워 둔다(validate 는 WARN 만).
"""
from __future__ import annotations

import argparse
import datetime
import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from lib import ID_PAST, KIT, dump_yaml  # noqa: E402

TODO = "TODO"


def subpart(sid: str, pts: int, page: str) -> dict:
    return {
        "id": sid,
        "stem": TODO + "\n",
        "points": pts,
        "depends_on": [],
        "answer": {"kind": "value", "value": TODO, "equivalents": [], "tolerance": None},
        "solutions": [{
            "id": f"S{sid.replace('-', '')}-B", "audience": "B", "method_no": 1,
            "steps": [{"id": f"{sid.replace('-', '')}b-1", "content": TODO + "\n",
                       "source_flow": page, "criteria": [f"{sid}.M1"]}],
        }],
        "rubric": {
            "rules": {"ft_scope": "cross_subpart_only", "ft_requires_working": True,
                      "a_requires_m": True, "misread_penalty": 2, "isw": True},
            "criteria": [{"id": f"{sid}.M1", "mark_type": "M", "points": pts,
                          "evidence": TODO, "steps": [f"{sid.replace('-', '')}b-1"]}],
        },
        "outcome_codes": [{"code": "20", "meaning": "만점"}],
        "misconceptions": [],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--year", type=int, required=True, help="학년도")
    ap.add_argument("--track", required=True)
    ap.add_argument("--qno", type=int, required=True)
    ap.add_argument("--sub", required=True, help="소문항id:배점,… 예 1-1:8,1-2:12")
    ap.add_argument("--from", dest="src", required=True, help="transcribed_from 문구")
    ap.add_argument("--url", default=None)
    ap.add_argument("--page", default="모범답안 p.?", help="source_flow 기본값")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    c = yaml.safe_load(pathlib.Path(a.conf).read_text(encoding="utf-8"))
    m = ID_PAST.match(a.id)
    if not m or m["school"] != c["school"]:
        sys.exit(f"ID 형식/학교 불일치: {a.id}")
    subs = [(s.split(":")[0], int(s.split(":")[1])) for s in a.sub.split(",")]
    url = a.url or (c.get("url_by_year") or {}).get(("M" if a.mock else "") + str(a.year)) \
        or (c.get("url_by_year") or {}).get(a.year)
    today = datetime.date.today()
    item = {
        "schema_version": "2.0", "id": a.id, "legacy_id": None, "version": 1, "status": "draft",
        "exam_family": "math_nonsul", "item_type": TODO,
        "origin": "mock_exam" if a.mock else "past_exam", "school": c["school"], "type": None,
        "variant_of": None, "replaces": None, "level": TODO, "form": TODO,
        "points": sum(p for _, p in subs), "time_estimate_min": None,
        "key_idea": TODO, "stuck": TODO, "tags": [],
        "elements": {"materials": [], "concepts": [], "tools": [], "conditions": [],
                     "stimuli_used": {}, "subpart_links": []},
        "curriculum": {"version": "2015", "subjects": [], "scope_2028": "unknown",
                       "scope_confirmed": False},
        "difficulty": {"rater": "author", "p_value": None, "n": 0},
        "source": {
            "rights_holder": c["institution"], "license": "all_rights_reserved",
            "permission_status": "not_requested", "allowed_use": "internal_only",
            "derivative_level": "verbatim", "institution": c["institution"],
            "admission_year": a.year,
            "admission_type": c.get("admission_type_mock" if a.mock else "admission_type"),
            "track": a.track, "question_no": a.qno, "official_url": url,
            "retrieved_at": str(c.get("retrieved_at", today)),
            "original_points": [p for _, p in subs], "rights_notice": "미확인",
            "transcribed_from": a.src,
        },
        "verify": f"verify/{a.id.rsplit('-', 1)[0]}.py::p{int(m['no']):02d}",
        "stimuli": [],
        "stem": TODO + "\n",
        "subparts": [subpart(s, p, a.page) for s, p in subs],
        "remedy_links": [],
        "reviews": [{"stage": "compose", "actor_type": "model", "actor_id": c.get("actor_id", "sonnet"),
                     "prompt_version": c.get("prompt_version", "past-v2-lean"), "item_version": 1,
                     "verdict": "pass", "date": today}],
        "created": today, "updated": today,
    }
    out = KIT / "items" / c["school"] / "PAST" / f"{a.id}.yaml"
    if out.exists() and not a.force:
        sys.exit(f"이미 있음(덮어쓰려면 --force): {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_yaml(item), encoding="utf-8")
    print(out.relative_to(KIT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
