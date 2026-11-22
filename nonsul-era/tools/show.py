#!/usr/bin/env python3
"""문항 YAML 에서 필요한 필드만 출력(파일 통째 읽기 대신 — 토큰 절약).

사용: python3 tools/show.py <ID 또는 경로> [필드 …]
  필드 예: stem  stimuli  sub.stem  sub.answer  sub.points  source.transcribed_from  elements
  필드를 안 주면: id·points·소문항별 (id, points, answer.value) 한 줄씩.
"""
from __future__ import annotations

import pathlib
import sys

import yaml

KIT = pathlib.Path(__file__).resolve().parent.parent


def find(key: str) -> pathlib.Path:
    p = pathlib.Path(key)
    if p.exists():
        return p
    hits = list((KIT / "items").rglob(f"{key}.yaml"))
    if not hits:
        sys.exit(f"없음: {key}")
    return hits[0]


def get(o, path):
    for k in path.split("."):
        o = o.get(k) if isinstance(o, dict) else None
    return o


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    it = yaml.safe_load(find(sys.argv[1]).read_text(encoding="utf-8"))
    fields = sys.argv[2:]
    if not fields:
        print(f"{it['id']} points={it.get('points')} status={it.get('status')}")
        for s in it.get("subparts") or []:
            print(f"  [{s['id']}] {s.get('points')}점 답: {(s.get('answer') or {}).get('value')}")
        return 0
    for f in fields:
        if f.startswith("sub."):
            for s in it.get("subparts") or []:
                v = get(s, f[4:])
                print(f"[{s['id']}] {f[4:]}: {v if not isinstance(v, (dict, list)) else yaml.dump(v, allow_unicode=True, width=200).strip()}")
        else:
            v = get(it, f)
            print(f"{f}: {v if not isinstance(v, (dict, list)) else yaml.dump(v, allow_unicode=True, width=200).strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
