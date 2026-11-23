---
name: originality-checker
description: 신규 문항이 기출·시판 교재와 지나치게 비슷한지 확인할 때.
tools: Read, Grep, Glob, Bash
model: sonnet
---
dup_check로 기출 DB 44곳과 교재 DB 772문항을 대조한다. 기준 초과 문항만 유사 원본 ID와 함께 보고.
