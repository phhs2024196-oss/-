#!/usr/bin/env python3
"""학교별 A4 인쇄용 PDF 조판.
사용: python3 build_print.py <work_dir> <katex_dist_dir> <fontsource_serif_pkg> <fontsource_sans_pkg> [school ...]
구성: 표지 -> 빠른 정답 -> 유형별 풀이 팁 -> (문제=홀수 쪽 / 해설=짝수 쪽) x 100문항
"""
import base64, html, pathlib, re, subprocess, sys, os

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORK = pathlib.Path(sys.argv[1]); KATEX = pathlib.Path(sys.argv[2])
SERIF = pathlib.Path(sys.argv[3]); SANS = pathlib.Path(sys.argv[4])
ONLY = sys.argv[5:] or ["soongsil", "dongguk", "sejong"]
HERE = pathlib.Path(__file__).resolve().parent

SCHOOLS = {
    "soongsil": dict(name="숭실대", pre="S", pts="25점 (제시문 + 논제 1개)",
                     form="한 문항은 교과서 정리를 제시문으로 주고 논제 하나를 묻는 단답 서술형이다. 문항당 25점이다.",
                     scope="수학 · 수Ⅰ · 수Ⅱ · 미적분"),
    "dongguk": dict(name="동국대", pre="D", pts="30점 / 40점 (【가】~【다】 + 소문항)",
                    form="【가】~【다】 제시문을 인용해 소문항에 답하는 형식이다. 30점형은 소문항 8·10·12점, 40점형(유형 9·10)은 8·10·10·12점이고 줄 수 제한(15줄·27줄)이 있다.",
                    scope="수Ⅰ · 수Ⅱ · 미적분 · 확률과 통계 (기하 제외)"),
    "sejong": dict(name="세종대", pre="J", pts="260점 / 270점 (소문항 3개 직렬)",
                   form="제시문 없이 조건을 서술하고 (1)→(2)→(3)으로 이어 묻는다. 문항마다 ‘보이시오’ 증명 소문항이 있고 (3)은 (1)(2)의 결과를 쓴다.",
                   scope="수Ⅰ · 수Ⅱ · 미적분"),
}
PANDOC = ["pandoc", "-f", "markdown-raw_html-fancy_lists-startnum-smart+tex_math_dollars+fenced_divs+bracketed_spans",
          "-t", "html", "--katex", "--wrap=none"]


def md2html(md):
    return subprocess.run(PANDOC, input=md, capture_output=True, text=True, check=True).stdout


def fix_lists(t):
    # 문단 바로 뒤에 붙은 번호 목록 앞에 빈 줄 삽입
    out, prev = [], ""
    for line in t.splitlines():
        if re.match(r"^\d+\.\s", line) and prev.strip() and not re.match(r"^\d+\.\s", prev):
            out.append("")
        out.append(line); prev = line
    return "\n".join(out)


def parse_type(path, pre):
    md = path.read_text()
    m = re.search(r"^#\s+" + pre + r"(\d\d)\.\s*(.+)$", md, re.M)
    tno, title = int(m.group(1)), m.group(2).strip()
    tips = re.search(r"## 유형 해설\s*(.*?)\n## 설계도", md, re.S).group(1).strip()
    probs = []
    for blk in re.split(r"^### ", md, flags=re.M)[1:]:
        pid = blk.split("\n", 1)[0].strip()
        if not re.match(pre + r"\d\d-\d\d$", pid):
            continue
        rest = blk.split("\n", 1)[1]
        rest = re.sub(r"\n---\s*$", "", rest.rstrip())
        meta = re.match(r"\s*`([^`]+)`", rest).group(1)
        rest = rest[rest.index("`", rest.index("`") + 1) + 1:]
        stmt, sol = rest.split("**모범답안**", 1)
        i_ans, i_gr, i_tail = sol.index("**정답**"), sol.index("**채점표**"), sol.index("**대표 오답**")
        assert i_ans < i_gr < i_tail, pid
        passage, prob = "", stmt
        if "**논제**" in stmt:
            passage, prob = stmt.split("**논제**", 1)
        passage = re.sub(r"^\s*\*\*제시문\*\*", "", passage).strip()
        probs.append(dict(
            id=pid, meta=meta, passage=passage, prob=prob.strip(),
            steps=sol[:i_ans].strip(), ans=sol[i_ans + len("**정답**"):i_gr].strip(),
            grade=sol[i_gr + len("**채점표**"):i_tail].strip(), tail=sol[i_tail:].strip()))
    return dict(no=tno, title=title, tips=fix_lists(tips), probs=probs)


def level_str(probs):
    from collections import Counter
    c = Counter(p["meta"].split("·")[0].strip() for p in probs)
    return " · ".join(f"{k}×{c[k]}" for k in sorted(c))


def build_md(key, pads=frozenset()):
    sc = SCHOOLS[key]; pre = sc["pre"]
    types = [parse_type(f, pre) for f in sorted((ROOT / {"S": "soongsil", "D": "dongguk", "J": "sejong"}[pre]).glob("T[0-9][0-9].md"))]
    n = 0
    for t in types:
        t["first"] = n + 1
        for p in t["probs"]:
            n += 1; p["n"] = n
        t["last"] = n
    total = n
    md = []
    # ---- 표지
    md.append("::: {.cover}\n")
    md.append(f"# {sc['name']} 수리논술 유형별 N제\n")
    md.append(f"2027학년도 대비 · 자연계열 · {len(types)}유형 {total}문항 · 문항마다 풀이 팁과 채점표 수록\n")
    md.append("| 유형 | 이름 | 문항 | 난이도 |\n|:-:|:--|:-:|:--|")
    for t in types:
        md.append(f"| {t['no']} | {t['title']} | {t['first']:02d}~{t['last']:02d} | {level_str(t['probs'])} |")
    md.append("\n:::\n")
    md.append("::: {.notes}\n")
    md.append(f"**구성** 한 문항이 한 쪽이다. 홀수 쪽에 문제와 풀이 공간, 바로 다음 짝수 쪽에 그 문항의 정답·모범답안·채점표가 있다. 해설이 한 쪽을 넘으면 다음 문제가 홀수 쪽에서 시작하도록 빈 쪽이 들어간다. 빠른 정답은 2쪽부터 있다.\n")
    md.append(f"**형식** {sc['form']} 범위는 {sc['scope']}이다.\n")
    md.append("**난이도** L4는 세 학교 실전 중 가장 어려운 문항, L5는 경우 분류·조건 해석·2단계 이상의 구성이 필요한 문항, L6은 보조함수·불변량·귀류 같은 방향을 스스로 세워야 하는 문항이다. 유형마다 L4 2문항, L5 6문항, L6 2문항을 쉬운 것부터 배열했다. 이 등급은 출제 의도에 따른 자체 판정이다.\n")
    md.append("**검산** 모든 문항의 정답은 sympy로 문제 조건에서 다시 계산해 일치를 확인했고, 경우의 수·확률은 전수 열거와 정확한 분수로 확인했다. 증명 문항은 명제가 성립하는지를 수치로 확인했을 뿐 서술의 논리는 기계로 검증하지 않았다. 직접 읽어 확인하길 권한다.\n")
    md.append("**답안 작성** 단계 배점은 채점표를 따른다. 식은 맞고 계산만 틀리면 그 단계의 절반, 결론을 가정하고 시작하는 증명은 해당 단계 0점이다. 정의역·등호 조건 누락은 감점된다.\n")
    md.append(":::\n")
    # ---- 빠른 정답
    md.append("::: {.qah}\n빠른 정답\n:::\n")
    md.append("::: {.qa}\n")
    for t in types:
        for p in t["probs"]:
            md.append(f"::: {{.qi}}\n[{p['n']:02d}]{{.qn}} {p['ans']}\n:::\n")
    md.append(":::\n")
    # ---- 풀이 팁
    md.append("::: {.tipsh}\n유형별 풀이 팁\n:::\n")
    for t in types:
        md.append(f"::: {{.tip}}\n::: {{.tiph}}\n유형 {t['no']} · {t['title']}\n:::\n\n{t['tips']}\n:::\n")
    # ---- 문항
    for t in types:
        for p in t["probs"]:
            lv, form, pt = [x.strip() for x in p["meta"].split("·")]
            if "p:" + p["id"] in pads:
                md.append("::: {.blank}\n&nbsp;\n:::\n")
            md.append("::: {.prob}\n")
            md.append(f"::: {{.phead}}\n[유형 {t['no']} · {t['title']}]{{.pt}} [난이도 {lv} · {pt}]{{.pl}}\n:::\n")
            md.append(f"::: {{.qrow}}\n[{p['n']:02d}]{{.num}}\n\n::: {{.qbody}}\n")
            if p["passage"]:
                md.append(f"::: {{.passage}}\n{p['passage']}\n:::\n")
            md.append(p["prob"] + "\n\n:::\n:::\n:::\n")
            if "s:" + p["id"] in pads:
                md.append("::: {.blank}\n&nbsp;\n:::\n")
            md.append("::: {.sol}\n")
            md.append(f"::: {{.sbar}}\n[{p['n']:02d} 해설]{{.sn}} [{p['id']} · {lv} · {form}]{{.sm}}\n:::\n")
            md.append(f"::: {{.ansbox}}\n**정답** {p['ans']}\n:::\n")
            md.append(f"::: {{.steps}}\n{p['steps']}\n:::\n")
            md.append(f"::: {{.grade}}\n**채점표**\n\n{p['grade']}\n:::\n")
            md.append(f"::: {{.tail}}\n{p['tail']}\n:::\n")
            md.append(":::\n")
    return "\n".join(md), total


def katex_css():
    css = (KATEX / "katex.min.css").read_text()
    def repl(m):
        name = m.group(1)
        if not name.endswith(".woff2"):
            return "url(data:,)"
        return "url(data:font/woff2;base64," + base64.b64encode((KATEX / name).read_bytes()).decode() + ")"
    return re.sub(r"url\(([^)]+)\)", repl, css)


def link_fonts():
    fd = WORK / "fonts"; fd.mkdir(parents=True, exist_ok=True)
    for tag, pkg, ws in (("serif", SERIF, (400, 700)), ("sans", SANS, (400, 700))):
        d = fd / tag; d.mkdir(exist_ok=True)
        if not (d / "files").exists():
            os.symlink(pkg / "files", d / "files")
        for w in ws:
            (d / f"{w}.css").write_text((pkg / f"{w}.css").read_text())


def render(key, pads, out_pdf):
    md, total = build_md(key, pads)
    raw = WORK / f"{key}_raw.html"
    raw.write_text(md2html(md))
    subprocess.run(["node", str(HERE / "prerender_print.js"), str(KATEX), str(raw)], check=True, stdout=subprocess.DEVNULL)
    name = SCHOOLS[key]["name"]
    page = (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>{name} 수리논술 유형별 N제</title>'
            '<link rel="stylesheet" href="fonts/serif/400.css"><link rel="stylesheet" href="fonts/serif/700.css">'
            '<link rel="stylesheet" href="fonts/sans/400.css"><link rel="stylesheet" href="fonts/sans/700.css">'
            f'<style>{KCSS}</style><style>{CSS}</style></head><body>{raw.read_text()}</body></html>')
    (WORK / f"{key}.html").write_text(page)
    subprocess.run(["node", str(HERE / "print_pdf.js"), str(WORK / f"{key}.html"), str(out_pdf), f"{name} 수리논술 유형별 N제"], check=True)
    return total


def page_map(pdf):
    txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout.split("\f")
    if not txt[-1].strip():
        txt = txt[:-1]
    pp, sp = [], []
    for i, t in enumerate(txt, 1):
        b = re.sub(r"\s+", " ", t)
        m = re.search(r"\d{2,3} 해설 ([SDJ]\d\d-\d\d)", b)
        if m:
            sp.append((i, m.group(1)))
        elif re.search(r"난이도 L\d · \d+점", b):
            pp.append(i)
    return len(txt), pp, sp


def plan_pads(pdf, ids):
    n, pp, sp = page_map(pdf)
    assert len(pp) == len(sp) == len(ids), (len(pp), len(sp), len(ids))
    pads, pos = set(), pp[0]
    for k, pid in enumerate(ids):
        plen = sp[k][0] - pp[k]
        slen = (pp[k + 1] if k + 1 < len(ids) else n + 1) - sp[k][0]
        if pos % 2 == 0:
            pads.add("p:" + pid); pos += 1
        pos += plen
        if pos % 2 == 1:
            pads.add("s:" + pid); pos += 1
        pos += slen
    return pads


def main():
    global KCSS, CSS
    WORK.mkdir(parents=True, exist_ok=True)
    link_fonts()
    CSS = (HERE / "print.css").read_text()
    KCSS = katex_css()
    OUT = ROOT / "pdf"; OUT.mkdir(exist_ok=True)
    for key in ONLY:
        tmp = WORK / f"{key}_pass1.pdf"
        _, total = build_md(key)
        render(key, frozenset(), tmp)
        ids = [p["id"] for p in sum((parse_type(f, SCHOOLS[key]["pre"])["probs"] for f in sorted((ROOT / key).glob("T[0-9][0-9].md"))), [])]
        pads = plan_pads(tmp, ids)
        out = OUT / f"{SCHOOLS[key]['name']}_수리논술_유형별N제.pdf"
        render(key, pads, out)
        print(key, total, "problems; blank pages added:", len(pads), "->", out.name)


main()
