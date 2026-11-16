#!/usr/bin/env python3
"""유형별 md 30개 -> 단일 HTML(아티팩트용). 사용: python3 build.py <katex_dist_dir>"""
import base64, html, pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).parent
KATEX = pathlib.Path(sys.argv[1])
SCHOOLS = [("soongsil", "숭실대", "S"), ("dongguk", "동국대", "D"), ("sejong", "세종대", "J")]


def katex_css():
    css = (KATEX / "katex.min.css").read_text()

    def repl(m):
        name = m.group(1)
        if not name.endswith(".woff2"):
            return 'url(data:,)'
        data = base64.b64encode((KATEX / name).read_bytes()).decode()
        return f"url(data:font/woff2;base64,{data})"

    return re.sub(r"url\(([^)]+)\)", repl, css)


def prep(md):
    """모범답안~채점표를 접이식으로 감싼다."""
    out, open_ = [], False
    for line in md.splitlines():
        s = line.strip()
        if open_ and (s == "---" or s.startswith("### ") or s.startswith("## ")):
            out += ["", "</details>", ""]
            open_ = False
        if s.startswith("**모범답안**") and not open_:
            out += ["", '<details class="sol"><summary>모범답안 · 채점표 보기</summary>', ""]
            open_ = True
        out.append(line)
    if open_:
        out += ["", "</details>"]
    return "\n".join(out)


def to_html(md):
    r = subprocess.run(
        ["pandoc", "-f", "markdown+tex_math_dollars-smart", "-t", "html", "--katex", "--wrap=none"],
        input=prep(md), capture_output=True, text=True, check=True)
    h = r.stdout
    h = re.sub(r"<table", '<div class="tw"><table', h).replace("</table>", "</table></div>")
    h = re.sub(r"<p><code>(L\d)([^<]*)</code></p>",
               lambda m: f'<p class="meta"><span class="lv {m.group(1).lower()}">{m.group(1)}</span>{m.group(2)}</p>', h)
    return h


def main():
    nav, secs, count = [], [], 0
    for folder, name, pre in SCHOOLS:
        items = []
        for f in sorted((ROOT / folder).glob("T[0-9][0-9].md")):
            md = f.read_text()
            m = re.search(r"^#\s+(.+)$", md, re.M)
            title = m.group(1).strip() if m else f.stem
            n = len(re.findall(r"^###\s+" + pre + r"\d\d-\d\d", md, re.M))
            count += n
            sid = f"{pre.lower()}{f.stem[1:]}"
            items.append(f'<a href="#{sid}" data-id="{sid}">{html.escape(title)}<span>{n}</span></a>')
            secs.append(f'<section id="sec-{sid}" class="type" hidden>{to_html(md)}</section>')
        nav.append(f'<div class="school"><h2>{name}</h2><nav>{"".join(items)}</nav></div>')
    page = (ROOT / "template.html").read_text()
    page = (page.replace("/*KATEX_CSS*/", katex_css())
                .replace("<!--NAV-->", "\n".join(nav))
                .replace("<!--SECTIONS-->", "\n".join(secs))
                .replace("<!--COUNT-->", str(count)))
    (ROOT / "dist").mkdir(exist_ok=True)
    (ROOT / "dist" / "index.html").write_text(page)
    print("built", count, "problems,", len(page) // 1024, "KB")


main()
