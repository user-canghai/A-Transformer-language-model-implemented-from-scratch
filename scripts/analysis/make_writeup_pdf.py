"""把 writeup/writeup.md 渲染成 writeup/writeup.pdf（Markdown -> HTML -> PDF）。"""

from __future__ import annotations

import os

import markdown
from weasyprint import HTML

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WRITEUP_DIR = os.path.join(ROOT, "writeup")
MD_PATH = os.path.join(WRITEUP_DIR, "writeup.md")
HTML_PATH = os.path.join(WRITEUP_DIR, "writeup.html")
PDF_PATH = os.path.join(WRITEUP_DIR, "writeup.pdf")

# 报告排版：A4 页面、中文字体、表格与代码块样式
CSS = """
@page { size: A4; margin: 2cm 2cm 2cm 2cm; }
body { font-family: "Noto Sans CJK SC", "DejaVu Sans", sans-serif; font-size: 10.5pt;
       line-height: 1.55; color: #111; }
h1 { font-size: 19pt; margin-top: 0; }
h2 { font-size: 14pt; margin-top: 1.2em; border-bottom: 1px solid #ddd; padding-bottom: 2px; }
h3 { font-size: 12pt; margin-top: 1em; }
code, pre { font-family: "DejaVu Sans Mono", monospace; font-size: 8.6pt; }
pre { background: #f6f8fa; padding: 6px 8px; border-radius: 3px; white-space: pre-wrap; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9pt; }
th, td { border: 1px solid #ccc; padding: 3px 6px; text-align: left; }
th { background: #f2f2f2; }
img { max-width: 100%; }
"""


def main() -> None:
    with open(MD_PATH, encoding="utf-8") as f:
        text = f.read()
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])
    html = "<html><head><meta charset='utf-8'><style>" + CSS + "</style></head><body>" + body + "</body></html>"
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    HTML(string=html, base_url=WRITEUP_DIR).write_pdf(PDF_PATH)
    print("已写入", PDF_PATH)


if __name__ == "__main__":
    main()
