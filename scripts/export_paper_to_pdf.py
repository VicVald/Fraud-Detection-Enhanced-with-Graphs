#!/usr/bin/env python3
"""
export_paper_to_pdf.py
Converts LOCAL_COMPLETE_PAPER.md to a publication-grade PDF using Python Markdown,
MathJax 3 (vector SVG math), academic CSS typography, and Puppeteer headless Chrome.
"""

import os
import re
import subprocess
import sys
import markdown

WORKSPACE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
MD_PATH = os.path.join(WORKSPACE_DIR, "LOCAL_COMPLETE_PAPER.md")
HTML_PATH = os.path.join(WORKSPACE_DIR, "LOCAL_COMPLETE_PAPER.html")
PDF_PATH = os.path.join(WORKSPACE_DIR, "LOCAL_COMPLETE_PAPER.pdf")

def sanitize_content(text):
    """
    Sanitizes currency and math blocks so that:
    1. Code blocks (```...```) are preserved untouched so literal US$ is printed.
    2. Display math ($$...$$) is reliably converted to \\[ ... \\] block divs.
    3. Currency symbols in prose (US$, $, R$) do not interfere with math parsing.
    4. Inline math ($...$) is parsed cleanly by pymdownx.arithmatex into \\( ... \\).
    """
    # 0. Temporarily extract code blocks
    code_blocks = []
    def save_code(m):
        code_blocks.append(m.group(0))
        return f"___CODE_BLOCK_{len(code_blocks)-1}___"

    text = re.sub(r'```.*?```', save_code, text, flags=re.DOTALL)

    # 1. Protect currency signs in prose: replace US$ with US&#36;
    text = re.sub(r'US\$', 'US&#36;', text)
    text = re.sub(r'\+US\$', '+US&#36;', text)
    text = re.sub(r'-US\$', '-US&#36;', text)
    text = re.sub(r'\\\$(\d+)', r'&#36;\1', text)
    text = re.sub(r'\+\\\$(\d+)', r'+&#36;\1', text)
    text = re.sub(r'-\\\$(\d+)', r'-&#36;\1', text)

    # 2. Convert all display math $$...$$ to explicit \\[ ... \\] in separate divs
    def rep_disp(m):
        eq = m.group(1).strip()
        return f"\n\n<div class=\"arithmatex-display\">\\[{eq}\\]</div>\n\n"

    text = re.sub(r'\$\$(.*?)\$\$', rep_disp, text, flags=re.DOTALL)

    # 3. Restore code blocks
    def restore_code(m):
        idx = int(m.group(1))
        return code_blocks[idx]

    text = re.sub(r'___CODE_BLOCK_(\d+)___', restore_code, text)

    return text

def build_html():
    print(f"Reading {MD_PATH}...")
    with open(MD_PATH, "r", encoding="utf-8") as f:
        md_content = f.read()

    md_content = sanitize_content(md_content)

    # Pre-process: enhance image tags to semantic <figure> with <figcaption>
    def image_replacer(match):
        alt = match.group(1)
        src = match.group(2)
        abs_src = os.path.join(WORKSPACE_DIR, src) if not src.startswith("/") else src
        file_uri = f"file://{abs_src}"
        return f'<figure class="paper-figure"><img src="{file_uri}" alt="{alt}" /><figcaption>{alt}</figcaption></figure>'

    md_content = re.sub(r'!\[(.*?)\]\((.*?)\)', image_replacer, md_content)

    print("Converting Markdown to HTML with pymdownx.arithmatex and tables...")
    html_body = markdown.markdown(
        md_content,
        extensions=[
            "tables",
            "fenced_code",
            "attr_list",
            "pymdownx.arithmatex"
        ],
        extension_configs={
            "pymdownx.arithmatex": {
                "generic": True
            }
        }
    )

    html_template = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <title>Engenharia de Grafos Bipartidos e Compressão Espectral na Detecção de Fraudes em E-Commerce</title>
  
  <!-- MathJax 3 Configuration: only process \\( ... \\) and \\[ ... \\] -->
  <script>
  window.MathJax = {{
    tex: {{
      inlineMath: [['\\\\(', '\\\\)']],
      displayMath: [['\\\\[', '\\\\]']],
      processEscapes: false,
      processEnvironments: true
    }},
    svg: {{
      fontCache: 'global'
    }},
    startup: {{
      typeset: true
    }}
  }};
  </script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>

  <style>
    @page {{
      size: A4;
      margin: 20mm 16mm 20mm 16mm;
    }}

    *, *:before, *:after {{
      box-sizing: border-box;
    }}

    body {{
      font-family: "Liberation Serif", "Times New Roman", Times, "DejaVu Serif", serif;
      font-size: 10pt;
      line-height: 1.52;
      color: #1a202c;
      background-color: #ffffff;
      margin: 0;
      padding: 0;
      text-align: justify;
      hyphens: auto;
      text-justify: inter-word;
    }}

    /* Title & Meta */
    h1 {{
      font-size: 18pt;
      font-weight: 700;
      line-height: 1.25;
      text-align: center;
      margin-top: 0;
      margin-bottom: 14px;
      color: #0f172a;
      letter-spacing: -0.3px;
    }}

    p > strong:first-child {{
      color: #1e293b;
    }}

    /* Horizontal divider */
    hr {{
      border: 0;
      border-top: 1px solid #cbd5e1;
      margin: 18px 0;
    }}

    /* Section Headings */
    h2 {{
      font-size: 13pt;
      font-weight: 700;
      color: #0f172a;
      border-bottom: 1.5px solid #0f172a;
      padding-bottom: 3px;
      margin-top: 24px;
      margin-bottom: 10px;
      page-break-after: avoid;
      break-after: avoid;
    }}

    h3 {{
      font-size: 11pt;
      font-weight: 700;
      color: #1e293b;
      margin-top: 16px;
      margin-bottom: 6px;
      page-break-after: avoid;
      break-after: avoid;
    }}

    h4 {{
      font-size: 10pt;
      font-weight: 700;
      font-style: italic;
      color: #334155;
      margin-top: 12px;
      margin-bottom: 4px;
      page-break-after: avoid;
      break-after: avoid;
    }}

    /* Paragraphs and Lists */
    p {{
      margin-top: 0;
      margin-bottom: 8px;
    }}

    ul, ol {{
      margin-top: 4px;
      margin-bottom: 10px;
      padding-left: 22px;
    }}

    li {{
      margin-bottom: 4px;
    }}

    /* Blockquotes */
    blockquote {{
      margin: 12px 0;
      padding: 8px 14px;
      background-color: #f8fafc;
      border-left: 3.5px solid #3b82f6;
      border-radius: 0 4px 4px 0;
      font-size: 9.5pt;
      color: #334155;
    }}

    blockquote p:last-child {{
      margin-bottom: 0;
    }}

    /* Tables (Academic Booktabs Style) */
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 14px 0;
      font-size: 8.2pt;
      line-height: 1.35;
      page-break-inside: avoid;
      break-inside: avoid;
    }}

    thead th {{
      border-top: 2px solid #0f172a;
      border-bottom: 1.5px solid #0f172a;
      padding: 5px 6px;
      font-weight: 700;
      color: #0f172a;
      background-color: #f8fafc;
      text-align: center;
      vertical-align: middle;
      white-space: nowrap;
    }}

    tbody td {{
      padding: 4px 6px;
      border-bottom: 1px solid #e2e8f0;
      vertical-align: middle;
      text-align: center;
    }}

    tbody td:first-child, thead th:first-child {{
      text-align: left;
    }}

    tbody tr:last-child td {{
      border-bottom: 2px solid #0f172a;
    }}

    tbody tr:nth-child(even) {{
      background-color: #fafbfc;
    }}

    /* Preformatted ASCII Art Diagrams & Code */
    pre {{
      background-color: #f8fafc;
      border: 1px solid #cbd5e1;
      border-radius: 4px;
      padding: 8px 10px;
      margin: 12px 0;
      page-break-inside: avoid;
      break-inside: avoid;
      overflow-x: hidden;
    }}

    pre code {{
      font-family: "Liberation Mono", "DejaVu Sans Mono", monospace;
      font-size: 7.2pt;
      line-height: 1.18;
      color: #0f172a;
      background: transparent;
      padding: 0;
      border: 0;
      white-space: pre;
      display: block;
    }}

    /* Figures and Captions */
    figure.paper-figure {{
      margin: 14px 0;
      text-align: center;
      page-break-inside: avoid;
      break-inside: avoid;
    }}

    figure.paper-figure img {{
      max-width: 95%;
      height: auto;
      max-height: 460px;
      border: 1px solid #e2e8f0;
      border-radius: 4px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}

    figcaption {{
      font-size: 8.8pt;
      font-style: italic;
      color: #475569;
      margin-top: 5px;
      text-align: center;
    }}

    /* Math formatting */
    .arithmatex-display {{
      margin: 12px 0;
      text-align: center;
      page-break-inside: avoid;
      break-inside: avoid;
    }}

    mjx-container {{
      color: #0f172a !important;
    }}

    .arithmatex {{
      display: inline-block;
    }}

    /* Code inline */
    code {{
      font-family: "Liberation Mono", "DejaVu Sans Mono", monospace;
      font-size: 8.5pt;
      background-color: #f1f5f9;
      padding: 1px 4px;
      border-radius: 3px;
      color: #0f172a;
    }}

    /* Prevent orphan lines */
    p, li {{
      orphans: 3;
      widows: 3;
    }}
  </style>
</head>
<body>
  {html_body}
</body>
</html>
"""

    print(f"Writing rendered HTML to {HTML_PATH}...")
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_template)
    print("HTML build complete.")

def render_pdf():
    print("Invoking Puppeteer to compile HTML to PDF...")
    
    node_script = f"""
const puppeteer = require("puppeteer");
const path = require("path");

(async () => {{
  console.log("Launching headless browser...");
  const browser = await puppeteer.launch({{
    headless: "new",
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
      "--disable-dev-shm-usage",
      "--disable-gpu"
    ]
  }});

  const page = await browser.newPage();
  const fileUrl = "file://" + path.resolve("{HTML_PATH}");
  console.log("Navigating to:", fileUrl);
  
  await page.goto(fileUrl, {{ waitUntil: "networkidle0", timeout: 60000 }});

  console.log("Waiting for MathJax typesetting...");
  await page.waitForFunction(() => window.MathJax && window.MathJax.startup && window.MathJax.startup.document, {{ timeout: 30000 }});
  await page.evaluate(async () => {{
    await window.MathJax.typesetPromise();
  }});

  console.log("Ensuring all images are fully loaded...");
  await page.evaluate(async () => {{
    const selectors = Array.from(document.images);
    await Promise.all(selectors.map(img => {{
      if (img.complete) return null;
      return new Promise((resolve) => {{
        img.onload = resolve;
        img.onerror = resolve;
      }});
    }}));
  }});

  // Additional brief pause for SVG layout stabilization
  await new Promise(r => setTimeout(r, 1500));

  console.log("Printing to PDF: {PDF_PATH}...");
  await page.pdf({{
    path: "{PDF_PATH}",
    format: "A4",
    printBackground: true,
    displayHeaderFooter: true,
    headerTemplate: `
      <div style="font-size: 7.5pt; color: #64748b; width: 100%; text-align: right; padding-right: 16mm; font-family: 'Liberation Serif', 'Times New Roman', serif;">
        Valadares (2026) — Engenharia de Grafos Bipartidos e Compressão Espectral na Detecção de Fraudes
      </div>
    `,
    footerTemplate: `
      <div style="font-size: 8pt; color: #64748b; width: 100%; text-align: center; font-family: 'Liberation Serif', 'Times New Roman', serif;">
        <span class="pageNumber"></span> / <span class="totalPages"></span>
      </div>
    `,
    margin: {{
      top: "20mm",
      bottom: "20mm",
      left: "16mm",
      right: "16mm"
    }}
  }});

  await browser.close();
  console.log("PDF generation successful!");
}})();
"""

    puppeteer_dir = "/home/victor/.npm/_npx/55158e48eb5c59f7/node_modules"
    env = os.environ.copy()
    env["NODE_PATH"] = puppeteer_dir

    result = subprocess.run(
        ["node", "-e", node_script],
        env=env,
        capture_output=True,
        text=True
    )

    print(result.stdout)
    if result.returncode != 0:
        print("ERROR running Puppeteer:", result.stderr, file=sys.stderr)
        sys.exit(result.returncode)

    file_size_mb = os.path.getsize(PDF_PATH) / (1024 * 1024)
    print(f"Output PDF: {PDF_PATH} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    build_html()
    render_pdf()
