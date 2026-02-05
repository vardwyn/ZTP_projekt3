import argparse
from pathlib import Path

from markdown_it import MarkdownIt
from weasyprint import HTML


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Renderuje raport Markdown do PDF."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def render_markdown_to_html(md_text: str) -> str:
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    body = md.render(md_text)
    style = """
    @page { size: A4; margin: 15mm; }
    body { font-family: Arial, sans-serif; font-size: 12pt; color: #111; }
    h1, h2, h3 { margin-top: 1.2em; }
    table { border-collapse: collapse; width: 100%; margin: 0.6em 0; }
    th, td { border: 1px solid #ccc; padding: 4px 6px; font-size: 10pt; }
    img { max-width: 100%; max-height: 250mm; height: auto; display: block; }
    img { page-break-inside: avoid; }
    ul { margin: 0.4em 0 0.8em 1.2em; }
    """
    return f"<html><head><style>{style}</style></head><body>{body}</body></html>"


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    md_path = args.input
    out_path = args.output

    md_text = md_path.read_text(encoding="utf-8")
    html = render_markdown_to_html(md_text)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html, base_url=str(md_path.parent)).write_pdf(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
