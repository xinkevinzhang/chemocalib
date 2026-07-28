#!/usr/bin/env python3
"""Compile LaTeX files to PDF using online compilation services."""

import requests
import sys
import os
import time
from pathlib import Path
from io import BytesIO


def compile_via_latexonline(tex_content: str, timeout: int = 120) -> bytes:
    """Use latexonline.cc API to compile LaTeX to PDF."""
    url = "https://latexonline.cc/compile"
    files = {
        "file": ("main.tex", tex_content.encode("utf-8"), "application/x-tex")
    }
    params = {"target": "main.tex"}

    resp = requests.post(url, files=files, params=params, timeout=timeout)
    if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith("application/pdf"):
        return resp.content

    # Try text response for error message
    error_text = resp.text[:500] if resp.text else "No response body"
    raise RuntimeError(
        f"latexonline.cc returned {resp.status_code}: {error_text}"
    )


def compile_via_latex_to(tex_content: str, timeout: int = 120) -> bytes:
    """Use latex.to API as fallback."""
    url = "https://latex.to/api"
    data = {
        "latex": tex_content,
        "format": "pdf",
    }
    resp = requests.post(url, json=data, timeout=timeout)
    if resp.status_code == 200:
        result = resp.json()
        pdf_url = result.get("url") or result.get("pdf_url")
        if pdf_url:
            pdf_resp = requests.get(pdf_url, timeout=timeout)
            if pdf_resp.status_code == 200:
                return pdf_resp.content

    raise RuntimeError(f"latex.to returned {resp.status_code}: {resp.text[:500]}")


def compile_tex_to_pdf(tex_path: str, output_dir: str = None, max_retries: int = 2):
    """Compile a .tex file to PDF, trying multiple services."""
    tex_path = Path(tex_path)
    if not tex_path.exists():
        print(f"ERROR: {tex_path} not found.")
        return None

    tex_content = tex_path.read_text(encoding="utf-8")

    if output_dir is None:
        output_dir = tex_path.parent
    else:
        output_dir = Path(output_dir)

    output_path = output_dir / f"{tex_path.stem}.pdf"

    services = [
        ("latexonline.cc", compile_via_latexonline),
        ("latex.to", compile_via_latex_to),
    ]

    for attempt in range(max_retries + 1):
        for svc_name, svc_fn in services:
            try:
                print(f"  Trying {svc_name} (attempt {attempt + 1})...")
                pdf_bytes = svc_fn(tex_content, timeout=180)
                output_path.write_bytes(pdf_bytes)
                print(f"  SUCCESS via {svc_name}: {output_path} ({len(pdf_bytes)} bytes)")
                return str(output_path)
            except Exception as e:
                print(f"  {svc_name} failed: {e}")
                time.sleep(1)

    print(f"ERROR: All compilation services failed after {max_retries + 1} attempts.")
    return None


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent

    if len(sys.argv) > 1:
        tex_files = [Path(p) for p in sys.argv[1:]]
    else:
        # Default: compile manuscript
        tex_files = [project_root / "manuscript" / "main.tex"]
        beam = project_root / "manuscript" / "beamer.tex"
        if beam.exists():
            tex_files.append(beam)

    for tex_path in tex_files:
        print(f"\nCompiling: {tex_path.name}")
        result = compile_tex_to_pdf(str(tex_path))
        if result:
            print(f"  Output: {result}")
        else:
            print(f"  FAILED to compile {tex_path.name}")
