"""Validate the actual explanation TeX and portable offline font bundle."""

import hashlib
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from cbf_demo.math_explanation import math_explanation_html
from cbf_demo.math_rendering import KATEX_VERSION, offline_math_assets


ASSETS = Path(__file__).resolve().parents[1] / "assets" / "katex"


class EquationParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.equations = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "math-equation" in attrs.get("class", "").split() and "data-tex" in attrs:
            self.equations.append(unescape(attrs["data-tex"]))


def test_all_actual_explanation_equations_compile_in_official_katex():
    executable = shutil.which("node")
    if executable is None:
        pytest.skip("Node is required for offline KaTeX parsing validation")
    parser = EquationParser()
    parser.feed(math_explanation_html())
    assert len(parser.equations) >= 13
    program = """
      const fs=require('fs'), katex=require(process.argv[1]);
      const equations=JSON.parse(fs.readFileSync(0,'utf8'));
      const outputs=equations.map(tex=>katex.renderToString(tex,{throwOnError:true,
        output:'htmlAndMathml',trust:false,displayMode:true}));
      console.log(JSON.stringify({count:outputs.length,
        allMathml:outputs.every(s=>s.includes('<math')&&s.includes('katex-html'))}));
    """
    result = subprocess.run([executable, "-e", program, str(ASSETS / "katex.min.js")],
                            input=json.dumps(parser.equations), text=True,
                            capture_output=True, check=True)
    assert json.loads(result.stdout) == dict(count=len(parser.equations), allMathml=True)


def test_offline_css_embeds_every_font_and_retains_license():
    bundle = offline_math_assets()
    css = re.search(r"<style[^>]*>(.*?)</style>", bundle, re.S).group(1)
    sources = re.findall(r"url\(([^)]+)\)", css)
    assert len(sources) == len(list((ASSETS / "fonts").glob("*.woff2"))) == 20
    assert all(source.strip("\"'").startswith("data:font/woff2;base64,") for source in sources)
    assert not any(reference in css for reference in ("http:", "https:", "fonts/", ".ttf", ".woff)"))
    assert "Copyright (c) 2013-2020 Khan Academy" in bundle
    assert "Permission is hereby granted" in bundle
    assert '<script data-katex-version="0.16.22">' in bundle


def test_vendored_assets_match_recorded_integrity_and_allowed_file_set():
    metadata = json.loads((ASSETS / "metadata.json").read_text())
    assert metadata["version"] == KATEX_VERSION == "0.16.22"
    assert metadata["integrity_verified"]
    retained = {file["path"] for file in metadata["retained_files"]}
    assert retained == {str(path.relative_to(ASSETS)) for path in ASSETS.rglob("*")
                        if path.is_file() and path.name != "metadata.json"}
    for file in metadata["retained_files"]:
        raw = (ASSETS / file["path"]).read_bytes()
        assert len(raw) == file["size_bytes"]
        assert hashlib.sha256(raw).hexdigest() == file["sha256"]
