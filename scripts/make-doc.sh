#!/usr/bin/env bash
# Build the Sphinx API documentation of the `scheduler` module into docs/build
# (HTML, and the PDF that readme.md links to).
#
# Needs: the built extension in the repository root, and
#     pip install sphinx furo sphinx-autodoc-typehints
# plus a LaTeX installation with latexmk for the PDF.
# Calls sphinx directly, so no docs/Makefile is required.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

rm -rf docs/build/html docs/build/doctrees
python3 -m sphinx -b html -d docs/build/doctrees docs/source docs/build/html
python3 -m sphinx -M latexpdf docs/source docs/build
