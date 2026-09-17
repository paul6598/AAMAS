#!/usr/bin/env bash
# 한국어 초안과 참고문헌을 컴파일하고 생성물은 build/에 모은다.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# TeX 실행 파일이 PATH에 있는 환경에서 실행한다.
mkdir -p build
xelatex -synctex=1 -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build main.tex
bibtex build/main
xelatex -synctex=1 -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build main.tex
xelatex -synctex=1 -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build main.tex
