DEPS := PySide6 opencv-python numpy tifffile imagecodecs
UV_RUN := uv run $(foreach d,$(DEPS),--with $(d)) --with pytest

.PHONY: check syntax lint test boot all

all: check

check: syntax lint test boot

syntax:
	@echo "=== Syntax check ==="
	@for f in puzzle_joiner/*.py tests/*.py; do \
		python -c "import ast; ast.parse(open('$$f').read())" || exit 1; \
	done
	@echo "OK"

lint:
	@echo "=== Lint (undefined names) ==="
	@$(UV_RUN) --with pyflakes -m pyflakes puzzle_joiner/ tests/ 2>&1 \
		| grep "undefined name" | grep -v "'PuzzlePiece'" || true
	@! $(UV_RUN) --with pyflakes -m pyflakes puzzle_joiner/ tests/ 2>&1 \
		| grep "undefined name" | grep -qv "'PuzzlePiece'"
	@echo "OK"

test:
	@echo "=== Unit tests ==="
	$(UV_RUN) -m pytest tests/ -x -q --ignore=tests/test_boot.py --tb=short

boot:
	@echo "=== Boot test ==="
	QT_QPA_PLATFORM=offscreen $(UV_RUN) -m pytest tests/test_boot.py -x -q --tb=short
