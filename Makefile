all: install

.PHONY: install test

install:
	@echo "Installing UM2N..."
	@python3 -m pip install -e .
	@echo "Done."

install_dev:
	@echo "Installing UM2N for development..."
	@python3 -m pip install -e .[dev]
	@echo "Done."
	@echo "Setting up pre-commit..."
	@pre-commit install
	@echo "Done."

lint:
	@echo "Checking lint..."
	@ruff check
	@echo "PASS"

test: lint
	@echo "Running all tests..."
	@python3 -m pytest -v --durations=20 test
	@echo "Done."

coverage:
	@echo "Generating coverage report..."
	@python3 -m coverage erase
	@python3 -m coverage run --source=UM2N -m pytest -v test
	@python3 -m coverage html
	@echo "Done."

tree:
	@tree -d .
