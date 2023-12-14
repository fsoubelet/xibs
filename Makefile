# Copyright 2019 Felix Soubelet <felix.soubelet@cern.ch>
# MIT License

# Documentation for most of what you will see here can be found at the following links:
# for the GNU make special targets: https://www.gnu.org/software/make/manual/html_node/Special-Targets.html
# for python packaging: https://docs.python.org/3/distutils/introduction.html

# ANSI escape sequences for colors
# In order, B=bold, C=cyan, D=dark blue, E=end, P=pink, R=red and Y=yellow
B=\033[1m
C=\033[96m
D=\033[34m
E=\033[0m
P=\033[95m
R=\033[31m
Y=\033[33m

.PHONY : help build clean docs format install lines lint typing tests testrepos

all: install

help:
	@echo "Please use 'make $(R)<target>$(E)' where $(R)<target>$(E) is one of:"
	@echo "  $(R) build $(E)  \t  to build wheel and source distribution with $(P)Hatch$(E)."
	@echo "  $(R) clean $(E)  \t  to recursively remove build, run and bitecode files/dirs."
	@echo "  $(R) docs $(E)  \t  to build the documentation for the package with $(P)Sphinx$(E)."
	@echo "  $(R) docrepos $(E)  \t  to $(P)git$(E) clone the necessary repositories to build the documentation gallery."
	@echo "  $(R) format $(E)  \t  to recursively apply PEP8 formatting through the $(P)Black$(E) and $(P)isort$(E) cli tools."
	@echo "  $(R) install $(E)  \t  to $(C)pip install$(E) this package into the current environment."
	@echo "  $(R) lines $(E)  \t  to count lines of code in the package folder with the $(P)tokei$(E) tool."
	@echo "  $(R) lint $(E)  \t  to lint the packages' code though $(P)Ruff$(E)."
	@echo "  $(R) typing $(E)  \t  to run type checking on the codebase with $(P)MyPy$(E)."
	@echo "  $(R) tests $(E)  \t  to run the test suite with $(P)pytest$(E)."
	@echo "  $(R) testrepos $(E)  \t  to $(P)git$(E) clone the necessary repositories for test files."


# ----- Dev Tools Targets ----- #

build: clean
	@echo "Re-building wheel and sdist"
	@hatch build --clean
	@echo "Created build is located in the $(C)dist$(E) folder."

clean:
	@echo "Cleaning up documentation pages."
	@rm -rf doc_build
	@echo "Cleaning up sphinx-gallery build artifacts."
	@rm -rf docs/gallery
	@rm -rf docs/gen_modules
	@echo "Cleaning up package builds and distutils remains."
	@find . -type d -name "*build" -exec rm -rf {} +
	@find . -type d -name "*dist" -exec rm -rf {} +
	@rm -rf xibs.egg-info
	@rm -rf .eggs
	@echo "Cleaning up bitecode files and python cache."
	@find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete
	@echo "Cleaning up pytest cache & test artifacts."
	@find . -type d -name '*.pytest_cache' -exec rm -rf {} + -o -type f -name '*.pytest_cache' -exec rm -rf {} +
	@echo "Cleaning up mypy and ruff caches."
	@find . -type d -name "*.mypy_cache" -exec rm -rf {} +
	@find . -type d -name "*.ruff_cache" -exec rm -rf {} +
	@echo "Cleaning up ipython notebooks cache."
	@find . -type d -name "*.ipynb_checkpoints" -exec rm -rf {} +
	@echo "Cleaning up coverage reports."
	@find . -type f -name '.coverage*' -exec rm -rf {} + -o -type f -name 'coverage.xml' -delete
	@echo "All cleaned up!\n"

docs:
	@echo "Building static pages with $(D)Sphinx$(E)."
	@python -m sphinx -v -b html docs doc_build -d doc_build
	@rm -rf docs/sg_execution_times.rst

docrepos:
	@echo "Cloning acc-models-lhc repo, 2023 branch."
	@git clone -b 2023 https://gitlab.cern.ch/acc-models/acc-models-lhc.git --depth 1
	@echo "Moving acc-models-lhc repo to examples folder."
	@mv acc-models-lhc examples/

format:
	@echo "Formatting code to PEP8 with $(P)isort$(E) and $(P)Black$(E). Max line length is 110 characters."
	@python -m isort tests xibs && black tests xibs

install: clean
	@echo "Installing with $(D)pip$(E) in the current environment."
	@python -m pip install . -v

lines: format
	@tokei xibs --exclude xibs/_old_michalis.py

lint: format
	@echo "Linting code with $(P)Ruff$(E)."
	@ruff check xibs/

typing: format
	@echo "Checking code typing with $(P)MyPy$(E)."
	@python -m mypy xibs


# ----- Tests Targets ----- #

tests: clean
	@python -m pytest -n auto -v

testrepos:  # git cloning the necessary repos for tests files - specify branch, could also specify tag to make sure we are static
	@echo "Cloning acc-models-ps repo, 2023 branch."
	@git clone -b 2023 https://gitlab.cern.ch/acc-models/acc-models-ps.git --depth 1
	@echo "Cloning acc-models-sps repo, 2021 branch."
	@git clone -b 2021 https://gitlab.cern.ch/acc-models/acc-models-sps.git --depth 1
	@echo "Cloning acc-models-lhc repo, 2023 branch."
	@git clone -b 2023 https://gitlab.cern.ch/acc-models/acc-models-lhc.git --depth 1

# Catch-all unknow targets without returning an error. This is a POSIX-compliant syntax.
.DEFAULT:
	@echo "Make caught an invalid target! See help output below for available targets."
	@make help
