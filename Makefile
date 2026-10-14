.DEFAULT_GOAL := help
.PHONY: help install fetch table1 clean

help:  ## Show this help
	@grep -hE '^[a-z0-9-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk -F':.*?## ' '{printf "  \033[1m%-9s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "  Order:  install -> fetch -> sbatch slurm/uq_all.sh -> table1"
	@echo "  See README.md."

install:  ## Install the package
	python -m pip install -e .

fetch:  ## Download the combined generation files (~100 MB)
	utoolcall fetch-combined

table1:  ## Build Table 1 from scores in ./uq (run the sweep first)
	utoolcall table1 -o table1.tex

clean:  ## Remove build artefacts
	rm -rf build dist src/*.egg-info table1.tex
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
