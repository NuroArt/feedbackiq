.PHONY: install hooks dev test lint fmt preflight e2e e2e-live eval eval-stub eval-baseline docker run-docker clean

install:
	pip install -r requirements-dev.txt

# Credential-scanning pre-commit hook. Run once after cloning.
hooks:
	scripts/setup-hooks.sh

dev:
	uvicorn app.main:app --reload --port 8000

test:
	LLM_OFFLINE_MODE=true pytest

lint:
	ruff check . && ruff format --check . && mypy app

fmt:
	ruff format . && ruff check --fix .

# First live run: validates key, schema, a real API call, and theme reuse.
preflight:
	python scripts/preflight.py

# End-to-end against real GitHub and Slack. e2e is a dry run; e2e-live writes
# for real and runs twice to prove the issue dedup actually holds.
e2e:
	python scripts/e2e.py

e2e-live:
	python scripts/e2e.py --live --twice

# Golden-set eval. `eval` costs tokens; `eval-stub` is free and exercises the
# harness. Baseline updates are deliberate — never automatic.
eval:
	python -m evals.runner --profile live --repeats 3

eval-stub:
	python -m evals.runner --profile stub --repeats 1

eval-baseline:
	python -m evals.runner --profile live --repeats 3 --update-baseline

docker:
	docker build -t feedbackiq:local .

run-docker: docker
	docker run --rm -p 8000:8000 --env-file .env feedbackiq:local

smoke:
	curl -fsS localhost:8000/healthz && \
	curl -fsS -X POST localhost:8000/v1/feedback/analyze \
	  -H 'content-type: application/json' \
	  -d @samples/request.json | head -c 600

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} + ; rm -rf .pytest_cache .ruff_cache
