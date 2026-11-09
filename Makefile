# camdl-scope v2 — dev/build/serve.
#
# API: FastAPI under uvicorn on :8800 (serves /api and, once built, web/dist).
# Web: Vite dev server with HMR, proxying /api -> :8800 during development.

API_HOST ?= 127.0.0.1
API_PORT ?= 8800

.PHONY: dev dev-api dev-web build serve test

## dev: run API (reload) + Vite dev server together; Ctrl-C stops both
dev:
	@trap 'kill 0' INT TERM EXIT; \
	uv run uvicorn camdl_scope.api.app:app --reload --host $(API_HOST) --port $(API_PORT) & \
	( cd web && npm run dev ) & \
	wait

## dev-api: API only, with autoreload
dev-api:
	uv run uvicorn camdl_scope.api.app:app --reload --host $(API_HOST) --port $(API_PORT)

## dev-web: Vite dev server only (expects the API on :8800)
dev-web:
	cd web && npm run dev

## build: produce the production frontend bundle in web/dist
build:
	cd web && npm run build

## serve: build the frontend, then serve API + SPA on one port
serve: build
	uv run camdl-scope --host $(API_HOST) --port $(API_PORT)

## test: Python test suite
test:
	uv run pytest -q

## fixture: (re)generate the deterministic golden fit store for dev/tests
fixture:
	uv run python -m tests.fixtures.make_golden_store

## types: regenerate web/src/api/types.ts from the FastAPI OpenAPI schema
types:
	uv run python -c "import json; from camdl_scope.api.app import app; print(json.dumps(app.openapi()))" > web/openapi.json
	cd web && npx --yes openapi-typescript openapi.json -o src/api/types.ts

## demo: serve API + SPA against the golden fixture store (reachable on LAN/Tailscale)
demo: fixture build
	uv run camdl-scope --host 0.0.0.0 --port 8800 --store tests/fixtures/golden-store
