"""camdl-scope v2 HTTP layer.

A thin FastAPI projection of the Python core (ingest / state / diagnostics)
as typed JSON, plus static hosting of the built React frontend. The browser
results viewer is a client of this API; nothing here computes statistics —
it serializes what the core already knows.
"""
