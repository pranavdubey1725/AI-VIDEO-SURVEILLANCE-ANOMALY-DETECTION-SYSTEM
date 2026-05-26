# Tests

All tests require the FastAPI server to be running unless noted.

**Start the server:**
```bash
cd surveillance-system
python run.py
```

---

## Test files

| File | Needs server | What it covers |
|---|---|---|
| `test_api.py` | Yes | Core API endpoints — health, job lifecycle, input validation |
| `test_fixes_static.py` | No | Static source-code audit of all 6 priority bug fixes |
| `test_fixes_runtime.py` | Yes | Live API behaviour — TTL eviction logic, static file serving, end-to-end smoke |

---

## Run all tests

```bash
# From surveillance-system/
pytest tests/ -v
```

## Run only static checks (no server needed)

```bash
pytest tests/test_fixes_static.py -v
```

## Run API + runtime tests

```bash
# Start server first, then:
pytest tests/test_api.py tests/test_fixes_runtime.py -v
```

---

## What each file tests

### `test_api.py` — 14 tests
Core endpoint coverage:
- `GET /health` returns correct model metadata
- `GET/DELETE /jobs/{id}` — 404 for unknown jobs
- `POST /analyze` — missing file, wrong extension, empty file, threshold out of range
- Too-short video fails gracefully with a readable error
- Full job lifecycle: submit → poll → fetch results → fetch frame → delete → confirm gone

### `test_fixes_static.py` — 33 tests
Reads source files directly (no imports, no server). Verifies:
- **Fix 1** — `showSection()` uses `"flex"` not `""` for processing/results sections
- **Fix 2 (frontend)** — `navigator.sendBeacon(` removed, `fetch keepalive:true` added
- **Fix 2 (backend)** — `JOB_TTL_SECONDS`, `_evict_expired_jobs`, `created_at`, `logging` present
- **Fix 3** — `docker-compose.yml` has no Streamlit `ui` service
- **Fix 4** — `README.md` has no port 8501 or Streamlit references
- **Fix 5** — `requirements.txt` does not list `streamlit`
- **Fix 6** — `*.log` is in `.gitignore` and log files are not tracked by git

### `test_fixes_runtime.py` — 15 tests
Hits the live API and verifies behaviour:
- Server reachable and pipeline loaded
- Job creation, status polling, DELETE
- UI served at `GET /` (StaticFiles mount working)
- `app.js` and `style.css` served correctly
- TTL eviction logic: expired jobs removed, fresh jobs kept
- Port 8501 (old Streamlit) not in use
- Too-short video returns a user-friendly error (no raw traceback)
- Full pipeline smoke test (skipped if NumPy incompatibility detected)
