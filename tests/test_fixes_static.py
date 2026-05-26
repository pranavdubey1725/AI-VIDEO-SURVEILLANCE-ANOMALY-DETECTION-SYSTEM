"""
Static code audit tests — no server required.

Verifies that all 6 priority fixes are actually present in the source files.
Run from the surveillance-system directory:
    pytest tests/test_fixes_static.py -v
"""

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── helpers ───────────────────────────────────────────────────────────────────

def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# FIX 1 — showSection() must use "flex" not "" for processing/results
# ══════════════════════════════════════════════════════════════════════════════

class TestFix1ShowSection:
    """ui/app.js showSection() must set explicit 'flex' for processing/results."""

    def setup_method(self):
        self.src = read("ui/app.js")

    def test_processing_uses_flex(self):
        """processingSection display must be set to "flex", not empty string."""
        assert re.search(
            r'processingSection\.style\.display\s*=\s*name\s*===\s*"processing"\s*\?\s*"flex"',
            self.src
        ), (
            "BUG: processingSection still uses '' (empty string). "
            "CSS display:none will override it and the section will never show."
        )

    def test_results_uses_flex(self):
        """resultsSection display must be set to "flex", not empty string."""
        # Use regex to ignore alignment whitespace between tokens
        assert re.search(
            r'resultsSection\.style\.display\s*=\s*name\s*===\s*"results"\s*\?\s*"flex"',
            self.src
        ), (
            "BUG: resultsSection still uses '' (empty string). "
            "CSS display:none will override it and the section will never show."
        )

    def test_empty_string_not_used_for_hidden_sections(self):
        """The old broken pattern should not appear in showSection."""
        # Old broken code: name === "processing" ? "" : "none"
        assert 'name === "processing" ? "" :' not in self.src
        assert 'name === "results" ? "" :' not in self.src

    def test_upload_section_still_uses_empty_string(self):
        """upload-section has no display:none in CSS so empty string is correct for it."""
        assert 'uploadSection.style.display     = name === "upload"     ? ""' in self.src, (
            "upload section should still use '' (clears inline style, CSS has no display:none)"
        )

    def test_processing_section_css_has_display_none(self):
        """Confirm CSS still has display:none so we know the fix was actually needed."""
        css = read("ui/style.css")
        # Find the .processing-section rule and confirm display:none is there
        match = re.search(r'\.processing-section\s*\{([^}]+)\}', css)
        assert match, ".processing-section rule not found in style.css"
        assert "display: none" in match.group(1), (
            ".processing-section no longer has display:none in CSS — "
            "verify the showSection fix is still necessary"
        )

    def test_results_section_css_has_display_none(self):
        """Confirm CSS still has display:none so we know the fix was actually needed."""
        css = read("ui/style.css")
        match = re.search(r'\.results-section\s*\{([^}]+)\}', css)
        assert match, ".results-section rule not found in style.css"
        assert "display: none" in match.group(1)


# ══════════════════════════════════════════════════════════════════════════════
# FIX 2a — Frontend: sendBeacon → fetch with keepalive
# ══════════════════════════════════════════════════════════════════════════════

class TestFix2FrontendSendBeacon:
    """ui/app.js must NOT use sendBeacon for DELETE (it only supports POST)."""

    def setup_method(self):
        self.src = read("ui/app.js")

    def test_no_send_beacon(self):
        """navigator.sendBeacon( call must be removed (comments about it are fine)."""
        # The word 'sendBeacon' may appear in comments explaining why it was replaced.
        # We check for the actual live call: navigator.sendBeacon(
        assert "navigator.sendBeacon(" not in self.src, (
            "BUG: navigator.sendBeacon( is still present as a live call. "
            "It only sends POST — DELETE endpoint will return 405."
        )

    def test_uses_fetch_with_keepalive(self):
        """Replacement must be fetch(..., { keepalive: true })."""
        assert "keepalive: true" in self.src, (
            "BUG: fetch with keepalive:true not found. "
            "Without keepalive, fetch is cancelled when the page unloads."
        )

    def test_keepalive_fetch_uses_delete_method(self):
        """The keepalive fetch must use DELETE method."""
        # Look for the pattern: method: "DELETE" near keepalive: true
        # Both should appear in the beforeunload handler
        beforeunload_block = re.search(
            r'beforeunload.*?}\s*\)', self.src, re.DOTALL
        )
        assert beforeunload_block, "beforeunload handler not found"
        block_text = beforeunload_block.group(0)
        assert '"DELETE"' in block_text, (
            "keepalive fetch in beforeunload must use DELETE method"
        )


# ══════════════════════════════════════════════════════════════════════════════
# FIX 2b — Backend: TTL eviction + logging + created_at
# ══════════════════════════════════════════════════════════════════════════════

class TestFix2BackendTTL:
    """api/main.py must have job TTL eviction, logging, and created_at."""

    def setup_method(self):
        self.src = read("api/main.py")

    def test_job_ttl_constant_defined(self):
        """JOB_TTL_SECONDS constant must exist."""
        assert "JOB_TTL_SECONDS" in self.src, (
            "BUG: JOB_TTL_SECONDS not defined. Jobs accumulate in memory forever."
        )

    def test_ttl_value_is_reasonable(self):
        """TTL should be between 5 minutes and 24 hours."""
        match = re.search(r'JOB_TTL_SECONDS\s*=\s*(\d+)', self.src)
        assert match, "JOB_TTL_SECONDS definition not found"
        ttl = int(match.group(1))
        assert 300 <= ttl <= 86400, (
            f"JOB_TTL_SECONDS={ttl} is outside reasonable range [300, 86400]"
        )

    def test_eviction_function_exists(self):
        """Background eviction coroutine must exist."""
        assert "_evict_expired_jobs" in self.src, (
            "BUG: _evict_expired_jobs not found. No automatic memory cleanup."
        )

    def test_eviction_is_scheduled_on_startup(self):
        """Eviction task must be started during app startup."""
        startup_block = re.search(
            r'async def load_pipeline.*?(?=\n@|\nclass |\Z)', self.src, re.DOTALL
        )
        assert startup_block, "load_pipeline startup handler not found"
        assert "_evict_expired_jobs" in startup_block.group(0), (
            "BUG: _evict_expired_jobs is defined but never scheduled at startup"
        )

    def test_created_at_added_to_job(self):
        """Each new job must record its creation time."""
        assert '"created_at"' in self.src or "'created_at'" in self.src, (
            "BUG: created_at not set on job creation. "
            "TTL eviction cannot work without knowing when the job was created."
        )

    def test_time_module_imported(self):
        """time module must be imported for TTL timestamps."""
        assert re.search(r'^import time', self.src, re.MULTILINE), (
            "BUG: 'import time' not found. time.time() calls will fail."
        )

    def test_logging_configured(self):
        """Logging must be set up so production errors are visible."""
        assert "logging.basicConfig" in self.src or "logging.getLogger" in self.src, (
            "BUG: No logging configured. All server-side errors are invisible."
        )

    def test_logger_used_in_run_analysis(self):
        """Failures in run_analysis must be logged, not silently re-raised."""
        assert "logger.exception" in self.src or "logger.error" in self.src, (
            "BUG: Errors in run_analysis are not logged — failures invisible in production."
        )


# ══════════════════════════════════════════════════════════════════════════════
# FIX 3 — docker-compose.yml must not contain Streamlit service
# ══════════════════════════════════════════════════════════════════════════════

class TestFix3DockerCompose:
    """docker-compose.yml must have one service (api) not two (api + ui)."""

    def setup_method(self):
        self.src = read("docker-compose.yml")

    def test_no_streamlit_command(self):
        """No service should run streamlit."""
        assert "streamlit" not in self.src.lower(), (
            "BUG: 'streamlit' still in docker-compose.yml. "
            "docker compose up will start a dead Streamlit container."
        )

    def test_no_port_8501(self):
        """Port 8501 (old Streamlit port) should not be mapped."""
        assert "8501" not in self.src, (
            "BUG: Port 8501 still mapped in docker-compose.yml."
        )

    def test_ui_service_removed(self):
        """There should be no 'ui:' service block."""
        assert not re.search(r'^\s{2}ui\s*:', self.src, re.MULTILINE), (
            "BUG: 'ui:' service still present in docker-compose.yml."
        )

    def test_api_service_still_present(self):
        """The api service must still exist."""
        assert re.search(r'^\s{2}api\s*:', self.src, re.MULTILINE), (
            "api: service block missing from docker-compose.yml"
        )

    def test_api_port_8000_still_mapped(self):
        """API port 8000 must still be exposed."""
        assert "8000:8000" in self.src, (
            "Port 8000 mapping removed from docker-compose.yml"
        )


# ══════════════════════════════════════════════════════════════════════════════
# FIX 4 — README.md must not reference Streamlit or port 8501
# ══════════════════════════════════════════════════════════════════════════════

class TestFix4Readme:
    """README.md must not describe the old Streamlit architecture."""

    def setup_method(self):
        self.src = read("README.md")

    def test_no_port_8501(self):
        """Port 8501 (Streamlit) must not appear anywhere in README."""
        assert "8501" not in self.src, (
            "BUG: Port 8501 still referenced in README. "
            "Users will try to open the wrong URL."
        )

    def test_quick_start_uses_port_8000(self):
        """Quick start instructions must point to port 8000."""
        assert "localhost:8000" in self.src, (
            "README quick start should direct users to http://localhost:8000"
        )

    def test_no_streamlit_as_frontend_label(self):
        """README should not describe Streamlit as the frontend technology."""
        # "Streamlit Frontend" label in architecture diagram should be gone
        assert "Streamlit Frontend" not in self.src, (
            "BUG: 'Streamlit Frontend' still in architecture diagram."
        )

    def test_no_streamlit_in_project_structure(self):
        """Old ui/app.py Streamlit entry should not be in project structure."""
        assert "Streamlit frontend" not in self.src, (
            "BUG: Streamlit frontend still listed in project structure."
        )

    def test_vanilla_js_frontend_mentioned(self):
        """README should now describe the vanilla JS / HTML frontend."""
        assert any(term in self.src for term in ["Vanilla JS", "vanilla JS", "HTML/CSS/JS", "index.html"]), (
            "README doesn't mention the new vanilla JS frontend"
        )


# ══════════════════════════════════════════════════════════════════════════════
# FIX 5 — requirements.txt must not contain streamlit
# ══════════════════════════════════════════════════════════════════════════════

class TestFix5Requirements:
    """requirements.txt must not list streamlit."""

    def setup_method(self):
        self.src = read("requirements.txt")

    def test_streamlit_removed(self):
        """streamlit package must not be in requirements."""
        lines = [l.strip().lower() for l in self.src.splitlines()]
        streamlit_lines = [l for l in lines if l.startswith("streamlit")]
        assert len(streamlit_lines) == 0, (
            f"BUG: streamlit still in requirements.txt: {streamlit_lines}"
        )

    def test_fastapi_still_present(self):
        """fastapi must still be in requirements."""
        assert "fastapi" in self.src.lower()

    def test_uvicorn_still_present(self):
        """uvicorn must still be in requirements."""
        assert "uvicorn" in self.src.lower()

    def test_torch_still_present(self):
        """torch must still be in requirements."""
        assert "torch" in self.src.lower()


# ══════════════════════════════════════════════════════════════════════════════
# FIX 6 — .gitignore must have *.log rule
# ══════════════════════════════════════════════════════════════════════════════

class TestFix6Gitignore:
    """*.log must be excluded by .gitignore."""

    def setup_method(self):
        self.src = read(".gitignore")

    def test_log_files_ignored(self):
        """*.log pattern must be in .gitignore."""
        lines = [l.strip() for l in self.src.splitlines()]
        assert "*.log" in lines, (
            "BUG: '*.log' not in .gitignore. "
            "Log files like api/api.log will be committed."
        )

    def test_log_files_not_tracked_by_git(self):
        """api/api.log and ui/streamlit.log must not be tracked by git."""
        import subprocess
        result = subprocess.run(
            ["git", "ls-files", "api/api.log", "ui/streamlit.log"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        tracked = result.stdout.strip()
        assert tracked == "", (
            f"BUG: These log files are still tracked by git: {tracked}\n"
            "Run: git rm --cached api/api.log ui/streamlit.log"
        )
