"""
Notifier: Notification system for HITL triggers and run completion.

Provides:
- Atomic sentinel file writes for external monitor polling
- Browser auto-open for HITL panel
- Desktop notifications (macOS, Linux, fallback)
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SENTINEL_FILENAME = ".notification.json"


class Notifier:
    """Notification hub for HITL events and run completion."""

    def __init__(self, output_dir: Path, auto_open: bool = False):
        """
        Args:
            output_dir: Directory for sentinel files and panel output.
            auto_open: If True, auto-open browser on HITL trigger.
                       Default False (safe for headless/CI).
        """
        self.output_dir = Path(output_dir)
        self.auto_open = auto_open

    def notify_hitl_required(
        self,
        checklist: Any,
        panel_path: Path | None = None,
        run_id: str = "",
    ) -> Path:
        """Notify that HITL review is required.

        Args:
            checklist: HITLChecklist instance with pending items.
            panel_path: Path to HITL panel HTML (if built).
            run_id: Current run identifier.

        Returns:
            Path to the sentinel file written.
        """
        pending = checklist.pending_count if hasattr(checklist, "pending_count") else 0

        # Count critical items
        critical_count = 0
        if hasattr(checklist, "items"):
            for item in checklist.items:
                if not item.resolved and hasattr(item, "trigger_id"):
                    # Heuristic: items with "CRITICAL" in context or trigger
                    if "CRITICAL" in str(getattr(item, "context", {})):
                        critical_count += 1

        message = f"{pending} issues require human review"
        if critical_count:
            message += f" ({critical_count} CRITICAL)"

        data = {
            "event": "hitl_required",
            "pending_count": pending,
            "critical_count": critical_count,
            "message": message,
            "panel_path": str(panel_path) if panel_path else None,
        }

        sentinel_path = self._write_sentinel(data, run_id=run_id)

        # Desktop notification
        self._desktop_notify(
            title="OpenCausality: HITL Review Required",
            message=message,
        )

        # Browser open
        if self.auto_open and panel_path and panel_path.exists():
            self._open_browser(panel_path)

        logger.info(f"HITL notification sent: {message}")
        return sentinel_path

    def notify_run_complete(
        self,
        report: Any,
        run_id: str = "",
    ) -> Path:
        """Notify that a run has completed.

        Args:
            report: SystemReport instance.
            run_id: Current run identifier.

        Returns:
            Path to the sentinel file written.
        """
        edge_count = len(report.edge_summaries) if hasattr(report, "edge_summaries") else 0
        blocked_count = len(report.blocked_edges) if hasattr(report, "blocked_edges") else 0
        mean_cred = getattr(report, "mean_credibility_score", 0.0)

        message = (
            f"Run complete: {edge_count} edges estimated, "
            f"{blocked_count} blocked, mean credibility {mean_cred:.2f}"
        )

        data = {
            "event": "run_complete",
            "edges_estimated": edge_count,
            "edges_blocked": blocked_count,
            "mean_credibility": round(mean_cred, 3),
            "message": message,
        }

        sentinel_path = self._write_sentinel(data, run_id=run_id)

        self._desktop_notify(
            title="OpenCausality: Run Complete",
            message=message,
        )

        logger.info(f"Run-complete notification sent: {message}")
        return sentinel_path

    def _write_sentinel(
        self,
        data: dict[str, Any],
        run_id: str = "",
    ) -> Path:
        """Write sentinel JSON atomically (write tmp then rename).

        Args:
            data: Event payload.
            run_id: Current run identifier.

        Returns:
            Path to the sentinel file.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sentinel_path = self.output_dir / SENTINEL_FILENAME

        payload = {
            "schema_version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            **data,
        }

        # Atomic write: write to temp file then rename
        fd, tmp_path = tempfile.mkstemp(
            suffix=".json.tmp",
            dir=str(self.output_dir),
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp_path, sentinel_path)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        return sentinel_path

    def _open_browser(self, path: Path) -> None:
        """Open a file in the default browser using file URI."""
        try:
            uri = path.resolve().as_uri()
            webbrowser.open(uri)
            logger.debug(f"Opened browser: {uri}")
        except Exception as e:
            logger.warning(f"Failed to open browser: {e}")

    def _desktop_notify(self, title: str, message: str) -> None:
        """Send a desktop notification (best-effort, never raises)."""
        try:
            system = platform.system()
            if system == "Darwin":
                self._notify_macos(title, message)
            elif system == "Linux":
                self._notify_linux(title, message)
            else:
                self._notify_fallback(title, message)
        except Exception as e:
            logger.debug(f"Desktop notification failed: {e}")
            self._notify_fallback(title, message)

    def _notify_macos(self, title: str, message: str) -> None:
        """macOS notification via osascript."""
        # Escape double quotes for AppleScript
        safe_title = title.replace('"', '\\"')
        safe_message = message.replace('"', '\\"')
        script = (
            f'display notification "{safe_message}" '
            f'with title "{safe_title}"'
        )
        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            timeout=5,
        )

    def _notify_linux(self, title: str, message: str) -> None:
        """Linux notification via notify-send."""
        if shutil.which("notify-send"):
            subprocess.run(
                ["notify-send", title, message],
                capture_output=True,
                timeout=5,
            )
        else:
            self._notify_fallback(title, message)

    def _notify_fallback(self, title: str, message: str) -> None:
        """Fallback: terminal bell + log."""
        print(f"\a")  # Terminal bell
        logger.info(f"[NOTIFICATION] {title}: {message}")
