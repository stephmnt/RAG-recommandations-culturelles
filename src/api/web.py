"""Web UI routes for local Flask frontend (HTML5UP-based)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Blueprint, abort, render_template, send_from_directory


TEMPLATES_ROOT = Path(__file__).resolve().parent / "templates"
ASSETS_ROOT = TEMPLATES_ROOT / "assets"
IMAGES_ROOT = TEMPLATES_ROOT / "images"

web_bp = Blueprint("web", __name__, template_folder="templates")


@web_bp.get("/")
@web_bp.get("/app")
def home() -> str:
    return render_template("eventually_app.html")


@web_bp.get("/assets/<path:filename>")
def assets(filename: str) -> Any:
    if not ASSETS_ROOT.exists():
        abort(404)
    return send_from_directory(ASSETS_ROOT, filename)


@web_bp.get("/images/<path:filename>")
def images(filename: str) -> Any:
    if not IMAGES_ROOT.exists():
        abort(404)
    return send_from_directory(IMAGES_ROOT, filename)


__all__ = ["web_bp"]
