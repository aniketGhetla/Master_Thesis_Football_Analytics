# app/routes/upload.py
import os
from flask import Blueprint, current_app, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from app.services.ingestion import process_upload

bp = Blueprint("upload", __name__, url_prefix="/upload")

ALLOWED_EXT = {".xml"}

def _save_file(file_storage, name_hint):
    filename = secure_filename(file_storage.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXT:
        raise ValueError(f"Only XML files are allowed: got {ext}")
    save_as = f"{name_hint}{ext}"
    path = os.path.join(current_app.config["UPLOAD_FOLDER"], save_as)
    file_storage.save(path)
    return path

@bp.get("/")
def upload_form():
    return render_template("upload.html")

@bp.post("/")
def handle_upload():
    try:
        tracking = request.files.get("tracking_file")
        event = request.files.get("event_file")
        meta = request.files.get("meta_file")
        if not (tracking and event and meta):
            flash("Please provide all three files.", "error")
            return redirect(url_for("upload.upload_form"))

        tracking_path = _save_file(tracking, "tracking")
        event_path = _save_file(event, "event")
        meta_path = _save_file(meta, "meta")

        match_id = process_upload(tracking_path, event_path, meta_path)
        return render_template("upload_result.html", match_id=match_id)

    except Exception as e:
        # Minimal error reporting for now
        return render_template("upload_result.html", match_id=f"ERROR: {e}")
