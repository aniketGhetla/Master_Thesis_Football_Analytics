from flask import Blueprint, render_template, request
from app.services.preprocessing import run_normalization

bp = Blueprint("prep", __name__)

@bp.route("/dashboard", methods=["POST"])
def generate_dashboard():
    # run preprocessing step
    table_name = run_normalization()

    # render dashboard.html (you’ll wire actual data here later)
    return render_template("dashboard.html", message=f"Using data from {table_name}")
