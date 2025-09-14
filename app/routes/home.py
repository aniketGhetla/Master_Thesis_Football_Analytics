# app/routes/home.py
from flask import Blueprint, render_template
from app.services.queries import count_rows, count_distinct_match_ids

bp = Blueprint("home", __name__)

@bp.route("/")
def index():
    counts = {
        "wide": count_rows("sportec_databall_wide_temp"),
        "event": count_rows("sportec_event_new_temp"),
        "merged": count_rows("sportec_databall_merged_temp"),
        "matches": count_distinct_match_ids("sportec_databall_merged_temp"),
    }
    return render_template("home.html", counts=counts)


