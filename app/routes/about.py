# app/routes/about.py
from flask import Blueprint, render_template

bp = Blueprint("about", __name__, url_prefix="/about")

@bp.route("/")
def about():
    # purely static page; no data dependencies
    return render_template("about.html")
