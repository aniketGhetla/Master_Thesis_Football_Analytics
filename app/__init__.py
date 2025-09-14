# app/__init__.py
import os
from flask import Flask, render_template, request
from sqlalchemy import create_engine
from .routes.about import bp as about_bp






# pipeline/services
from app.services.preprocessing import run_normalization
from app.services.image_generation import generate_images_from_sportec
from app.services.phase_inference import run_phase_inference
from app.services.phase_masks import run_phase_masks
from app.services.formation_templates import main as run_formation_templates
from app.services.apply_shaw_ordering import run_apply_shaw
from app.services.formation_prediction import run_formation_prediction
from app.services.formation_vs_formation import run_formation_vs_formation
from app.services.gnn_dataset_step import run_gnn_dataset_step
from app.services.kg_adapter import ensure_kg_tables, ingest_counters_from_fvf
from app.services.reco_batch_step import run_reco_batch


def create_app():
    # Resolve project root and wire template/static explicitly
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    app = Flask(
        __name__,
        template_folder=os.path.join(BASE_DIR, "app", "templates"),
        static_folder=os.path.join(BASE_DIR, "app", "static"),
    )

    # Basic config
    app.config["SECRET_KEY"] = "dev"
    app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "uploads")
    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    DB_URL = "postgresql+psycopg2://postgres:123@localhost:5432/Football"

    # Blueprints
    
    from .routes.home import bp as home_bp
    from .routes.upload import bp as upload_bp
    from app.routes.dashboard_new import bp as dashboard_new_bp


    app.register_blueprint(about_bp)
    app.register_blueprint(home_bp)
    app.register_blueprint(upload_bp)
    app.register_blueprint(dashboard_new_bp)

    # Simple pipeline trigger page
    @app.route("/dashboard-new", methods=["GET", "POST"])
    def generate_dashboard():
        messages = []

        if request.method == "POST":
            umid = request.form.get("unique_match_id")  # may be None

            # 1) normalization
            try:
                table_name = run_normalization(unique_match_id=umid)
                messages.append(f"Normalization done ({table_name}).")
            except Exception as e:
                messages.append(f"Normalization failed: {e}")

            # 2) images
            try:
                n_img, outdir = generate_images_from_sportec(unique_match_id=umid)
                messages.append(f"Generated {n_img} images at '{outdir}'.")
            except Exception as e:
                messages.append(f"Image generation failed: {e}")

            # 3) phase inference
            try:
                n_pred, folder = run_phase_inference(unique_match_id=umid)
                messages.append(f"Wrote {n_pred} phase predictions from '{folder}'.")
            except Exception as e:
                messages.append(f"Phase inference failed: {e}")

            # 4) phase masks
            try:
                n_mask = run_phase_masks()
                messages.append(f"Applied masks/eligibility to {n_mask} rows.")
            except Exception as e:
                messages.append(f"Phase masks failed: {e}")

            # 5) formation templates
            try:
                n_form = run_formation_templates()
                messages.append(f"Wrote {n_form} formation blocks.")
            except Exception as e:
                messages.append(f"Formation templates failed: {e}")

            # 6) apply Shaw ordering
            try:
                n_shaw = run_apply_shaw()
                messages.append(
                    f"Applied Shaw ordering: wrote {n_shaw} rows to formation_templates_shaw_temp."
                )
            except Exception as e:
                messages.append(f"Apply Shaw ordering failed: {e}")

            # 7) formation detection
            try:
                n_pred_forms = run_formation_prediction()
                messages.append(
                    f"Formation predictions: {n_pred_forms} rows into formation_predictions_temp."
                )
            except Exception as e:
                messages.append(f"Formation prediction failed: {e}")

            # 8) formation vs formation
            try:
                n_fvf = run_formation_vs_formation()
                messages.append(f"Formation-vs-formation rows: {n_fvf}.")
            except Exception as e:
                messages.append(f"Formation-vs-formation failed: {e}")

            # 9) GNN dataset
            try:
                gnn_info = run_gnn_dataset_step(
                    provider="sportec",
                    formation_table="formation_vs_formation_new_temp",
                    limit=500,
                )
                messages.append(
                    f"GNN-ready graphs: {gnn_info['num_graphs']} "
                    f"(avg nodes {gnn_info['avg_nodes']}, avg edges {gnn_info['avg_edges']})."
                )
            except Exception as e:
                messages.append(f"GNN dataset step failed: {e}")

            # 10) KG ingest
            try:
                engine = create_engine(DB_URL)
                ensure_kg_tables(engine)  # creates temp-aware kg tables
                ingest_counters_from_fvf(engine, provider=None)
                messages.append("KG built (kg_nodes_temp/kg_edges_temp).")
            except Exception as e:
                messages.append(f"KG step failed: {e}")

            # 11) (optional) recommender batch
            try:
                reco_batch = run_reco_batch(
                    provider="sportec", our_side="home", strategy="both"
                )
                recos = ", ".join(
                    f"{r['strategy']}→{r['recommended_formation']}"
                    for r in reco_batch["results"]
                )
                messages.append(
                    f"Recos ready vs {reco_batch['opponent_form']} "
                    f"({reco_batch['opponent_phase']}): {recos}."
                )
            except Exception as e:
                messages.append(f"Recommender batch failed: {e}")

        else:
            messages.append("Dashboard loaded (no processing).")

        msg = " ".join(messages)
        return render_template("dashboard_new.html", message=msg)

    return app
