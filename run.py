# run.py
from app import create_app
from app.routes import prep
app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
