"""Development entry point:  python run.py   (PythonAnywhere uses its WSGI file instead)."""
from app import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
