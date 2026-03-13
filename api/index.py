"""
Unified Vercel API entrypoint for AI Horizons.

This mounts each product-specific Flask app under a namespaced path:
- /api/riboreach/api/*
- /api/vaxflow/api/*
- /api/viroseek/api/*
"""

import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, send_from_directory
from werkzeug.middleware.dispatcher import DispatcherMiddleware


# The api folder is now self-contained, so imports work naturally from 'api' root
# or relative if structure allows.
# But since we moved everything into 'api/', the previous 'ROOT_DIR' logic is unneeded
# unless imports rely on 'riboreach' being top-level module.
# Let's adjust sys.path to be current directory (api).

API_DIR = Path(__file__).resolve().parent
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))


from riboreach import api as riboreach_app_module
from vaxflow import api as vaxflow_app_module
from viroseek import api as viroseek_app_module

riboreach_app = riboreach_app_module.app
vaxflow_app = vaxflow_app_module.app
viroseek_app = viroseek_app_module.app


root_app = Flask(__name__, static_folder='static', template_folder='templates')


@root_app.get("/")
def home():
    return render_template("index.html")

@root_app.get("/riboreach")
def riboreach_page():
    return render_template("riboreach.html")

@root_app.get("/vaxflow")
def vaxflow_page():
    return render_template("vaxflow.html")

@root_app.get("/viroseek")
def viroseek_page():
    return render_template("viroseek.html")

@root_app.get("/styles.css")
def styles():
    return send_from_directory("static", "styles.css")

@root_app.route("/<path:path>")
def catch_all(path):
    """
    Catch-all route for other static files or client-side routing if necessary.
    But primarily to ensure 404s are handled gracefully or static files like images work.
    """
    try:
        if (API_DIR / "static" / path).exists():
            return send_from_directory("static", path)
        elif (API_DIR / "templates" / path).exists():
             return render_template(path)
        else:
             # Fallback for paths that might be templates without extension
             if (API_DIR / "templates" / f"{path}.html").exists():
                 return render_template(f"{path}.html")
    except Exception as e:
        print(f"Error serving {path}: {e}")
    
    return f"Page not found: {path}", 404



@root_app.get("/index.html")
def home_explicit():
    return render_template("index.html")

@root_app.get("/riboreach.html")
def riboreach_page_explicit():
    return render_template("riboreach.html")

@root_app.get("/vaxflow.html")
def vaxflow_page_explicit():
    return render_template("vaxflow.html")

@root_app.get("/viroseek.html")
def viroseek_page_explicit():
    return render_template("viroseek.html")


app = DispatcherMiddleware(
    root_app,
    {
        "/api/riboreach": riboreach_app,
        "/api/vaxflow": vaxflow_app,
        "/api/viroseek": viroseek_app,
    },
)

