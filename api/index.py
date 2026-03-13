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
        "/riboreach": riboreach_app,
        "/vaxflow": vaxflow_app,
        "/viroseek": viroseek_app,
    },
)

