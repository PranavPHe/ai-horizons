"""
Unified Vercel API entrypoint for AI Horizons.

This mounts each product-specific Flask app under a namespaced path:
- /api/riboreach/api/*
- /api/vaxflow/api/*
- /api/viroseek/api/*
"""

import sys
from pathlib import Path

from flask import Flask, jsonify
from werkzeug.middleware.dispatcher import DispatcherMiddleware


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from riboreach.api import app as riboreach_app
from vaxflow.api import app as vaxflow_app
from viroseek.api import app as viroseek_app


root_app = Flask(__name__)


@root_app.get("/")
def root_health():
    return jsonify(
        {
            "status": "ok",
            "service": "ai-horizons-api",
            "routes": {
                "riboreach": "/api/riboreach/api/*",
                "vaxflow": "/api/vaxflow/api/*",
                "viroseek": "/api/viroseek/api/*",
            },
        }
    )


app = DispatcherMiddleware(
    root_app,
    {
        "/riboreach": riboreach_app,
        "/vaxflow": vaxflow_app,
        "/viroseek": viroseek_app,
    },
)
