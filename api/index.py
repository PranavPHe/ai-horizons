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

