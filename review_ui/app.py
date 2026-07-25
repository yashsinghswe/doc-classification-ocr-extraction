"""
Review UI — Flask backend
Run: python review_ui/app.py
Open: http://localhost:5050
"""

import os
import sys

# Allow imports from the parent doc_pipeline/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, render_template, request
from phase6_queue import confirm_review, get_item, get_pending_items, get_stats

app = Flask(__name__)


@app.route("/")
def index():
    stats = get_stats()
    items = get_pending_items(limit=50)
    return render_template("review.html", stats=stats, items=items)


@app.route("/api/items")
def api_items():
    items = get_pending_items(limit=50)
    return jsonify(items)


@app.route("/api/item/<item_id>")
def api_item(item_id):
    item = get_item(item_id)
    if not item:
        return jsonify({"error": "not found"}), 404
    return jsonify(item)


@app.route("/api/confirm", methods=["POST"])
def api_confirm():
    data = request.get_json(force=True)
    ok = confirm_review(
        item_id       = data["id"],
        confirmed_type = data["confirmed_type"],
        corrected_data = data.get("corrected_data", {}),
        note           = data.get("note", ""),
    )
    return jsonify({"success": ok})


@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats())


if __name__ == "__main__":
    print("Review UI → http://localhost:5050")
    app.run(debug=True, port=5050)
