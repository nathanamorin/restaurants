import json
import os
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

DATA_FILE = Path(__file__).parent / "data" / "restaurants.json"


def load_restaurants():
    """Load restaurants from JSON file."""
    if not DATA_FILE.exists():
        return []
    with open(DATA_FILE, "r") as f:
        return json.load(f)


def save_restaurants(restaurants):
    """Save restaurants to JSON file."""
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_FILE, "w") as f:
        json.dump(restaurants, f, indent=2)


def get_next_id(restaurants):
    """Get the next available restaurant ID."""
    if not restaurants:
        return 1
    return max(r["id"] for r in restaurants) + 1


@app.route("/")
def index():
    """Render the main page."""
    restaurants = load_restaurants()
    return render_template("index.html", restaurants=restaurants)


@app.route("/api/restaurants", methods=["GET"])
def get_restaurants():
    """Return all restaurants as JSON."""
    restaurants = load_restaurants()
    return jsonify(restaurants)


@app.route("/api/restaurants", methods=["POST"])
def add_restaurant():
    """Add a new restaurant to the list."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    required_fields = ["name", "cuisine", "address"]
    missing = [f for f in required_fields if not data.get(f)]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    restaurants = load_restaurants()
    restaurant = {
        "id": get_next_id(restaurants),
        "name": data["name"].strip(),
        "cuisine": data["cuisine"].strip(),
        "address": data["address"].strip(),
        "price_range": data.get("price_range", "$$"),
        "rating": float(data["rating"]) if data.get("rating") else None,
        "notes": data.get("notes", "").strip(),
        "added_by": data.get("added_by", "team").strip(),
    }
    restaurants.append(restaurant)
    save_restaurants(restaurants)
    return jsonify(restaurant), 201


@app.route("/api/restaurants/<int:restaurant_id>", methods=["DELETE"])
def delete_restaurant(restaurant_id):
    """Delete a restaurant from the list."""
    restaurants = load_restaurants()
    original_count = len(restaurants)
    restaurants = [r for r in restaurants if r["id"] != restaurant_id]
    if len(restaurants) == original_count:
        return jsonify({"error": "Restaurant not found"}), 404
    save_restaurants(restaurants)
    return jsonify({"message": "Restaurant deleted"}), 200


@app.route("/api/search", methods=["POST"])
def search_with_claude():
    """Use Claude AI to search for Indianapolis restaurants."""
    data = request.get_json()
    if not data or not data.get("query"):
        return jsonify({"error": "No search query provided"}), 400

    query = data["query"].strip()
    existing_restaurants = load_restaurants()
    existing_names = [r["name"] for r in existing_restaurants]

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_anthropic_api_key_here":
        return jsonify({"error": "Anthropic API key not configured. Please set ANTHROPIC_API_KEY in your .env file."}), 503

    client = anthropic.Anthropic(api_key=api_key)

    existing_list = "\n".join(f"- {name}" for name in existing_names) if existing_names else "None yet"
    system_prompt = (
        "You are a helpful restaurant recommendation assistant for a team in Indianapolis, Indiana. "
        "Your job is to suggest restaurants in the Indianapolis area that are great for team building events. "
        "Focus on restaurants that are good for groups, have a fun atmosphere, and are located in Indianapolis or its nearby suburbs. "
        "Always provide practical information including the restaurant name, cuisine type, approximate address, price range ($, $$, $$$, $$$$), and why it's great for team building. "
        "Format your response in a clear, friendly way."
    )

    user_message = (
        f"The team is looking for Indianapolis restaurants. Here is their search request:\n\n"
        f"\"{query}\"\n\n"
        f"Current restaurants already on the list:\n{existing_list}\n\n"
        f"Please suggest Indianapolis-area restaurants that match this request. "
        f"If any suggestions match restaurants already on the list, note that. "
        f"For each suggestion, provide: name, cuisine, address, price range, and why it's great for team building."
    )

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": user_message}],
        system=system_prompt,
    )

    return jsonify({"result": message.content[0].text, "query": query})


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug)
