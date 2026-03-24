import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Set up a temp data file before importing the app
_tmp_data_dir = tempfile.mkdtemp()
os.environ.setdefault("FLASK_SECRET_KEY", "test-secret")


@pytest.fixture()
def tmp_data_file(tmp_path):
    """Return a temporary restaurants data file path."""
    return tmp_path / "restaurants.json"


@pytest.fixture()
def app_with_tmp_data(tmp_data_file):
    """Create a Flask test client backed by a temporary data file."""
    import app as app_module

    original = app_module.DATA_FILE
    app_module.DATA_FILE = tmp_data_file
    app_module.app.config["TESTING"] = True

    with app_module.app.test_client() as client:
        yield client

    app_module.DATA_FILE = original


@pytest.fixture()
def seeded_client(app_with_tmp_data, tmp_data_file):
    """Test client pre-loaded with two restaurants."""
    sample = [
        {
            "id": 1,
            "name": "Test Bistro",
            "cuisine": "American",
            "address": "123 Main St, Indianapolis, IN",
            "price_range": "$$",
            "rating": 4.2,
            "notes": "Great for groups",
            "added_by": "admin",
        },
        {
            "id": 2,
            "name": "Indy Tacos",
            "cuisine": "Mexican",
            "address": "456 Broad Ripple Ave, Indianapolis, IN",
            "price_range": "$",
            "rating": 4.5,
            "notes": "Casual and fun",
            "added_by": "admin",
        },
    ]
    tmp_data_file.write_text(json.dumps(sample))
    return app_with_tmp_data


# ── GET /api/restaurants ──────────────────────────────────────────────────────

class TestGetRestaurants:
    def test_returns_empty_list_when_no_data_file(self, app_with_tmp_data):
        resp = app_with_tmp_data.get("/api/restaurants")
        assert resp.status_code == 200
        assert resp.get_json() == []

    def test_returns_all_restaurants(self, seeded_client):
        resp = seeded_client.get("/api/restaurants")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) == 2
        names = {r["name"] for r in data}
        assert names == {"Test Bistro", "Indy Tacos"}


# ── POST /api/restaurants ─────────────────────────────────────────────────────

class TestAddRestaurant:
    def test_adds_restaurant_successfully(self, app_with_tmp_data):
        payload = {
            "name": "New Place",
            "cuisine": "Italian",
            "address": "789 College Ave, Indianapolis, IN",
            "price_range": "$$$",
            "rating": 4.0,
            "notes": "Wood-fired pizza",
            "added_by": "alice",
        }
        resp = app_with_tmp_data.post(
            "/api/restaurants",
            json=payload,
            content_type="application/json",
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["name"] == "New Place"
        assert data["id"] == 1  # first entry

    def test_returns_400_when_name_missing(self, app_with_tmp_data):
        resp = app_with_tmp_data.post(
            "/api/restaurants",
            json={"cuisine": "Italian", "address": "123 St"},
        )
        assert resp.status_code == 400
        assert "Missing required fields" in resp.get_json()["error"]

    def test_returns_4xx_when_no_body(self, app_with_tmp_data):
        # Flask returns 415 when no content-type is set; 400 when body is empty JSON
        resp = app_with_tmp_data.post("/api/restaurants")
        assert 400 <= resp.status_code < 500

    def test_id_increments(self, seeded_client):
        resp = seeded_client.post(
            "/api/restaurants",
            json={"name": "Third", "cuisine": "Thai", "address": "999 St"},
        )
        assert resp.status_code == 201
        assert resp.get_json()["id"] == 3

    def test_defaults_added_by_to_team(self, app_with_tmp_data):
        resp = app_with_tmp_data.post(
            "/api/restaurants",
            json={"name": "Solo", "cuisine": "Sushi", "address": "1 St"},
        )
        assert resp.get_json()["added_by"] == "team"


# ── DELETE /api/restaurants/<id> ──────────────────────────────────────────────

class TestDeleteRestaurant:
    def test_deletes_existing_restaurant(self, seeded_client):
        resp = seeded_client.delete("/api/restaurants/1")
        assert resp.status_code == 200
        remaining = seeded_client.get("/api/restaurants").get_json()
        assert len(remaining) == 1
        assert remaining[0]["id"] == 2

    def test_returns_404_for_unknown_id(self, seeded_client):
        resp = seeded_client.delete("/api/restaurants/999")
        assert resp.status_code == 404


# ── POST /api/search ──────────────────────────────────────────────────────────

class TestSearch:
    def test_returns_400_with_no_query(self, app_with_tmp_data):
        resp = app_with_tmp_data.post("/api/search", json={})
        assert resp.status_code == 400

    def test_returns_503_when_api_key_not_configured(self, app_with_tmp_data):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "your_anthropic_api_key_here"}):
            resp = app_with_tmp_data.post("/api/search", json={"query": "Italian food"})
        assert resp.status_code == 503
        assert "API key" in resp.get_json()["error"]

    def test_returns_claude_response(self, seeded_client):
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Try Mama Carolla's on 54th Street!")]

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-key"}):
            with patch("anthropic.Anthropic") as mock_anthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = mock_message
                mock_anthropic.return_value = mock_client

                resp = seeded_client.post("/api/search", json={"query": "Italian in Indianapolis"})

        assert resp.status_code == 200
        data = resp.get_json()
        assert "Mama Carolla" in data["result"]
        assert data["query"] == "Italian in Indianapolis"

    def test_claude_receives_existing_restaurants_in_context(self, seeded_client):
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Some response")]

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-key"}):
            with patch("anthropic.Anthropic") as mock_anthropic:
                mock_client = MagicMock()
                mock_client.messages.create.return_value = mock_message
                mock_anthropic.return_value = mock_client

                seeded_client.post("/api/search", json={"query": "burgers"})

                call_args = mock_client.messages.create.call_args
                user_content = call_args[1]["messages"][0]["content"]
                # Existing restaurants should be included in the prompt
                assert "Test Bistro" in user_content
                assert "Indy Tacos" in user_content


# ── GET / (index page) ────────────────────────────────────────────────────────

class TestIndexPage:
    def test_renders_html(self, seeded_client):
        resp = seeded_client.get("/")
        assert resp.status_code == 200
        assert b"Indy Team Building Restaurants" in resp.data

    def test_shows_existing_restaurants(self, seeded_client):
        resp = seeded_client.get("/")
        assert b"Test Bistro" in resp.data
        assert b"Indy Tacos" in resp.data
