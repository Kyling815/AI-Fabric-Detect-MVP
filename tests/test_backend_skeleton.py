from backend.main import app


def test_app_title() -> None:
    assert app.title == "AI Fabric Detect MVP"


def test_core_routes_registered() -> None:
    paths = {route.path for route in app.routes}
    assert "/api/health" in paths
    assert "/api/datasets/discover" in paths
    assert "/api/datasets/paths" in paths
    assert "/api/inspect" in paths
    assert "/api/reference/build" in paths
    assert "/api/reference/status" in paths
    assert "/api/reference/list" in paths
    assert "/api/reference/status/{reference_id}" in paths
    assert "/api/model-info" in paths
