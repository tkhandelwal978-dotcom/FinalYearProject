"""API tests against the real trained model (Upload -> preprocessing -> model -> DB -> response)."""

import pytest

from tests.conftest import image_bytes, requires_model, test_split_images


def upload(client, data: bytes, name="photo.jpg", content_type="image/jpeg", gradcam=True):
    return client.post(f"/api/predict?gradcam={str(gradcam).lower()}",
                       files={"file": (name, data, content_type)})


def test_health_reports_model_status(client):
    body = client.get("/api/health").json()
    assert body["database"] == "ok"
    assert isinstance(body["model_loaded"], bool)


@pytest.mark.parametrize("name, data, content_type, status", [
    ("notes.txt", b"hello", "text/plain", 415),
    ("anim.gif", image_bytes(fmt="GIF"), "image/gif", 415),
    ("fake.jpg", b"this is not really a jpeg", "image/jpeg", 400),
    ("empty.png", b"", "image/png", 400),
    ("bmp_in_disguise.png", image_bytes(fmt="BMP"), "image/png", 415),
    ("tiny.jpg", image_bytes(size=(10, 10)), "image/jpeg", 400),
])
@requires_model
def test_invalid_uploads_are_rejected(client, name, data, content_type, status):
    response = upload(client, data, name, content_type)
    assert response.status_code == status, response.text
    assert client.get("/api/history").json()["total"] == 0


@requires_model
def test_oversized_upload_is_rejected(client):
    big = b"\xff\xd8" + b"0" * (3 * 1024 * 1024)  # settings fixture caps uploads at 2 MB
    assert upload(client, big).status_code == 413


@requires_model
def test_predict_persists_and_serves_result(client):
    paths = test_split_images("fire", 1)
    data = paths[0].read_bytes() if paths else image_bytes()
    response = upload(client, data, "wildfire.jpg")
    assert response.status_code == 201, response.text
    result = response.json()

    assert result["predicted_class"] in {"fire", "non_fire"}
    assert abs(result["fire_probability"] + result["no_fire_probability"] - 1) < 1e-5
    expected_conf = result["fire_probability"] if result["predicted_class"] == "fire" else result["no_fire_probability"]
    assert result["confidence"] == pytest.approx(expected_conf)
    if paths:
        assert result["predicted_class"] == "fire"

    for key in ("image_url", "thumbnail_url", "overlay_url", "heatmap_url"):
        assert client.get(result[key]).status_code == 200
    stored = client.get(f"/api/history/{result['id']}").json()
    assert stored["fire_probability"] == result["fire_probability"]

    report = client.get(result["report_url"])
    assert report.status_code == 200 and report.content.startswith(b"%PDF")


@requires_model
def test_history_filter_search_pagination_and_delete(client):
    ids = [upload(client, image_bytes(color=(i * 40, 120, 200)), f"img_{i}.png", "image/png", gradcam=False).json()["id"]
           for i in range(3)]
    page = client.get("/api/history?page=1&page_size=2").json()
    assert page["total"] == 3 and len(page["items"]) == 2 and page["pages"] == 2

    assert client.get("/api/history?search=img_1").json()["total"] == 1
    for cls in ("fire", "non_fire"):
        items = client.get(f"/api/history?predicted_class={cls}&page_size=100").json()["items"]
        assert all(item["predicted_class"] == cls for item in items)
    assert client.get("/api/history?predicted_class=smoke").status_code == 422

    csv_text = client.get("/api/history/export").text
    assert csv_text.count("\n") == 4  # header + 3 rows

    assert client.delete(f"/api/history/{ids[0]}").status_code == 200
    assert client.get(f"/api/history/{ids[0]}").status_code == 404
    assert client.delete(f"/api/history/{ids[0]}").status_code == 404
    assert client.get("/api/history").json()["total"] == 2


@requires_model
def test_analytics_reflect_stored_predictions(client):
    empty = client.get("/api/analytics").json()
    assert empty["total_predictions"] == 0 and empty["fire_detection_rate"] == 0

    classes = []
    images = test_split_images("fire", 2) + test_split_images("non_fire", 2)
    payloads = [p.read_bytes() for p in images] or [image_bytes(), image_bytes(color=(20, 120, 40))]
    for data in payloads:
        classes.append(upload(client, data, gradcam=False).json()["predicted_class"])

    stats = client.get("/api/analytics?days=7").json()
    assert stats["total_predictions"] == len(classes)
    assert stats["fire_detections"] == classes.count("fire")
    assert stats["non_fire_detections"] == classes.count("non_fire")
    assert stats["fire_detection_rate"] == pytest.approx(classes.count("fire") / len(classes) * 100, abs=0.01)
    assert len(stats["daily_trend"]) == 7 and stats["daily_trend"][-1]["total"] == len(classes)
    assert sum(b["fire"] + b["non_fire"] for b in stats["confidence_distribution"]) == len(classes)


@requires_model
def test_model_info_reads_artifacts(client):
    info = client.get("/api/model-info").json()
    assert info["model_loaded"] is True
    assert info["architecture"].startswith("EfficientNet-B0")
    assert info["image_size"] == [224, 224]
    if info["test_metrics"]:
        assert 0 <= info["test_metrics"]["recall"] <= 1
