import json
from fastapi.testclient import TestClient
import pytest
from app.main import create_app


PAYLOAD = {"queue_wait_sec": 120, "agent_exp_months": 24, "previous_calls": 3,
           "issue_count": 2, "hour": 14, "caller_age": 35,
           "language": "hindi", "team": "Team Alpha"}


class FakeModel:
    def __init__(self):
        self.calls = 0

    def predict(self, frame):
        self.calls += 1
        assert set(frame.columns) == set(PAYLOAD)
        return [12.5]


def test_health_prediction_and_private_json_logging(capsys):
    model = FakeModel()
    with TestClient(create_app(lambda: model)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = client.post("/predict", json=PAYLOAD)
        assert response.status_code == 200
        assert response.json() == {"prediction": 12.5}
    entries = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(entries) == 1
    assert entries[0]["event"] == "prediction"
    assert not set(PAYLOAD).intersection(entries[0])


@pytest.mark.parametrize("change", [{"previous_calls": 21}, {"queue_wait_sec": -1},
                                   {"duration_from_phone_bill": 10}, {"language": "unknown"}])
def test_invalid_request_is_rejected_before_model(change):
    model = FakeModel()
    with TestClient(create_app(lambda: model)) as client:
        response = client.post("/predict", json={**PAYLOAD, **change})
        assert response.status_code == 422
    assert model.calls == 0


def test_exact_client_boundary_values_are_allowed():
    model = FakeModel()
    with TestClient(create_app(lambda: model)) as client:
        assert client.post("/predict", json={**PAYLOAD, "previous_calls": 20,
                                             "queue_wait_sec": 0}).status_code == 200


def test_missing_field_is_rejected():
    request = PAYLOAD.copy()
    del request["caller_age"]
    model = FakeModel()
    with TestClient(create_app(lambda: model)) as client:
        assert client.post("/predict", json=request).status_code == 422
    assert model.calls == 0
