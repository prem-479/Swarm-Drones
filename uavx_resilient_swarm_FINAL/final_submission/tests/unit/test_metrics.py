from uavx.autonomy.metrics import MissionMetrics, completion_rate


def test_completion_rate():
    assert completion_rate(8, 10) == 0.8


def test_metric_serialization(tmp_path):
    m = MissionMetrics(mission_completion_rate=0.8)
    path = tmp_path / "metrics.json"
    m.save(str(path))

    assert path.exists()
    assert "mission_completion_rate" in path.read_text()
