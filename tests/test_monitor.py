from src.monitor import determine_drift


def test_drift_is_assessed_for_small_but_valid_prediction_sets():
    drift_detected, drift_status, drift_score = determine_drift(
        observations=144,
        label_psi=0.0563,
        text_length_psi=0.0,
        edge_rate=0.0,
        thresholds={
            "text_length_psi": 0.20,
            "label_psi": 0.25,
            "edge_rate": 0.10,
        },
    )

    assert drift_detected is False
    assert drift_status == "stable"
    assert drift_score == 0
