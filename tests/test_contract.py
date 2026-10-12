from training import TrainConfig, capabilities, train_model


def test_errors_are_returned_not_raised():
    r = train_model("x", TrainConfig(modality="timeseries", task="classification", model="xgboost"))
    assert r.status == "error" and r.error
    assert r.to_dict()["modality"] == "timeseries"


def test_unsupported_combo_is_readable():
    r = train_model("x", TrainConfig(modality="video", task="regression"))
    assert r.status == "error" and "unsupported" in r.error


def test_capabilities_lists_registered_pieces():
    c = capabilities()
    assert "image" in c["modalities"]
    assert "timeseries/classification/xgboost" in c["models"]
