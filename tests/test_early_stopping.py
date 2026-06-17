from src.engine.early_stopping import EarlyStopping


def test_early_stopping_does_not_stop_while_improving():
    early_stopping = EarlyStopping(
        patience=2,
        min_delta=0.1,
    )

    assert early_stopping.should_stop(1.0) is False
    assert early_stopping.should_stop(0.8) is False
    assert early_stopping.should_stop(0.6) is False


def test_early_stopping_stops_after_patience_is_exceeded():
    early_stopping = EarlyStopping(patience=2)

    assert early_stopping.should_stop(1.0) is False
    assert early_stopping.should_stop(1.1) is False
    assert early_stopping.should_stop(1.2) is True
