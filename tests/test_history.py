from src.engine.history import TrainingHistory


def test_training_history_add_epoch_and_to_dict():
    history = TrainingHistory()

    history.add_epoch(
        train_loss=0.5,
        train_accuracy=0.8,
        val_loss=0.4,
        val_accuracy=0.9,
        learning_rate=0.001,
    )

    assert history.to_dict() == {
        "train_loss": [0.5],
        "train_accuracy": [0.8],
        "val_loss": [0.4],
        "val_accuracy": [0.9],
        "learning_rate": [0.001],
    }
