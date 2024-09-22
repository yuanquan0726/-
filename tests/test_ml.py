from src.train import Model


def test_train_model():
    model = Model()
    assert model.eval > 0.5
