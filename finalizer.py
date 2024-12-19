from train import train
from test import evaluate
from utils import load_config

config = load_config()

best_model, X_test, y_test = train(config)

evaluate(best_model, X_test, y_test, config)