from train import func_train
from test import evaluate
from utils import load_config

try:
    config = load_config()
    print('successfully loaded config')
except Exception as e:
    raise f"Cannot load config due to {e}"


best_model, X_test, y_test = func_train()

evaluate(best_model, X_test, y_test, config)