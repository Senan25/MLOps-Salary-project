from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate(best_model, X_test, y_test, config): 
    # Test the model
    y_pred = best_model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Print evaluation metrics
    print("Model Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R^2): {r2:.4f}")

    # Save evaluation metrics if needed
    evaluation_results = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

    # Save metrics to a file
    import json
    with open(f"{config['model_output']}_evaluation_results.json", "w") as f:
        json.dump(evaluation_results, f indent=4)

    print(f"Evaluation results saved to {config['model_output']}_evaluation_results.json")
