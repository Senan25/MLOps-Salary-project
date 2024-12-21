import yaml
import json
from joblib import dump
import os

def load_config(config_file="params.yaml"):
    """
    Load configuration from a YAML file.

    Parameters:
    config_file (str): Path to the configuration file. Default is 'params.yaml'.

    Returns:s
    dict: Configuration parameters as a dictionary.
    """
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"The configuration file {config_file} was not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing the YAML file: {e}")

# def save_model_and_params(model, params, model_file="serve/rf_model.pkl", params_file="best_params.json"):
#     """
#     Save a trained model as a .pkl file and its parameters as a .json file.

#     Parameters:
#     model: Trained model object to save.
#     params (dict): Dictionary of model parameters to save.
#     model_file (str): Path to save the model file. Default is 'model.pkl'.
#     params_file (str): Path to save the parameters file. Default is 'best_params.json'.
#     """
#     try:
#         # Save the model as a .pkl file
#         dump(model, model_file)
#         print(f"Model saved as '{model_file}'")

#         # Save the parameters as a .json file
#         with open(params_file, 'w') as file:
#             json.dump(params, file, indent=4)
#         print(f"Best parameters saved as '{params_file}'")
#     except Exception as e:
#         raise RuntimeError(f"An error occurred while saving the model or parameters: {e}")


def save_model_and_params(model, params, model_file="serve/rf_model.pkl", params_file="best_params.json"):
    """
    Save a trained model as a .pkl file and its parameters as a .json file.

    Parameters:
    model: Trained model object to save.
    params (dict): Dictionary of model parameters to save.
    model_file (str): Path to save the model file. Default is 'serve/rf_model.pkl'.
    params_file (str): Path to save the parameters file. Default is 'best_params.json'.
    """
    try:
        model_file = 'serve/'+model_file

        # Save the model as a .pkl file
        dump(model, model_file)
        print(f"Model successfully saved at: {model_file}")

        # Save the parameters as a .json file
        with open(params_file, 'w') as file:
            json.dump(params, file, indent=4)
        print(f"Parameters successfully saved at: {params_file}")

    except Exception as e:
        raise RuntimeError(f"An error occurred while saving the model or parameters: {e}")
    


    


