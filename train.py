from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from utils import load_config, save_model_and_params
from data_pull_push import retrieve_data_from_azure

def func_train():
    # Load configurations
    config = load_config()

    # Read the dataset
    # try:
    data = retrieve_data_from_azure()
    # print('raw data successfully loaded', len(data))

    # except Exception as e:
    #     print(f"raw data CANNOT loaded due to {e}")

    # Define input and target columns
    X = data[[col for col in config['preprocessor']['categorical']['features']] + [col for col in config['preprocessor']['numerical']['features']]]
    y = data['Salary']

    # Preprocessing transformers
    categorical_transformer = OneHotEncoder()
    numerical_transformer = RobustScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, config['preprocessor']['categorical']['features']),
            ('num', numerical_transformer, config['preprocessor']['numerical']['features'])
        ]
    )

    # Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=config['data_split']['random_state']))
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['data_split']['test_size'], 
        random_state=config['data_split']['random_state']
    )

    # CSV olarak kaydetme
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

    # Grid search parameters
    param_grid = {
        'regressor__n_estimators': config['regressor']['params']['n_estimators'],
        'regressor__max_depth': config['regressor']['params']['max_depth'],
        'regressor__min_samples_split': config['regressor']['params']['min_samples_split'],
        'regressor__min_samples_leaf': config['regressor']['params']['min_samples_leaf']
    }

    # Grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=config['grid_search']['cv'],
        scoring=config['grid_search']['scoring'],
        verbose=config['grid_search']['verbose'],
        n_jobs=config['grid_search']['n_jobs']
    )

    grid_search.fit(X_train, y_train)

    # Save the best model
    best_model = grid_search.best_estimator_

    # To save model and best prams
    save_model_and_params(best_model, grid_search.best_params_, config['model_output'], "best_params.json")

    return best_model, X_test, y_test

