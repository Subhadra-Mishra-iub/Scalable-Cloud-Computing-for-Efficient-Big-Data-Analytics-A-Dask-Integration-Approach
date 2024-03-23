import dask.array as da
from dask_ml.model_selection import train_test_split
import xgboost as xgb
import optuna
from optuna.integration import DaskOptunaPruner
from dask.distributed import Client, LocalCluster


cluster = LocalCluster() #creating a local cluster
client = Client(cluster)

# we will plug in our data here once we finish the pre-processing pipeline
X, y = make_regression(n_samples=100000, n_features=5, noise=0.1, random_state=42)

# Convert the NumPy arrays to Dask arrays
X_dask = da.from_array(X, chunks=10000)  # Chunk size can be adjusted based on available memory
y_dask = da.from_array(y, chunks=10000)

#splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_dask, y_dask, test_size=0.2)

def objective(trial):
    # XGBoost hyperparameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 100.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 100.0),
    }

    # Train XGBoost model with dask
    dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train)
    bst = xgb.dask.train(client=client, params=params, dtrain=dtrain, num_boost_round=100)

    # Make predictions
    y_pred = xgb.dask.predict(client=client, model=bst['booster'], data=X_test)

    # performance metric RMSE
    rmse = ((y_pred - y_test) ** 2).mean().compute() ** 0.5

    return rmse

#
study = optuna.create_study(pruner=DaskOptunaPruner(), direction='minimize')
study.optimize(objective, n_trials=100)

print("Best parameters:", study.best_params)
print("Best RMSE:", study.best_value)

# Close the Dask client and cluster
client.close()
cluster.close()
