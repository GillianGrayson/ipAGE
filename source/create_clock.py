import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import copy


def calc_metrics(model, X, y, comment, params):
    y_pred = model.predict(X)
    score = model.score(X, y)
    rmse = np.sqrt(mean_squared_error(y_pred, y))
    mae = mean_absolute_error(y_pred, y)
    params[f'{comment} R2'] = score
    params[f'{comment} RMSE'] = rmse
    params[f'{comment} MAE'] = mae
    return y_pred


def process_clock():
    path = f'../data/'  # Path to the directory with data table
    table_name = f'all_features'  # Name of xlsx table with all features
    features_file = 'biomarkers'  # Name of file with features, which will use to build the clock
    y_col_name = 'Age'  # Column name of chronological age
    df_global = pd.read_excel(f'{path}/{table_name}.xlsx')

    with open(f'{path}/{features_file}.txt') as f:
        target_features = f.read().splitlines()

    # Selecting only controls
    X_C_df = df_global.loc[df_global['Group'] == 'Control']
    X_C = X_C_df[list(target_features)].to_numpy()
    y_C = X_C_df[y_col_name].to_numpy()

    # Selecting only subjects with ESRD
    X_T_df = df_global.loc[df_global['Group'] == 'ESRD']
    X_T = X_T_df[list(target_features)].to_numpy()
    y_T = X_T_df[y_col_name].to_numpy()

    # Complete dataset
    X_all = df_global[list(target_features)].to_numpy()
    y_all = df_global[y_col_name].to_numpy()

    # We build clocks only on controls
    X_target = X_C
    y_target = y_C

    # Cross-validation parameters
    scoring = 'r2'
    cv = RepeatedKFold(n_splits=3, n_repeats=5, random_state=1)
    model_type = ElasticNet(max_iter=10000, tol=0.01)

    # Define the grid of parameters
    alphas = np.logspace(-4, 1, 51)
    l1_ratios = [0.5]
    grid = dict()
    grid['alpha'] = alphas
    grid['l1_ratio'] = l1_ratios

    # Define the search
    search = GridSearchCV(estimator=model_type, scoring=scoring, param_grid=grid, cv=cv, verbose=3)
    results = search.fit(X_target, y_target)

    # Get the clock
    model = results.best_estimator_

    # Saving the information about the clock
    score = model.score(X_target, y_target)
    params = copy.deepcopy(results.best_params_)
    model_dict = {'feature': ['Intercept'], 'coef': [model.intercept_]}
    num_features = 0
    for f_id, f in enumerate(target_features):
        coef = model.coef_[f_id]
        if abs(coef) > 0:
            model_dict['feature'].append(f)
            model_dict['coef'].append(coef)
            num_features += 1
    model_df = pd.DataFrame(model_dict)
    model_df.to_excel(f'{path}/clock.xlsx', index=False)

    # Saving the result metrics of the clock
    calc_metrics(model, X_C, y_C, 'Control', params)
    calc_metrics(model, X_T, y_T, 'ESRD', params)
    calc_metrics(model, X_all, y_all, 'All', params)
    params['num_features'] = num_features
    params_df = pd.DataFrame({'Feature': list(params.keys()), 'Value': list(params.values())})
    params_df.to_excel(f'{path}/results.xlsx', index=False)


if __name__ == "__main__":
    process_clock()