from surprise import Dataset
from surprise import Reader
from surprise import SVD, SVDpp, NMF
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV

data = Dataset.load_builtin('ml-100k')


param_grid = {
    "n_factors": [50, 100],
    "n_epochs": [20, 30],
    "lr_all": [0.002, 0.005],
    "reg_all": [0.02, 0.1]
}

gs = GridSearchCV(
    SVD,
    param_grid,
    measures=['rmse'],
    cv=3,
    n_jobs=-1
)

gs.fit(data)

print("Найкращий RMSE для SVD:")
print(gs.best_score['rmse'])

print("\nНайкращі параметри:")
print(gs.best_params['rmse'])


best_svd = gs.best_estimator['rmse']

results_svd = cross_validate(
    best_svd,
    data,
    measures=['RMSE', 'MAE'],
    cv=5,
    verbose=True
)



svdpp = SVDpp()

results_svdpp = cross_validate(
    svdpp,
    data,
    measures=['RMSE', 'MAE'],
    cv=5,
    verbose=True
)



nmf = NMF()

results_nmf = cross_validate(
    nmf,
    data,
    measures=['RMSE', 'MAE'],
    cv=5,
    verbose=True
)


print("\nСередній RMSE:")

print("SVD:", results_svd['test_rmse'].mean())

print("SVD++:", results_svdpp['test_rmse'].mean())

print("NMF:", results_nmf['test_rmse'].mean())