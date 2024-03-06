from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score,f1_score
import xgboost as xgb
from xgboost import XGBClassifier
def RF_pred(X, y):
    
    # Build models with hyperparameters sets
    RSC = RandomizedSearchCV(
        estimator=RandomForestClassifier(),
        param_distributions={
            'n_estimators': range(650,850,50),
            'max_depth': range(8,14),
            'min_samples_split': range(2, 10),
            'max_features': ['auto', 'sqrt', 'log2']}, cv=5, scoring='roc_auc', n_jobs=-1)
    
    # Fit RandomizedSearchCV to find best hyperparameters
    search_result = RSC.fit(X, y)
    print("Best using: ", search_result.best_params_, "Score: ", search_result.best_score_,flush=True)

    # Build models with optimized hyperparameters
    model_RF = RandomForestClassifier(
        n_estimators=search_result.best_params_["n_estimators"],
        max_depth=search_result.best_params_["max_depth"],
        max_features=search_result.best_params_["max_features"])

    return model_RF


def XGB_pred(X, y):
    estimator = xgb.XGBClassifier(objective= 'binary:logistic',nthread=4,seed=123)
    parameters= { 'max_depth': range (5,11,4),'n_estimators': range(800, 1000, 50),'learning_rate': [0.1, 0.01, 0.001]}
    grid_search = RandomizedSearchCV(estimator=estimator,param_distributions=parameters,scoring='roc_auc',n_jobs = -1,cv = 5,verbose=True)
    search_result=grid_search.fit(X, y)
    print("Best using: ", search_result.best_params_, "Score: ", search_result.best_score_,flush=True)
    # Build models with optimized hyperparameters
    model_XGB = xgb.XGBClassifier(
        n_estimators=search_result.best_params_["n_estimators"],
        max_depth=search_result.best_params_["max_depth"],
        learning_rate=search_result.best_params_["learning_rate"])

    return model_XGB