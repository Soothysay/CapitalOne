from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score,f1_score

def DT_pred(X, y):
    
    # Build models with hyperparameters sets
    RSC = RandomizedSearchCV(
        estimator=DecisionTreeClassifier(),
        param_distributions={
            'criterion': ['gini', 'entropy'],
            'max_depth': range(1, 300, 10),
            'max_features': ['auto', 'sqrt', 'log2']}, 
        cv=5, scoring='roc_auc', n_jobs=-1, verbose = True)
    
    # Fit RandomizedSearchCV to find best hyperparameters
    search_result = RSC.fit(X, y)
    print("Best using: ", search_result.best_params_, "Score: ", search_result.best_score_)

    # Build models with optimized hyperparameters
    model_DT = DecisionTreeClassifier(
        criterion=search_result.best_params_["criterion"],
        max_depth=search_result.best_params_["max_depth"],
        max_features=search_result.best_params_["max_features"])

    # Split dataset into 3 consecutive folds
    kf = KFold(n_splits=5, shuffle=True, random_state=None)
    
    i = 1
    for train, test in kf.split(X):  
        X_train = X.iloc[train]
        y_train = y.iloc[train]
        X_test = X.iloc[test]
        y_test = y.iloc[test]
        model_DT.fit(X_train, y_train)
        train_pred = model_DT.predict(X_train)
        y_pred = model_DT.predict(X_test)
        
        train_precision = precision_score(y_train, train_pred)
        train_recall = recall_score(y_train, train_pred)
        train_auc = roc_auc_score(y_train, train_pred)
        train_f1 = f1_score(y_train, train_pred)
        test_precision = precision_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred)
        test_auc = roc_auc_score(y_test, y_pred)        
        test_f1 = f1_score(y_test, y_pred)
        print('Fold '+ str(i), ':  Training precision: ', train_precision, 'Testing precision: ', test_precision,flush=True)
        print('Fold '+ str(i), ':  Training recall: ', train_recall, 'Testing accuracy: ', test_recall,flush=True)
        print('Fold '+ str(i), ':  Training auc: ', train_auc, 'Testing auc: ', test_auc,flush=True)
        print('Fold '+ str(i), ':  Training f1: ', train_f1, 'Testing f1: ', test_f1,flush=True)
        i += 1
        
    return model_DT