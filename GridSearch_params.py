import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def split_data(data_red, labels):
    X_Train, X_Test, y_train, y_test = train_test_split(data_red, labels, test_size=0.2, random_state=42)
    return X_Train, X_Test, y_train, y_test


# Searching the Best set of parameters for Decision Tree
def grid_search_dt(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier(random_state=42)
    param_grid_dt = [
        {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20],
            'max_features': [None, 'sqrt', 'log2'],
            'ccp_alpha' : [0.0, 0.1, 0.01, 0.001, 0.2, 0.002]
        }
    ]
    gs_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, n_jobs=-1)
    gs_dt.fit(X_train, y_train)
    print("Optimal Decision Parameters: \n\n")
    print(gs_dt.best_params_)
    score = cross_val_score(gs_dt, X_train, y_train, scoring='accuracy', cv=5)
    print("Average Accuracy Score: %.4f +/- %.4f" % ((np.mean(score)), np.std(score)))

def grid_search_rft(X_train, X_test, y_train, y_test):
    rft = RandomForestClassifier(random_state=42)
    param_grid_rft = [
        {
            'n_estimators' : [100, 10, 200, 150, 250, 300],
            'max_depth' : [None, 2, 3, 4, 5, 6 ,7, 8, 10, 13, 15, 17, 19, 20],
            'max_features' : [None, 'sqrt', 'log2'],
            'class_weight' : [None, 'balanced', 'balanced_subsample'],
            'ccp_alpha' : [0, 0.1, 0.01, 0.001, 0.001, 0.0015]

        }
    ]
    gs_rft = GridSearchCV(rft, param_grid_rft, scoring='accuracy', cv = 5)
    gs_rft.fit(X_train, y_train)
    print("Optimal Random Forest Tree Parameters: \n\n")
    print(gs_rft.best_params_)
    score = cross_val_score(gs_rft, X_train, y_train, scoring='accuracy', cv=5)
    print("Average Accuracy Score: %.4f +/- %.4f" % ((np.mean(score)), np.std(score)))