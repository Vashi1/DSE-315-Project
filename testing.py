import pandas as pd
from Data_loader import dummy_enc
from Data_imputation import knn_imputer
from GridSearch_params import split_data
from FeatureSelection import Kernal_PCA
import matplotlib.pyplot as plt
from XGBoost_classifier import xgb_kn_classifier_pca, xgb_kn_classifier_pca_kfold, xgb_kn_classifier_pca_stratfold
data = pd.read_csv('train.csv')
y = data.outcome
x = data.drop(['outcome'], axis=1)
x_dum = dummy_enc(x)
x_kn = knn_imputer(x_dum)
x_kn_pca = Kernal_PCA(x_kn)
y_dum = dummy_enc(y)
x_tr, x_te, y_tr, y_te = split_data(x_kn_pca, y_dum)


# xgb_kn_classifier_pca(x_tr, x_te, y_tr, y_te)
# xgb_kn_classifier_pca_kfold(x_kn_pca, y_dum, x_tr, x_te, y_tr, y_te)
xgb_kn_classifier_pca_stratfold(x_kn_pca, y_dum, x_tr, x_te, y_tr, y_te)

