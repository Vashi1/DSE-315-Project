from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, cross_val_predict
def svm_kn_classifier_pca(x_tr, x_te, y_tr, y_te):
    svm_cl_kn_pca = SVC()
    print("Support Vector Classifier without using any Cross-Validation Technique and PCA Feature Selection")
    svm_cl_kn_pca.fit(x_tr, y_tr)
    ypr = svm_cl_kn_pca.predict(x_te)
    print("Classification Report")
    print(classification_report(y_te, ypr))
    print("Plotting the Confusion Matrix")
    cormat = confusion_matrix(y_te, ypr)
    sns.heatmap(cormat, annot=True, fmt = 'g')
    plt.savefig("Implementing Logistic Regression Classifier on Data with Dummy Encoding with KNN-Imputation and PCA Feature Selection.png")
    plt.show()


def svm_kn_classifier_pca_kfold(x_data, y_data,x_tr, x_te, y_tr, y_te):
    originalclass, predictedclass = [], []
    svm_cl_kn_pca = SVC()
    print("Support Vector Classifier with K-Fold Cross-Validation Technique(K = 10) and PCA Feature Extraction")
    outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)
    pre_scorer_macro = make_scorer(precision_score, average = 'macro')
    pre_scorer_weighted = make_scorer(precision_score, average = 'weighted')
    nested_score_precision_macro = cross_val_score(svm_cl_kn_pca, X=x_data, y=y_data, cv=outer_cv, scoring=pre_scorer_macro)
    nested_score_precision_weighted = cross_val_score(svm_cl_kn_pca, X=x_data, y=y_data, cv=outer_cv, scoring=pre_scorer_weighted)
    rec_score_macro = make_scorer(recall_score, average = 'macro')
    rec_score_weighted = make_scorer(recall_score, average = 'weighted')
    nested_score_recall_macro = cross_val_score(svm_cl_kn_pca, X=x_data, y=y_data, cv=outer_cv, scoring=rec_score_macro)
    nested_score_recall_weighted = cross_val_score(svm_cl_kn_pca, X=x_data, y=y_data, cv=outer_cv, scoring=rec_score_weighted)
    f1_scorer_macro = make_scorer(f1_score, average = 'micro')
    f1_scorer_weighted = make_scorer(f1_score, average = 'weighted')
    nested_score_f1_macro = cross_val_score(svm_cl_kn_pca, X=x_data, y=y_data, cv=outer_cv, scoring=f1_scorer_macro)
    nested_score_f1_weighted = cross_val_score(svm_cl_kn_pca, X=x_data, y=y_data, cv=outer_cv, scoring=f1_scorer_weighted)
    print("Classification Report")
    print(f"\tPrecision Score(Macro): {nested_score_precision_macro.mean():.2f}", )
    print(f"\tPrecision Score(Weighted): {nested_score_precision_weighted.mean():.2f}", )
    print()
    print(f"\tRecall Score(Macro): {nested_score_recall_macro.mean():.2f}", )
    print(f"\tRecall Score(Weighted): {nested_score_recall_weighted.mean():.2f}", )
    print()
    print(f"\tF1 Score(Macro): {nested_score_f1_macro.mean():.2f}", )
    print(f"\tF1 Score(Weighted): {nested_score_f1_weighted.mean():.2f}", )
    acc_scorer = make_scorer(accuracy_score)
    nested_score_acc = cross_val_score(svm_cl_kn_pca, X=x_data, y=y_data, cv=outer_cv, scoring=acc_scorer)
    print(f"\tAccuracy Score: {nested_score_acc.mean():.2f}")
    print("Plotting the Confusion Matrix")
    y_pr = cross_val_predict(svm_cl_kn_pca, x_te, y_te, cv = outer_cv)
    cormat = confusion_matrix(y_te, y_pr)
    sns.heatmap(cormat, annot=True, fmt='g')
    plt.savefig("Implementing Support Vector Classifier on Data with Dummy Encoding with KNN-Imputation and PCA Feature Selection with K-Fold Cross Validation.png")
    plt.show()


def svm_kn_classifier_pca_stratfold(x_data, y_data, x_tr, x_te, y_tr, y_te):
    originalclass, predictedclass = [], []
    lr_cl_kn_pca_stratf = SVC()
    print("Support Vector Classifier with Stratified K-Fold Cross-Validation Technique and PCA Feature Extraction Technique")
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    pre_scorer_macro = make_scorer(precision_score, average='macro')
    pre_scorer_weighted = make_scorer(precision_score, average='weighted')
    nested_score_precision_macro = cross_val_score(lr_cl_kn_pca_stratf, X=x_data, y=y_data, cv=outer_cv,
                                                   scoring=pre_scorer_macro)
    nested_score_precision_weighted = cross_val_score(lr_cl_kn_pca_stratf, X=x_data, y=y_data, cv=outer_cv,
                                                      scoring=pre_scorer_weighted)
    rec_score_macro = make_scorer(recall_score, average='macro')
    rec_score_weighted = make_scorer(recall_score, average='weighted')
    nested_score_recall_macro = cross_val_score(lr_cl_kn_pca_stratf, X=x_data, y=y_data, cv=outer_cv, scoring=rec_score_macro)
    nested_score_recall_weighted = cross_val_score(lr_cl_kn_pca_stratf, X=x_data, y=y_data, cv=outer_cv,
                                                   scoring=rec_score_weighted)
    f1_scorer_macro = make_scorer(f1_score, average='micro')
    f1_scorer_weighted = make_scorer(f1_score, average='weighted')
    nested_score_f1_macro = cross_val_score(lr_cl_kn_pca_stratf, X=x_data, y=y_data, cv=outer_cv, scoring=f1_scorer_macro)
    nested_score_f1_weighted = cross_val_score(lr_cl_kn_pca_stratf, X=x_data, y=y_data, cv=outer_cv,
                                               scoring=f1_scorer_weighted)
    print("Classification Report")
    print(f"\tPrecision Score(Macro): {nested_score_precision_macro.mean():.2f}", )
    print(f"\tPrecision Score(Weighted): {nested_score_precision_weighted.mean():.2f}", )
    print()
    print(f"\tRecall Score(Macro): {nested_score_recall_macro.mean():.2f}", )
    print(f"\tRecall Score(Weighted): {nested_score_recall_weighted.mean():.2f}", )
    print()
    print(f"\tF1 Score(Macro): {nested_score_f1_macro.mean():.2f}", )
    print(f"\tF1 Score(Weighted): {nested_score_f1_weighted.mean():.2f}", )
    acc_scorer = make_scorer(accuracy_score)
    nested_score_acc = cross_val_score(lr_cl_kn_pca_stratf, X=x_data, y=y_data, cv=outer_cv, scoring=acc_scorer)
    print(f"\tAccuracy Score: {nested_score_acc.mean():.2f}")
    print("Plotting the Confusion Matrix")
    y_pr = cross_val_predict(lr_cl_kn_pca_stratf, x_te, y_te, cv = outer_cv)
    cormat = confusion_matrix(y_te, y_pr)
    sns.heatmap(cormat, annot=True, fmt = 'g')
    plt.savefig("Implementing Support Vector Classifier on Data with Dummy Encoding with KNN-Imputation and PCA Feature Selection with Stratified K-Fold Cross Validation.png")
    plt.show()


def svm_kn_classifier_kpca(x_tr, x_te, y_tr, y_te):
    svm_cl_kn_pca = SVC()
    print("Support Vector Classifier without using any Cross-Validation Technique and Kernel PCA Feature Selection")
    svm_cl_kn_pca.fit(x_tr, y_tr)
    ypr = svm_cl_kn_pca.predict(x_te)
    print("Classification Report")
    print(classification_report(y_te, ypr))
    print("Plotting the Confusion Matrix")
    cormat = confusion_matrix(y_te, ypr)
    sns.heatmap(cormat, annot=True, fmt='g')
    plt.savefig(
        "Implementing Logistic Regression Classifier on Data with Dummy Encoding with KNN-Imputation and Kernel PCA Feature Selection.png")
    plt.show()


def svm_kn_classifier_kpca_kfold(x_data, y_data, x_tr, x_te, y_tr, y_te):
    originalclass, predictedclass = [], []
    svm_cl_kn_pca = SVC()
    print("Support Vector Classifier with K-Fold Cross-Validation Technique(K = 10) and Kernel PCA Feature Extraction")
    outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)
    pre_scorer_macro = make_scorer(precision_score, average='macro')
    pre_scorer_weighted = make_scorer(precision_score, average='weighted')
    nested_score_precision_macro = cross_val_score(svm_cl_kn_pca, X=x_data, y=y_data, cv=outer_cv,
                                                   scoring=pre_scorer_macro)
    nested_score_precision_weighted = cross_val_score(svm_cl_kn_pca, X=x_data, y=y_data, cv=outer_cv,
                                                      scoring=pre_scorer_weighted)
    rec_score_macro = make_scorer(recall_score, average='macro')
    rec_score_weighted = make_scorer(recall_score, average='weighted')
    nested_score_recall_macro = cross_val_score(svm_cl_kn_pca, X=x_data, y=y_data, cv=outer_cv, scoring=rec_score_macro)
    nested_score_recall_weighted = cross_val_score(svm_cl_kn_pca, X=x_data, y=y_data, cv=outer_cv,
                                                   scoring=rec_score_weighted)
    f1_scorer_macro = make_scorer(f1_score, average='micro')
    f1_scorer_weighted = make_scorer(f1_score, average='weighted')
    nested_score_f1_macro = cross_val_score(svm_cl_kn_pca, X=x_data, y=y_data, cv=outer_cv, scoring=f1_scorer_macro)
    nested_score_f1_weighted = cross_val_score(svm_cl_kn_pca, X=x_data, y=y_data, cv=outer_cv,
                                               scoring=f1_scorer_weighted)
    print("Classification Report")
    print(f"\tPrecision Score(Macro): {nested_score_precision_macro.mean():.2f}", )
    print(f"\tPrecision Score(Weighted): {nested_score_precision_weighted.mean():.2f}", )
    print()
    print(f"\tRecall Score(Macro): {nested_score_recall_macro.mean():.2f}", )
    print(f"\tRecall Score(Weighted): {nested_score_recall_weighted.mean():.2f}", )
    print()
    print(f"\tF1 Score(Macro): {nested_score_f1_macro.mean():.2f}", )
    print(f"\tF1 Score(Weighted): {nested_score_f1_weighted.mean():.2f}", )
    acc_scorer = make_scorer(accuracy_score)
    nested_score_acc = cross_val_score(svm_cl_kn_pca, X=x_data, y=y_data, cv=outer_cv, scoring=acc_scorer)
    print(f"\tAccuracy Score: {nested_score_acc.mean():.2f}")
    print("Plotting the Confusion Matrix")
    y_pr = cross_val_predict(svm_cl_kn_pca, x_te, y_te, cv = outer_cv)
    cormat = confusion_matrix(y_te, y_pr)
    sns.heatmap(cormat, annot=True, fmt='g')
    plt.savefig(
        "Implementing Support Vector Classifier on Data with Dummy Encoding with KNN-Imputation and Kernel PCA Feature Selection with K-Fold Cross Validation.png")
    plt.show()


def svm_kn_classifier_kpca_stratfold(x_data, y_data, x_tr, x_te, y_tr, y_te):
    originalclass, predictedclass = [], []
    lr_cl_kn_pca_stratf = SVC()
    print(
        "Support Vector Classifier with Stratified K-Fold Cross-Validation Technique and Kernel PCA Feature Extraction Technique")
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    pre_scorer_macro = make_scorer(precision_score, average='macro')
    pre_scorer_weighted = make_scorer(precision_score, average='weighted')
    nested_score_precision_macro = cross_val_score(lr_cl_kn_pca_stratf, X=x_data, y=y_data, cv=outer_cv,
                                                   scoring=pre_scorer_macro)
    nested_score_precision_weighted = cross_val_score(lr_cl_kn_pca_stratf, X=x_data, y=y_data, cv=outer_cv,
                                                      scoring=pre_scorer_weighted)
    rec_score_macro = make_scorer(recall_score, average='macro')
    rec_score_weighted = make_scorer(recall_score, average='weighted')
    nested_score_recall_macro = cross_val_score(lr_cl_kn_pca_stratf, X=x_data, y=y_data, cv=outer_cv,
                                                scoring=rec_score_macro)
    nested_score_recall_weighted = cross_val_score(lr_cl_kn_pca_stratf, X=x_data, y=y_data, cv=outer_cv,
                                                   scoring=rec_score_weighted)
    f1_scorer_macro = make_scorer(f1_score, average='micro')
    f1_scorer_weighted = make_scorer(f1_score, average='weighted')
    nested_score_f1_macro = cross_val_score(lr_cl_kn_pca_stratf, X=x_data, y=y_data, cv=outer_cv,
                                            scoring=f1_scorer_macro)
    nested_score_f1_weighted = cross_val_score(lr_cl_kn_pca_stratf, X=x_data, y=y_data, cv=outer_cv,
                                               scoring=f1_scorer_weighted)
    print("Classification Report")
    print(f"\tPrecision Score(Macro): {nested_score_precision_macro.mean():.2f}", )
    print(f"\tPrecision Score(Weighted): {nested_score_precision_weighted.mean():.2f}", )
    print()
    print(f"\tRecall Score(Macro): {nested_score_recall_macro.mean():.2f}", )
    print(f"\tRecall Score(Weighted): {nested_score_recall_weighted.mean():.2f}", )
    print()
    print(f"\tF1 Score(Macro): {nested_score_f1_macro.mean():.2f}", )
    print(f"\tF1 Score(Weighted): {nested_score_f1_weighted.mean():.2f}", )
    acc_scorer = make_scorer(accuracy_score)
    nested_score_acc = cross_val_score(lr_cl_kn_pca_stratf, X=x_data, y=y_data, cv=outer_cv, scoring=acc_scorer)
    print(f"\tAccuracy Score: {nested_score_acc.mean():.2f}")
    print("Plotting the Confusion Matrix")
    y_pr = cross_val_predict(lr_cl_kn_pca_stratf, x_te, y_te, cv = outer_cv)
    cormat = confusion_matrix(y_te, y_pr)
    sns.heatmap(cormat, annot=True, fmt='g')
    plt.savefig(
        "Implementing Support Vector Classifier on Data with Dummy Encoding with KNN-Imputation and Kernel PCA Feature Selection with Stratified K-Fold Cross Validation.png")
    plt.show()


def svm_kn_classifier_trsvd(x_tr, x_te, y_tr, y_te):
    svm_cl_kn_pca = SVC()
    print("Support Vector Classifier without using any Cross-Validation Technique and Truncated SVD Feature Selection")
    svm_cl_kn_pca.fit(x_tr, y_tr)
    ypr = svm_cl_kn_pca.predict(x_te)
    print("Classification Report")
    print(classification_report(y_te, ypr))
    print("Plotting the Confusion Matrix")
    cormat = confusion_matrix(y_te, ypr)
    sns.heatmap(cormat, annot=True, fmt='g')
    plt.savefig(
        "Implementing Support Vector Classifier on Data with Dummy Encoding with KNN-Imputation and Truncated SVD Feature Selection.png")
    plt.show()


def svm_kn_classifier_trsvd_kfold(x_data, y_data, x_tr, x_te, y_tr, y_te):
    originalclass, predictedclass = [], []
    svm_cl_kn_pca = SVC()
    print("Support Vector Classifier with K-Fold Cross-Validation Technique(K = 10) and Truncated SVD Feature Extraction")
    outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)
    pre_scorer_macro = make_scorer(precision_score, average='macro')
    pre_scorer_weighted = make_scorer(precision_score, average='weighted')
    nested_score_precision_macro = cross_val_score(svm_cl_kn_pca, X=x_data, y=y_data, cv=outer_cv,
                                                   scoring=pre_scorer_macro)
    nested_score_precision_weighted = cross_val_score(svm_cl_kn_pca, X=x_data, y=y_data, cv=outer_cv,
                                                      scoring=pre_scorer_weighted)
    rec_score_macro = make_scorer(recall_score, average='macro')
    rec_score_weighted = make_scorer(recall_score, average='weighted')
    nested_score_recall_macro = cross_val_score(svm_cl_kn_pca, X=x_data, y=y_data, cv=outer_cv, scoring=rec_score_macro)
    nested_score_recall_weighted = cross_val_score(svm_cl_kn_pca, X=x_data, y=y_data, cv=outer_cv,
                                                   scoring=rec_score_weighted)
    f1_scorer_macro = make_scorer(f1_score, average='micro')
    f1_scorer_weighted = make_scorer(f1_score, average='weighted')
    nested_score_f1_macro = cross_val_score(svm_cl_kn_pca, X=x_data, y=y_data, cv=outer_cv, scoring=f1_scorer_macro)
    nested_score_f1_weighted = cross_val_score(svm_cl_kn_pca, X=x_data, y=y_data, cv=outer_cv,
                                               scoring=f1_scorer_weighted)
    print("Classification Report")
    print(f"\tPrecision Score(Macro): {nested_score_precision_macro.mean():.2f}", )
    print(f"\tPrecision Score(Weighted): {nested_score_precision_weighted.mean():.2f}", )
    print()
    print(f"\tRecall Score(Macro): {nested_score_recall_macro.mean():.2f}", )
    print(f"\tRecall Score(Weighted): {nested_score_recall_weighted.mean():.2f}", )
    print()
    print(f"\tF1 Score(Macro): {nested_score_f1_macro.mean():.2f}", )
    print(f"\tF1 Score(Weighted): {nested_score_f1_weighted.mean():.2f}", )
    acc_scorer = make_scorer(accuracy_score)
    nested_score_acc = cross_val_score(svm_cl_kn_pca, X=x_data, y=y_data, cv=outer_cv, scoring=acc_scorer)
    print(f"\tAccuracy Score: {nested_score_acc.mean():.2f}")
    print("Plotting the Confusion Matrix")
    y_pr = cross_val_predict(svm_cl_kn_pca, x_te, y_te, cv = outer_cv)
    cormat = confusion_matrix(y_te, y_pr)
    sns.heatmap(cormat, annot=True, fmt='g')
    plt.savefig(
        "Implementing Support Vector Classifier on Data with Dummy Encoding with KNN-Imputation and Truncated SVD Feature Selection with K-Fold Cross Validation.png")
    plt.show()


def svm_kn_classifier_trsvd_stratfold(x_data, y_data, x_tr, x_te, y_tr, y_te):
    originalclass, predictedclass = [], []
    lr_cl_kn_pca_stratf = SVC()
    print(
        "Support Vector Classifier with Stratified K-Fold Cross-Validation Technique and Truncated SVD Feature Extraction Technique")
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    pre_scorer_macro = make_scorer(precision_score, average='macro')
    pre_scorer_weighted = make_scorer(precision_score, average='weighted')
    nested_score_precision_macro = cross_val_score(lr_cl_kn_pca_stratf, X=x_data, y=y_data, cv=outer_cv,
                                                   scoring=pre_scorer_macro)
    nested_score_precision_weighted = cross_val_score(lr_cl_kn_pca_stratf, X=x_data, y=y_data, cv=outer_cv,
                                                      scoring=pre_scorer_weighted)
    rec_score_macro = make_scorer(recall_score, average='macro')
    rec_score_weighted = make_scorer(recall_score, average='weighted')
    nested_score_recall_macro = cross_val_score(lr_cl_kn_pca_stratf, X=x_data, y=y_data, cv=outer_cv,
                                                scoring=rec_score_macro)
    nested_score_recall_weighted = cross_val_score(lr_cl_kn_pca_stratf, X=x_data, y=y_data, cv=outer_cv,
                                                   scoring=rec_score_weighted)
    f1_scorer_macro = make_scorer(f1_score, average='micro')
    f1_scorer_weighted = make_scorer(f1_score, average='weighted')
    nested_score_f1_macro = cross_val_score(lr_cl_kn_pca_stratf, X=x_data, y=y_data, cv=outer_cv,
                                            scoring=f1_scorer_macro)
    nested_score_f1_weighted = cross_val_score(lr_cl_kn_pca_stratf, X=x_data, y=y_data, cv=outer_cv,
                                               scoring=f1_scorer_weighted)
    print("Classification Report")
    print(f"\tPrecision Score(Macro): {nested_score_precision_macro.mean():.2f}", )
    print(f"\tPrecision Score(Weighted): {nested_score_precision_weighted.mean():.2f}", )
    print()
    print(f"\tRecall Score(Macro): {nested_score_recall_macro.mean():.2f}", )
    print(f"\tRecall Score(Weighted): {nested_score_recall_weighted.mean():.2f}", )
    print()
    print(f"\tF1 Score(Macro): {nested_score_f1_macro.mean():.2f}", )
    print(f"\tF1 Score(Weighted): {nested_score_f1_weighted.mean():.2f}", )
    acc_scorer = make_scorer(accuracy_score)
    nested_score_acc = cross_val_score(lr_cl_kn_pca_stratf, X=x_data, y=y_data, cv=outer_cv, scoring=acc_scorer)
    print(f"\tAccuracy Score: {nested_score_acc.mean():.2f}")
    print("Plotting the Confusion Matrix")
    y_pr = cross_val_predict(lr_cl_kn_pca_stratf, x_te, y_te, cv = outer_cv)
    cormat = confusion_matrix(y_te, y_pr)
    sns.heatmap(cormat, annot=True, fmt='g')
    plt.savefig(
        "Implementing Support Vector Classifier on Data with Dummy Encoding with KNN-Imputation and Truncated SVD Feature Selection with Stratified K-Fold Cross Validation.png")
    plt.show()