import os
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
from tqdm import tqdm

RESULTS_PATH = Path('results')
RESULTS_PATH.mkdir(exist_ok=True)

timestamp = pd.Timestamp.now().strftime('%Y_%m_%d_%H_%M')
result_file_path = str(RESULTS_PATH / f'results_{timestamp}.csv')

#TARGETS = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
TARGETS = ['Dirtiness']
train_df = pd.read_csv('data/train.csv', index_col=0)
test_df = pd.read_csv('data/test.csv', index_col=0)

lr_liblinear_param_space = {
    'model': [LogisticRegression(max_iter=10000, random_state=7)],
    'model__penalty': ['l1', 'l2'],
    'model__solver': Categorical(['liblinear']),
    'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
    'model__fit_intercept': Categorical([True, False]),
    'model__warm_start': Categorical([True, False]),
}

lr_l2_param_space = {
    'model': [LogisticRegression(max_iter=10000, random_state=7, class_weight='balanced')],
    'model__penalty': ['l2'],
    'model__solver': Categorical(['lbfgs', 'newton-cg', 'newton-cholesky', 'sag']),
    'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
    'model__fit_intercept': Categorical([True, False]),
    'model__warm_start': Categorical([True, False]),
}

lr_saga_param_space = {
    'model': [LogisticRegression(max_iter=10000, random_state=7)],
    'model__penalty': ['l1', 'l2'],
    'model__solver': Categorical(['saga']),
    'model__C': Real(1e-6, 1e+6, prior='log-uniform'),
    'model__fit_intercept': Categorical([True, False]),
    'model__warm_start': Categorical([True, False]),
}

rf_param_space = {
    'model': [RandomForestClassifier(random_state=7)],
    'model__n_estimators': Integer(50, 300),
    'model__max_features': Integer(3, 27),
    'model__max_depth': Integer(2, 20),
    'model__min_samples_split': Integer(2, 200),
    'model__min_samples_leaf': Integer(10, 200),
    'model__bootstrap': Categorical([True, False]),
    'model__criterion': Categorical(['gini', 'entropy']),
    #'model__warm_start': Categorical([True, False]),
    'model__class_weight': Categorical(['balanced', 'balanced_subsample']),
}

#Use SMOTE to oversample the minority class
oversample = SMOTE()

def run_experiments():
    target_df = train_df[TARGETS]
    X = train_df.drop(columns=TARGETS)

    for param_space, n_iter in [
        (lr_liblinear_param_space, 2),
        (lr_l2_param_space, 2),
        (lr_saga_param_space, 2),
        (rf_param_space, 50),
    ]:
        model_name = param_space['model'][0].__class__.__name__
        for target in tqdm(TARGETS, desc=model_name):
            y = target_df[target].values
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=0.2,
                random_state=7,
                stratify=y,
                shuffle=True
            )
            X_train_smote, y_train_smote = oversample.fit_resample(X_train, y_train)
            # print(f'\nbefore class counter: {dict(pd.Series(y_train).value_counts())}')
            # print(f' smote class counter: {dict(pd.Series(y_train_smote).value_counts())}')

            pipe = Pipeline(steps=[
                ('scaler', StandardScaler()),
                ('model', None)
            ])
            opt = BayesSearchCV(
                pipe,
                search_spaces=[(param_space, n_iter)],
                cv=3,
                return_train_score=True,
                n_jobs=-1,
                refit=True,
                random_state=7,
                scoring='f1',
                verbose=1,
                error_score=0,
            )
            # Dirty workaround for error - https://github.com/scikit-optimize/scikit-optimize/issues/1200#issuecomment-1915122563
            import numpy as np
            np.int = int
            opt.fit(X_train_smote, y_train_smote)

            cv_results = pd.DataFrame(opt.cv_results_)
            cv_results = cv_results.sort_values(by='rank_test_score').reset_index(drop=True)
            cv_results['model_name'] = model_name
            cv_results['target'] = target
            cv_results['n_iter'] = n_iter

            best_estimator = opt.best_estimator_
            y_pred = best_estimator.predict(X_val)
            y_score = best_estimator.predict_proba(X_val)[:, 1]

            cm = confusion_matrix(y_true=y_val, y_pred=y_pred)
            true_positives = cm[1, 1]
            false_negatives = cm[1, 0]
            false_positives = cm[0, 1]
            true_negatives = cm[0, 0]
            val_acc = accuracy_score(y_true=y_val, y_pred=y_pred)
            val_precision = precision_score(y_true=y_val, y_pred=y_pred)
            val_recall = recall_score(y_true=y_val, y_pred=y_pred)
            val_f1_score = f1_score(y_true=y_val, y_pred=y_pred)
            # Handle Value Error: Only one class present in y_true. ROC AUC score is not defined in that case.
            if len(set(y_val)) == 1:
                val_roc_auc_score = 0.0
            else:
                val_roc_auc_score = roc_auc_score(y_true=y_val, y_score=y_score, )

            scores = {
                'val_acc': val_acc,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1_score': val_f1_score,
                'val_roc_auc_score': val_roc_auc_score,
                'true_positives': true_positives,
                'false_negatives': false_negatives,
                'false_positives': false_positives,
                'true_negatives': true_negatives
            }

            for key in scores.keys():
                cv_results.loc[0, key] = scores.get(key)

            if not os.path.exists(result_file_path):
                # If the CSV file doesn't exist, save the DataFrame directly.
                cv_results.to_csv(result_file_path, index=False)
            else:
                # If the CSV exists, read it, merge with the new data, and save.
                existing_df = pd.read_csv(result_file_path)

                # Ensure all columns present, combining new and existing columns
                all_columns = sorted(set(existing_df.columns).union(cv_results.columns))

                # Reindex both DataFrames to have the same column order, adding missing columns with NaNs
                existing_df = existing_df.reindex(columns=all_columns)
                cv_results = cv_results.reindex(columns=all_columns)

                # Append the new results and save
                updated_df = pd.concat([existing_df, cv_results], ignore_index=True)
                updated_df.to_csv(result_file_path, index=False)


if __name__ == '__main__':
    run_experiments()
