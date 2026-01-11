import os
import sys
from sklearn.metrics import classification_report,roc_auc_score 
from sklearn.model_selection import GridSearchCV

import pickle

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            params = param[model_name]

            gs = GridSearchCV(
                estimator=model,
                param_grid=params,
                scoring="roc_auc",
                cv=3,
                n_jobs=-1,
                verbose=0
            )

            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_prob = best_model.predict_proba(X_test)[:, 1]
            y_pred = best_model.predict(X_test)

            roc_auc = roc_auc_score(y_test, y_prob)

            report[model_name] = {
                "roc_auc_score": roc_auc,
                "classification_report": classification_report(
                    y_test,
                    y_pred,
                    zero_division=0
                )
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)

def print_model_report(model_report: dict):
    print("\n" + "=" * 80)
    print("MODEL EVALUATION RESULTS")
    print("=" * 80)

    for model_name, metrics in model_report.items():
        print(f"\n🔹 Model: {model_name}")
        print("-" * 80)

        print(f"ROC-AUC Score: {metrics['roc_auc_score']:.4f}\n")

        print("Classification Report:")
        print(metrics["classification_report"])

    print("=" * 80)
