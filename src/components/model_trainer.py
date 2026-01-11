import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=3000,
                    solver="lbfgs",
                    class_weight="balanced"
                ),
                "Random Forest": RandomForestClassifier(random_state=42),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(random_state=42),
                "AdaBoost": AdaBoostClassifier(random_state=42),
                "XgBoost": XGBClassifier(
                    eval_metric="logloss",
                    random_state=42
                )
            }

            params = {
                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10]
                },

                "Random Forest": {
                    "n_estimators": [50, 100, 200]
                },

                "Decision Tree": {
                    "max_depth": [None, 5, 10, 20]
                },

                "Gradient Boosting": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [50, 100]
                },

                "AdaBoost": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [50, 100]
                },

                "XgBoost": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [50, 100],
                    "max_depth": [3, 5]
                }
            }

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            best_model_name = max(
                model_report,
                key=lambda k: model_report[k]["roc_auc_score"]
            )
            best_model_score = model_report[best_model_name]["roc_auc_score"]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found")

            logging.info(f"Best model: {best_model_name} with ROC-AUC: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            return model_report

        except Exception as e:
            raise CustomException(e, sys)
