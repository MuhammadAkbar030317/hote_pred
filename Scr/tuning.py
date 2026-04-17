import os
import optuna
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

class Tuning:
    def __init__(self, df, target_col="is_canceled", test_size=0.2, random_state=42):
        self.df = df
        self.target_col = target_col
        self.results = []
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y )

    def _run_optuna(self, model_name, model_fn, n_trials=10):

        def objective(trial):
            model = model_fn(trial)
            model.fit(self.X_train, self.y_train)
            preds = model.predict(self.X_test)
            return accuracy_score(self.y_test, preds)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_model = model_fn(study.best_trial)
        best_model.fit(self.X_train, self.y_train)
        y_pred = best_model.predict(self.X_test)

        self.results.append([
            model_name,
            accuracy_score(self.y_test, y_pred),
            precision_score(self.y_test, y_pred),
            recall_score(self.y_test, y_pred),
            f1_score(self.y_test, y_pred)
        ])

    def decision_tree(self):
        self._run_optuna(
            "DecisionTree",
            lambda t: DecisionTreeClassifier(
                max_depth=t.suggest_int("max_depth", 3, 30),
                min_samples_split=t.suggest_int("min_samples_split", 2, 10),
                random_state=42
            )
        )

    def random_forest(self):
        self._run_optuna(
            "RandomForest",
            lambda t: RandomForestClassifier(
                n_estimators=t.suggest_int("n_estimators", 100, 400),
                max_depth=t.suggest_int("max_depth", 5, 30),
                max_features=t.suggest_categorical("max_features", ["sqrt", "log2"]),
                random_state=42,
                n_jobs=-1
            )
        )

    def bagging(self):
        self._run_optuna(
            "Bagging",
            lambda t: BaggingClassifier(
                n_estimators=t.suggest_int("n_estimators", 50, 300),
                random_state=42
            )
        )

    def knn(self):
        self._run_optuna(
            "KNN",
            lambda t: KNeighborsClassifier(
                n_neighbors=t.suggest_int("n_neighbors", 3, 15),
                weights=t.suggest_categorical("weights", ["uniform", "distance"])
            )
        )

    def xgboost(self):
        self._run_optuna(
            "XGBoost",
            lambda t: XGBClassifier(
                n_estimators=t.suggest_int("n_estimators", 100, 400),
                max_depth=t.suggest_int("max_depth", 3, 10),
                learning_rate=t.suggest_float("learning_rate", 0.01, 0.3),
                subsample=t.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
                eval_metric="logloss",
                random_state=42
            )
        )

    def lightgbm(self):
        self._run_optuna(
            "LightGBM",
            lambda t: LGBMClassifier(
                n_estimators=t.suggest_int("n_estimators", 100, 400),
                num_leaves=t.suggest_int("num_leaves", 20, 100),
                learning_rate=t.suggest_float("learning_rate", 0.01, 0.3),
                random_state=42
            )
        )

    def save_results(self, path):
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, "results.txt")

        table = tabulate(
            self.results,
            headers=["Model", "Accuracy", "Precision", "Recall", "F1"],
            tablefmt="grid",
            floatfmt=".2f"
        )

        with open(file_path, "a", encoding="utf-8") as f:
            f.write("\n\n" + table)

        print(f" Natijalar saqlandi → {file_path}")
