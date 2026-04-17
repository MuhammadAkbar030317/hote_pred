import os
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,BaggingClassifier,VotingClassifier,StackingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class Trainer:
    def __init__(self, df, target_col="is_canceled", test_size=0.2, random_state=42):
        self.df = df
        self.target_col = target_col
        self.results = []

        X = df.drop(target_col, axis=1)
        y = df[target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    def _evaluate(self, name, model):
        model.fit(self.X_train, self.y_train)
        y_pred = (model.predict_proba(self.X_test)[:,1]>0.5).astype(int)
        self.results.append([
            name,
            accuracy_score(self.y_test, y_pred),
            precision_score(self.y_test, y_pred),
            recall_score(self.y_test, y_pred),
            f1_score(self.y_test, y_pred)
        ])

    def dt(self):
        self._evaluate("DecisionTree", DecisionTreeClassifier(random_state=42))

    def rf(self):
        self._evaluate("RandomForest", RandomForestClassifier(n_estimators=300, random_state=42))

    def bagging(self):
        self._evaluate("Bagging",BaggingClassifier(estimator=DecisionTreeClassifier(),n_estimators=200,random_state=42))

    def knn(self):
        self._evaluate("KNN", KNeighborsClassifier(n_neighbors=7))

    def xgboost(self):
        self._evaluate("XGBoost",XGBClassifier(eval_metric="logloss",random_state=42))

    def lightgbm(self):
        self._evaluate("LightGBM",LGBMClassifier(random_state=42))

    def voting(self):
        self._evaluate("Voting",VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(max_iter=1000)),
                    ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
                    ("knn", KNeighborsClassifier())
                ],
                voting="soft"
            )
        )

    def stacking(self):
        self._evaluate(
            "Stacking",
            StackingClassifier(
                estimators=[
                    ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
                    ("knn", KNeighborsClassifier())
                ],
                final_estimator=LogisticRegression(max_iter=1000)
            )
        )


    def save_results(self, path):
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, "results.txt")

        table = tabulate(
            self.results,
            headers=["Model", "Accuracy", "Precision", "Recall", "F1"],
            tablefmt="grid",
            floatfmt=".4f"
        )

        with open(file_path, "a", encoding="utf-8") as f:
            f.write("\n\n" + table)

        print(f"Natijalar saqlandi → {file_path}")
