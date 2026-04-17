import pandas as pd
import sys 
sys.path.append(r"C:\Users\User\Desktop\Hotel_pred\Scr")
from tuning import Tuning

df = pd.read_csv(r"C:\Users\User\Desktop\Hotel_pred\Data\Preprocessed\preprocessed.csv")

trainer = Tuning(df, target_col="is_canceled")

trainer.decision_tree()
trainer.random_forest()
trainer.knn()
trainer.bagging()
trainer.xgboost()
trainer.lightgbm()

trainer.save_results(r"C:\Users\User\Desktop\Hotel_pred\Results\preprocessed_tuning")