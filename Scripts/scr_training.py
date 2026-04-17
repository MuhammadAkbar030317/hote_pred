import pandas as pd
import sys 
sys.path.append(r"C:\Users\User\Desktop\Hotel_pred\Scr")
from training import Trainer
df=pd.read_csv(r"C:\Users\User\Desktop\Hotel_pred\Data\Preprocessed\preprocessed.csv")
PATH = r"C:\Users\User\Desktop\Hotel_pred\Results\threashold_result"
trainer = Trainer(df)
trainer.dt()
trainer.rf()
trainer.bagging()
trainer.knn()
trainer.xgboost()
trainer.lightgbm()
trainer.voting()
trainer.stacking()

trainer.save_results(PATH)
