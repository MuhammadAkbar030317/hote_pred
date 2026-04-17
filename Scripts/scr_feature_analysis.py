import pandas as pd
import sys
sys.path.append(r"C:\Users\User\Desktop\Hotel_pred\Scr")
from feature_analysis import FeatureSelector
PATH = r"C:\Users\User\Desktop\Hotel_pred\Data\Engineered"
df=pd.read_csv(r"C:\Users\User\Desktop\Hotel_pred\Data\Preprocessed\preprocessed.csv")
selection=FeatureSelector(df, "is_canceled", PATH)
selection.rf_selection()
selection.lasso_selection()



