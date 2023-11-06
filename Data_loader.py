import pandas as pd
import numpy as np

# Loading the data
train_data = pd.read_csv("train.csv")

# Converting Categorical Data into Numerical Data through one-hot-encoding
from sklearn.preprocessing import OneHotEncoder


def onehot_encode(data_df):
    #Data Imputation Not Needed
    ohc = OneHotEncoder(sparse_output=False)
    data_enc = ohc.fit_transform(data_df)
    return data_enc


# Dummy Encoding
def dummy_enc(data_df):
    enc = pd.get_dummies(data_df,
                         columns=['surgery', 'age', 'temp_of_extremities', 'peripheral_pulse', 'mucous_membrane',
                                  'capillary_refill_time', 'pain', 'peristalsis', 'abdominal_distention',
                                  'nasogastric_tube', 'nasogastric_reflux', 'rectal_exam_feces', 'abdomen',
                                  'abdomo_appearance', 'surgical_lesion', 'cp_data', 'outcome'], dtype = int)
    op = input("Handle Missing Values(y/n): ")
    if op == 'y':
        #Use Data Imputation
        enc.loc[train_data.temp_of_extremities.isnull(), enc.columns.str.startswith("temp_of_extremities_")] = np.nan
        enc.loc[train_data.peripheral_pulse.isnull(), enc.columns.str.startswith("peripheral_pulse_")] = np.nan
        enc.loc[train_data.mucous_membrane.isnull(), enc.columns.str.startswith("mucous_membrane_")] = np.nan
        enc.loc[
            train_data.capillary_refill_time.isnull(), enc.columns.str.startswith("capillary_refill_time_")] = np.nan
        enc.loc[train_data.pain.isnull(), enc.columns.str.startswith("pain_")] = np.nan
        enc.loc[train_data.peristalsis.isnull(), enc.columns.str.startswith("peristalsis_")] = np.nan
        enc.loc[train_data.abdominal_distention.isnull(), enc.columns.str.startswith("abdominal_distention_")] = np.nan
        enc.loc[train_data.nasogastric_tube.isnull(), enc.columns.str.startswith("nasogastric_tube_")] = np.nan
        enc.loc[train_data.nasogastric_reflux.isnull(), enc.columns.str.startswith("nasogastric_reflux_")] = np.nan
        enc.loc[train_data.rectal_exam_feces.isnull(), enc.columns.str.startswith("rectal_exam_feces_")] = np.nan
        enc.loc[train_data.abdomen.isnull(), enc.columns.str.startswith("abdomen_")] = np.nan
        enc.loc[train_data.abdomo_appearance.isnull(), enc.columns.str.startswith("abdomo_appearance_")] = np.nan
        return enc
    else:
        return enc

