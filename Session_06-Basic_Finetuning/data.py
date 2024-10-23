# load mimic data
import pandas as pd
# import dask.dataframe as pd
import csv

data_MIMIC_path = r'/gpfs/commons/groups/gursoy_lab/fpollet/data/MIMIC/physionet.org/files/mimic-iv-note/2.2/note/discharge.csv'

print("Loading")
data_MIMIC = pd.read_csv(data_MIMIC_path, encoding='utf-8')

print(data_MIMIC)

print("Saving")
data_MIMIC.iloc[:11000].to_csv('/gpfs/commons/groups/gursoy_lab/fpollet/data/MIMIC/11k.csv')

print("End")