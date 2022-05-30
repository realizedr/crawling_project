import pandas as pd
import glob
import datetime

df_read_csv = pd.read_csv('./crawling_data/wadiz_20220530.csv')

# for i in df_read_csv :
#     if df_read_csv[0].isnull() == 1 :
#         i.dropna(inplace=True)
#
print(df_read_csv.isnull().sum())

df_read_csv.to_csv('./crawling_data/wadiz_{}(2).csv'.format(
    datetime.datetime.now().strftime('%Y%m%d')), index=False)