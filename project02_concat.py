import pandas as pd
import glob
import datetime

data_path = glob.glob('./crawling_data/*')
print(data_path)

df = pd.DataFrame()
for path in data_path[1:] :
    df_temp = pd.read_csv(path) #임시 저장
    df = pd.concat([df, df_temp])
df.dropna(inplace=True) #Nane 값 제거
df.reset_index(inplace=True,drop = True) #제거했으니 reset #인덱스 있는거 합칠 땐 drop = True
print(df.head())
print(df.tail())
df.info()

df = pd.read_csv('./crawling_data/crawling_data0.csv')
df_headline = pd.read_csv('./crawling_data/naver_news_titles_20220526.csv')
df_all = pd.concat([df, df_headline])
df_all.to_csv('./crawling_data/wadiz_fundung_{}.csv'.format(
     datetime.datetime.now().strftime('%Y%m%d')), index=False)