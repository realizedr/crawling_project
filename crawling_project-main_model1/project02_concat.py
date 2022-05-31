import pandas as pd
import glob
import datetime

# data_path = glob.glob('./crawling_data/*')
data_path = glob.glob('./예측데이터/*')                          #glob 모듈의 glob 함수는 사용자가 제시한 조건에 맞는 파일명을 리스트 형식으로 반환한다
print(data_path)

df = pd.DataFrame()
for path in data_path[1:] :
    df_temp = pd.read_csv(path)                                 #임시 저장
    df = pd.concat([df, df_temp])
df.dropna(inplace=True)                                         #None 값 제거
df.reset_index(inplace=True,drop = True)                        #제거했으니 reset #인덱스 있는거 합칠 땐 drop = True
print(df.head())
print(df.tail())
df.info()

df.to_csv('./예측데이터/wadiz_data_예측데이터_{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d')), index=False)   #DataFrame을 csv 파일로 내보내기
# df.to_excel('./crawling_data/wadiz_data_{}.xlsx'.format(
#     datetime.datetime.now().strftime('%Y%m%d')), index=False)