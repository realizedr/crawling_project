from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import datetime
import csv
import re

import sys

df_read_csv = pd.read_csv('./예측데이터/wadiz_data_예측데이터_20220530.csv')        # 전처리 전의 예측데이터를 불러 온다.
print(df_read_csv.head())


titles = []
categories = []
for title in df_read_csv['title'] :
    title = re.compile('[^가-힣 ]').sub(' ', title)                              # 한글과 띄어쓰기를 제외한 모든 부분을 제거한다.
    titles.append(title)
print(titles)

df_section_titles = pd.DataFrame(titles, columns=['title'])
for i in df_read_csv['category'] :
    categories.append(i)
df_section_title = pd.DataFrame(categories, columns=['category'])

df_titles = pd.concat([df_section_titles, df_section_title], axis=1)             # axis=1은 컬럼끼리 더하게 한다.

# print(df_titles.head())
# df_titles.info()

# def title_tag(i):
#     hangul = re.compile('[^ ㄱ-ㅣ가-힣+]') # 한글과 띄어쓰기를 제외한 모든 글자
#     # hangul = re.compile('[^ \u3131-\u3163\uac00-\ud7a3]+')  # 위와 동일
#     titles = hangul.sub('', i) # 한글과 띄어쓰기를 제외한 모든 부분을 제거
#     return(titles)
#
# df_titles = title_tag(df_read_csv)

df_titles.to_csv('./예측데이터/wadiz_예측데이터_{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d')), index=False)                    #





