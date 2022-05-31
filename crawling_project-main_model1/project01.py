from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import pandas as pd
import re
import time
#
#
# category = ['테크·가전', '패션·잡화', '뷰티', '푸드', '홈·리빙', '여행·레저', '스포츠·모빌리티', '출판'] # 8개
import time
from selenium import webdriver
driver = webdriver.Chrome(executable_path= r"./chromedriver.exe")
driver.get("https://www.wadiz.kr/web/wreward/category/293?keyword=&endYn=ALL&order=recent")

time.sleep(2)

try:
    for i in range(1):
        button = driver.find_element_by_xpath(
            f'//*[@id="main-app"]/div[2]/div/div[5]/div[2]/div[2]/div/button')  # 더보기버튼 xpath
        #find_element_by_xpath 는 html의 xpath 를 이용애 원하는 값을 긁어온다. 뒤에 .text를 입력하면 그중 텍스트만 읽어올 수 있다.
        time.sleep(0.5)
        driver.execute_script("arguments[0].click();", button)  # click()으로 에러가나서 써줌
        print('page:', i)

except:
    button = driver.find_element_by_class_name
    button.click()
    print('끝남')

table = driver.find_element_by_class_name('ProjectCardList_container__3Y14k') # 상품들을 포함하는 껍데기 클래스
rows = table.find_elements_by_class_name("ProjectCardList_item__1owJa")       # 열 하나=상품하나 

wadiz_title = []       # 상품 제목
results_reward = []    # 목표 펀딩 달성률
category = []          # 카테고리 분류

for index, value in enumerate(rows):  #enumerate는 리스트가 있는 경우 순서와 리스트의 값을 전달하는 기능
    title=value.find_element_by_class_name("CommonCard_title__1oKJY")               # 상품 제목 클래스
    # result=value.find_element_by_class_name("RewardProjectCard_percent__3TW4_")   # 목표 펀딩 달성률 클래스
    wadiz_title.append(title.text)                                                  # 상품 제목을 wadiz_title 리스트에 append
    # results_reward.append(result.text)                                            # 목표 펀딩 달성률을 results_reward 리스트에  apeend
    category.append("출판")                                                          # 카테고리 분류 한 것을 category 리스트에 append
    # print(title.text, result.text)
    time.sleep(0.3)

import pandas as pd
import numpy as np
df1 = pd.DataFrame({'title':wadiz_title, 'category':category})                       #상품제목과 카테고리 분류 한 것을 데이터 프레임 형태로 반환
# df1 = pd.DataFrame({'title':wadiz_title, 'category':category, 'reward':results_reward})
print(len(df1))
import csv
df1.to_csv("./예측데이터/wadiz_출판_예측데이터.csv".format(len(df1)), mode='w',encoding='utf-8-sig', index=False)