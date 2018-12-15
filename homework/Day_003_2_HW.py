# -*- coding: utf-8 -*-
"""
Day 03_2 讀取非csv的資料
author： Martin.Lee
Date：2018/12/15

懶人複製連結: https://raw.githubusercontent.com/vashineyu/slides_and_others/master/tutorial/examples/imagenet_urls_examples.txt
Hints: 使用 Request 抓取資料 #https://blog.gtwang.org/programming/python-requests-module-tutorial/
Hints: 字串分割 str.split #http://www.runoob.com/python/att-string-split.html
Hints: 例外處理: Try-Except #https://pydoing.blogspot.com/2011/01/python-try.html

"""
import requests
import pandas as pd

## 假如我們不想把資料載到自己的電腦裡?


target_url = 'https://raw.githubusercontent.com/vashineyu/slides_and_others/master/tutorial/examples/imagenet_urls_examples.txt'

response = requests.get(target_url)
data = response.text

# 用 request 傳送回來的資料不會認得斷行符號
print(len(data))
data[0:100]

# 找到換行符號，用該符號做字串分割後，把它拿掉
split_tag = '\n'

data = data.split(split_tag)
print(len(data))
data[0]

##將 txt 轉成 pandas dataframe
# arrange_data = [['id': 'url']]
arrange_data = list()
for line in data:
    line = line.split('\t')
    arrange_data.append(line)
    
df = pd.DataFrame(arrange_data, columns = ['id', 'url'])
df.head()


##讀取圖片，請讀取上面 data frame 中的前 5 張圖片
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

# 請用 df.loc[...] 得到第一筆資料的連結
first_link = df.loc[0,'url'] #https://blog.csdn.net/qq1483661204/article/details/77587881

response = requests.get(first_link)
img = Image.open(BytesIO(response.content))

# Convert img to numpy array

plt.imshow(img)
plt.show()

def img2arr_fromURLs(url_list, resize = False):
    """
    請完成這個 Function
    Args
        - url_list: list of URLs
        - resize: bool
    Return
        - list of array
    """
    img_list = []
    for url in url_list:
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img_list.append(img)
        except Exception as e:
            print('連結失效 %s' % (url))
    return img_list

#result = img2arr_fromURLs(df.loc[0:5,'url'])

result = img2arr_fromURLs(df[0:5]['url'].values)
print("Total images that we got: %i " % len(result)) # 如果不等於 5, 代表有些連結失效囉

for im_get in result:
    plt.imshow(im_get)
    plt.show()