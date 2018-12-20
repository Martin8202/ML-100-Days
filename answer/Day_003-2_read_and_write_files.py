
# coding: utf-8

# ## 讀取 txt 檔
# * 請讀取 [text file](https://raw.githubusercontent.com/vashineyu/slides_and_others/master/tutorial/examples/imagenet_urls_examples.txt)
# * 懶人複製連結: https://raw.githubusercontent.com/vashineyu/slides_and_others/master/tutorial/examples/imagenet_urls_examples.txt

# In[1]:


## 假如我們不想把資料載到自己的電腦裡?
import requests
target_url = "https://raw.githubusercontent.com/vashineyu/slides_and_others/master/tutorial/examples/imagenet_urls_examples.txt"

response = requests.get(target_url)
data = response.text

# 用 request 傳送回來的資料不會認得斷行符號
print(len(data))
data[0:100]


# In[2]:


# 我們自己斷行
data = data.split("\n")
print(len(data))
data[0]


# ## 將 txt 轉成 pandas dataframe

# In[3]:


import pandas as pd

arrange_data = []
for d in data:
    line = d.split("\t")
    arrange_data.append(line)
    
df = pd.DataFrame(arrange_data)
df.head()


# ## 懶人解法：直接用 pandas 讀取網路連結

# In[4]:


## 注意：一行中的分隔符號有時候為 "," (預設), 但也常常是 tab (\t)
## 如果非 "," 的話，記得要加上參數告訴 pd.read_csv
## 建議你/妳可以把 sep 的參數抽掉試看看

df_lazy = pd.read_csv(target_url, sep="\t", header=None)


# In[5]:


df_lazy.head()


# ## 讀取圖片，請讀取上面 data frame 中的前 5 張圖片

# In[6]:


from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt

response = requests.get(df.loc[0, 1]) # 先讀取第一筆資料的圖片
img = Image.open(BytesIO(response.content))
img = np.array(img)
print(img.shape)
plt.imshow(img)
plt.show()


# In[7]:


def img2arr_fromURLs(url_list, resize = False):
    img_list = []
    for url in url_list:
        response = requests.get(url)
        try:
            img = Image.open(BytesIO(response.content))
            if resize:
                img = img.resize((256,256)) # 假如 resize, 就先統一到 256 x 256
            img = np.array(img)
            img_list.append(img)
        except:
            # 只有在 response.status_code 為 200 時，才可以取得圖片，若有 404 或其他 status code, 會碰到 Error, 所以我們用 Try 語法避開取不到的狀況
            pass
    
    return img_list


# In[8]:


result = img2arr_fromURLs(df[0:5][1].values)
print("Total images that we got: %i " % len(result)) # 如果不等於 5, 代表有些連結失效囉

for im_get in result:
    plt.imshow(im_get)
    plt.show()


# In[9]:


result = img2arr_fromURLs(df[0:5][1].values, resize=True)
print("Total images that we got: %i " % len(result)) # 如果不等於 5, 代表有些連結失效囉

for im_get in result:
    plt.imshow(im_get)
    plt.show()

