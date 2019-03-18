
# coding: utf-8

# # 作業:
# 嘗試調整參數:  
# sg:sg=1表示採用skip-gram,sg=0 表示採用cbow  
# window:能往左往右看幾個字的意思 

# In[ ]:


import gensim, logging
from gensim.models import word2vec


# In[ ]:


sentences = [['I am a hero', 'sentence'], ['She is a teacher', 'sentence']] 

# train word2vec on the two sentences  

# sg=0 表示COBW, sg=1 表示skip-gram

model = word2vec.Word2Vec(sentences, size=256, min_count=1, window=5, workers=4, sg=0)  


print(model)


model.similarity('I am a hero','She is a teacher')


model.save('mymodel')  
new_model = gensim.models.Word2Vec.load('mymodel')  