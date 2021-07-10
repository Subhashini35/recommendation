# -*- coding: utf-8 -*-
"""
Created on Mon May 25 20:51:41 2020

@author: josh
"""

import numpy as np
import pandas as pd
import nltk
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import re
from pandas import DataFrame 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#import data
df = pd.read_csv("Reviews.csv")
#Basic Information shape and columns
#print(df.columns)
print(df.shape)
count = df.groupby("UserId", as_index=False).count()
mean = df.groupby("UserId", as_index=False).mean()
#merge two dataset create df1
df1 = pd.merge(df, count, how='right', on=["UserId"])
#rename column
df1["Count"] = df1["ProductId_y"]
df1["Score"] = df1["Score_x"]
df1["Summary"] = df1["Summary_x"]
#Create New datafram with selected variables
df1 = df1[["UserId",'Summary','Score',"Count"]]
#choose only products have over 100 reviews
df1 = df1.sort_values(['Count'], ascending=False)
df2 = df1[df1.Count >= 100]
df4 = df.groupby("UserId", as_index=False).mean()
combine_summary = df2.groupby("UserId")["Summary"].apply(list)
combine_summary = pd.DataFrame(combine_summary)
combine_summary.to_csv("combine_summary.csv")
df3 = pd.read_csv("combine_summary.csv")
df3 = pd.merge(df3, df4, on="UserId", how='inner')
df3 = df3[['UserId','Summary','Score']]
#function for tokenizing summary
cleanup_re = re.compile('[^a-z]+')
def cleanup(sentence):
    sentence = sentence.lower()
    sentence = cleanup_re.sub(' ', sentence).strip()
    sentence = " ".join(nltk.word_tokenize(sentence))
    return sentence
df3["Summary_Clean"] = df3["Summary"].apply(cleanup)
df3 = df3.drop_duplicates(['Score'], keep='last')
df3 = df3.reset_index()
docs = df3["Summary_Clean"] 
vect = CountVectorizer(max_features = 100, stop_words='english') 
X = vect.fit_transform(docs) 
#print(DataFrame(X.A, columns=vect.get_feature_names()).to_string()) 
df5 = DataFrame(X.A, columns=vect.get_feature_names())
df5 = df5.astype(int)
df5.to_csv("df5.csv")
kkk  = df.drop_duplicates(['Summary'], keep='last')
kkk = kkk.reset_index()
# First let's create a dataset called X, with 6 records and 2 features each.
X = np.array(df5)

tpercent = 0.95
tsize = int(np.floor(tpercent * len(df5)))
df5_train = X[:tsize]
df5_test = X[tsize:]

lentrain = len(df5_train)
lentest = len(df5_test)
# Next we will instantiate a nearest neighbor object, and call it nbrs. Then we will fit it to dataset X.
nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(df5_train)

# Let's find the k-neighbors of each point in object X. To do that we call the kneighbors() function on object X.
distances, indices = nbrs.kneighbors(df5_train)
#finding similar user and intereting products
for i in range(lentest):
    a = nbrs.kneighbors([df5_test[i]])
    related_product_list = a[1]
    
    first_related_product = [item[0] for item in related_product_list]
    first_related_product = str(first_related_product).strip('[]')
    first_related_product = int(first_related_product)
    second_related_product = [item[1] for item in related_product_list]
    second_related_product = str(second_related_product).strip('[]')
    second_related_product = int(second_related_product)
    
    print ("Based on  reviews, for user is ", df3["UserId"][lentrain + i])
    print ("The first similar user is ", df3["UserId"][first_related_product], ".") 
    print ("He/She likes following products")
    for i in range(295743):
        if (kkk["UserId"][i] == df3["UserId"][first_related_product]) & (kkk["Score"][i] == 5):
            aaa= kkk["ProductId"][i]
            print (aaa),
    print ("--------------------------------------------------------------------")
#predicting review score 
df5_train_target = df3["Score"][:lentrain]
df5_test_target = df3["Score"][lentrain:lentrain+lentest]
df5_train_target = df5_train_target.astype(int)
df5_test_target = df5_test_target.astype(int)

n_neighbors = 3
knnclf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
knnclf.fit(df5_train, df5_train_target)
knnpreds_test = knnclf.predict(df5_test)
#print(classification_report(df5_test_target, knnpreds_test))
mor=classification_report(df5_test_target, knnpreds_test)
cab=(mor[0:65]+mor[108:])
print(cab.replace('4','NFA',1))
print ("Accuracy is",accuracy_score(df5_test_target, knnpreds_test))
