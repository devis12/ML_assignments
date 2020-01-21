# pylint: disable=E1136 
import numpy as np
import pandas as pd


#loading training data
#raw_train_data = pd.read_csv("tweets-train-data.csv", names=["text, datetime","retweet_count","favorite_count","place_full_name"])
#raw_train_targets = pd.read_csv("tweets-train-targets.csv", names=["who"])

raw_train_data = pd.read_csv("tweets-train-data.csv", header=None, names=["text", "datetime","retweet_count","favorite_count","place_full_name"])
raw_train_targets = pd.read_csv("tweets-train-targets.csv", header=None, names=["who"])

if(raw_train_data.shape[0] != raw_train_targets.shape[0]):
    print("ERROR!\nTraining samples not valid!")
    exit

train_data = raw_train_data.join(raw_train_targets)
print("# of examples in raw training data = ", len(train_data))

#print(train_data.shape)
#print(train_data)

maskDT = (train_data["who"] == "DT")
train_data_DT = train_data[maskDT]
avg_retweets_DT = train_data_DT["retweet_count"].mean()
avg_fav_DT = train_data_DT["favorite_count"].mean()

print("# of DT examples in raw training data = ", len(train_data_DT))
print("Avg # of retweets for DT = ", avg_retweets_DT)
print("Avg # of favourites for DT = ", avg_fav_DT)

maskHC = (train_data["who"] == "HC")
train_data_HC = train_data[maskHC]
avg_retweets_HC = train_data_HC["retweet_count"].mean()
avg_fav_HC = train_data_HC["favorite_count"].mean()

print("# of HC examples in raw training data = ", len(train_data_HC))
print("Avg # of retweets for HC = ", avg_retweets_HC)
print("Avg # of favourites for HC = ", avg_fav_HC)
