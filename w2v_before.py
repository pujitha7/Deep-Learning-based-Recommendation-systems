import pandas as pd
import numpy as np
import datetime

np.random.seed(123)

# converting the unix time to readable time
def unix_time_conversion(unix_col):
    for i in range(0,len(unix_col)):
        unix_col[i]=datetime.datetime.fromtimestamp(int(unix_col[i])).strftime('%Y-%m-%d')

    return unix_col

# Grouping all items purchased by a similar user
def user_grouping(data):
    x=data.groupby(list(data)[0]).groups
    data=np.array(data)
    data[:,2]=unix_time_conversion(data[:,2])
    dictList=[]

    for key, value in x.items():
        temp = value
        val=data[temp,]
        val=val.tolist()
        dictList.append(val)

    return dictList

# Creating a sequence of items
def time_sorting(seq):
    return sorted(seq, key=lambda x: x[2])


# Considering sequences which are of minimum length 6
def min_six_len_seq(dictList):
    for i in range(0,len(dictList)):
        dictList[i]=time_sorting(dictList[i])
        for j in range(0,len(dictList[i])):
            dictList[i][j]=dictList[i][j][1]


    for i in range(0,len(dictList)):
        if len(dictList[i])<6:
            w=6-len(dictList[i])
            dictList[i]=['padding_id']*w+dictList[i]
    return dictList



df=pd.read_csv('beauty_df.csv')
data=df[['reviewerID','asin','unixReviewTime']]

dictList=user_grouping(data)
dictList=min_six_len_seq(dictList)

print("Each user's product purchase sequences made.")

np.save('user_purchase_seq.npy',dictList)
