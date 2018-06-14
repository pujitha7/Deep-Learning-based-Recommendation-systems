import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

np.random.seed(123)


def string_processing(data):
    for i in range(0,len(data)):
        x=len(data[:,1][i][0])
        if x<3:
            data[:,1][i]=data[:,1][i][0][x-1]
        else:
            data[:,1][i]=data[:,1][i][0][2]
    return data

# Preprocessing the meta data to remove NAN values
def preprocess(data):


    print(data.groupby(1,as_index=False)[2].sum())

    data.columns=['a','b','c']
    data['b'].replace('Baby & Child Care', 'rare', inplace=True)
    data['b'].replace('Bags, Packs & Accessories', 'rare', inplace=True)
    data['b'].replace('Deodorants & Antiperspirants', 'rare', inplace=True)
    data['b'].replace('Fan Shop', 'rare', inplace=True)
    data['b'].replace('Headbands', 'rare', inplace=True)
    data['b'].replace('Home Brewing & Wine Making', 'rare', inplace=True)
    data['b'].replace('Household Cleaning', 'rare', inplace=True)
    data['b'].replace('Lip Care', 'rare', inplace=True)
    data['b'].replace('Massage & Relaxation', 'rare', inplace=True)
    data['b'].replace('Nails, Screws & Fasteners', 'rare', inplace=True)
    data['b'].replace('Pain Relievers', 'rare', inplace=True)
    data['b'].replace('Party Supplies', 'rare', inplace=True)
    data['b'].replace('Skiing', 'rare', inplace=True)
    data['b'].replace('Sports Sunglasses', 'rare', inplace=True)
    data['b'].replace('Storage & Organization', 'rare', inplace=True)
    data['b'].replace('Bath & Body', 'rare', inplace=True)
    data['b'].replace('Candles & Home Scents', 'rare', inplace=True)
    data['b'].replace("Children's", 'rare', inplace=True)
    data['b'].replace("Cotton & Swabs", 'rare', inplace=True)
    data['b'].replace("Hair Perms & Texturizers", 'rare', inplace=True)
    data['b'].replace("Shampoo Plus Conditioner", 'rare', inplace=True)

    data['c'] = data['c'].fillna(data['c'].mean())
    data=np.array(data)

    tot_d=one_hot(data)

    return data,tot_d

# One hot representation of the meta data
def one_hot(data):
    one_hot=LabelEncoder()
    one_hot.fit(data[:,1])
    transformed=one_hot.transform(data[:,1])

    n_values = np.max(transformed) + 1
    transformed=np.eye(n_values)[transformed]

    print("One hot conversion completed")
    transformed=transformed.reshape([len(data),-1])
    tot_d=np.concatenate([data[:,(0,2)],transformed],axis=1)

    return tot_d

# Saving One hot representation for every product in user purchase sequence
def appending(pro_seq,tot_d):
    tot_seq = []

    print("Adding Meta data")
    for i in range(0, len(pro_seq)):
        sen_seq = []
        for j in range(0, len(pro_seq[i])):
            if pro_seq[i][j] == 'padding_id':
                sen_seq.append(np.zeros([40]))
            else:
                q = tot_d[np.where(tot_d[:, 0] == pro_seq[i][j])][0]
                q = q[1:]
                sen_seq.append(q)
        tot_seq.append(sen_seq)
    return tot_seq


data=np.load('meta_required.npy')
pro_seq=np.load("tot_x_seq.npy")


data=string_processing(data)

data=pd.DataFrame(data)

data.drop(3,axis=1,inplace=True)



data,tot_d=preprocess(data)
tot_seq=appending(pro_seq,tot_d)

print("Saved meta data")

np.save("meta_five_seq.npy",tot_seq)

