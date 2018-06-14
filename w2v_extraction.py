import numpy as np
from gensim.models import Word2Vec

np.random.seed(123)

tot_d=np.load("meta_five_seq.npy")
new_list=np.load('tot_x_seq.npy')
model = Word2Vec.load('word2vec_model')
y_tot=np.load('tot_y.npy')

print(len(new_list))


# Appending product vectors with their meta data
def w2v_data_ext(new_list):

    w2v_data=[]

    for i in range(0,len(new_list)):
        seq_vec=[]
        for j in range(0,len(new_list[i])):
            w = tot_d[i][j]
            q = np.concatenate([model.wv[new_list[i][j]], w.reshape([40])])
            seq_vec.append(q)
        w2v_data.append(seq_vec)

    return np.asarray(w2v_data)


# Train and test split
def train_test_split(w2v_data,y_tot):
    train_x=w2v_data[0:17890]
    test_x=w2v_data[17890:]
    train_y=y_tot[0:17890]
    test_y=y_tot[17890:]
    print(np.shape(train_x))
    print(np.shape(train_y))
    return train_x,train_y,test_x,test_y


w2vdata=w2v_data_ext(new_list)
train_x,train_y,test_x,test_y=train_test_split(w2vdata,y_tot)

print("Train and test data saved")

np.save("train_x.npy", train_x)
np.save("train_y.npy", train_y)
np.save("test_x.npy", test_x)
np.save("test_y.npy", test_y)



