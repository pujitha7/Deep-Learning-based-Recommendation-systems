from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
import numpy as np

np.random.seed(123)

# Model for numerical representation of the products
def word2vec_model(new_list):
    model = Word2Vec(new_list,size =50,window = 3,min_count =1)
    return model

# Last item purchased by the user is taken as target
def get_target_data(new_list):
    target = []
    for i in range(0, len(new_list)):
        target.append(new_list[i][-1])

    return target

# Numbering every unique product to create one hot for target
def num_products(model,new_list):
    entire_products=[]
    for key,value in model.wv.vocab.items():
        entire_products.append(key)


    product_label=LabelEncoder()
    product_label.fit(entire_products)
    target=get_target_data(new_list)
    target_int = product_label.transform(target)
    return target_int

# Creating five length sequences
def five_len_sequences(new_listy):
    for i in range(0,len(new_listy)):
        q = len(new_listy[i])
        new_listy[i] = new_listy[i][q - 6:q - 1]
    return new_listy



new_list=np.load('user_purchase_seq.npy')

model=word2vec_model(new_list)
model.save('word2vec_model')
y_seq=num_products(model,new_list)
x_seq=five_len_sequences(new_list)



print("Word2vec model built.")
print("Labelled target sequences")


np.save('tot_x_seq.npy',x_seq)
np.save('tot_y.npy',y_seq)


