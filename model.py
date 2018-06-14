import numpy as np
from keras.layers import Dense,LSTM
from keras.models import Sequential
from sklearn.metrics import accuracy_score
from tensorflow import set_random_seed
from keras.models import load_model

np.random.seed(123)
set_random_seed(2)


train_x=np.load("train_x.npy")
train_seq=np.load("train_y.npy")
test_x=np.load("test_x.npy")
test_seq=np.load("test_y.npy")

total_vocab=12101

# One hot representation for the targets
def one_hot(seq,total_vocab):
    seq_one_hot=np.zeros([len(seq),total_vocab])
    for i in range(0,len(seq)):
        seq_one_hot[i][seq[i]]=1
    return seq_one_hot

# Model architecture
def model_arch():
    model=Sequential()
    model.add(LSTM(64,input_shape=(5,90)))
    model.add(Dense(20,activation='relu'))
    model.add(Dense(total_vocab,activation='softmax'))
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='ADAM')
    return model

# Training the model
def model_fit(model,train_x,train_seq,total_vocab):
    train_y=one_hot(train_seq,total_vocab)
    print("model is building")
    model.fit(batch_size=64,epochs=10,x=train_x,y=train_y,verbose=0)
    print("model building done")
    model.save('keras_model.h5')
    return model

# Hit rate at 1 on test data
def hit_rate_at_1(prediction,actual):
    return accuracy_score(prediction,actual)

# Hit rata at 5 on test data
def hit_rate_at_5(pred,actual):
    predics = []
    for i in range(0, len(pred)):
        predics.append(np.argsort(pred[i])[-5:])
    count = 0
    for i in range(0, len(predics)):
        if actual[i] in predics[i]:
            count = count + 1

    return count/len(actual)

# Hit rate at 10 on test data
def hit_rate_at_10(pred, actual):
    predics = []
    for i in range(0, len(pred)):
        predics.append(np.argsort(pred[i])[-10:])
    count = 0
    for i in range(0, len(predics)):
        if actual[i] in predics[i]:
            count = count + 1

    return count /len(actual)

# Prediction on test data
def model_predict(model,test_x,test_seq):
    pred=model.predict(x=test_x)
    preddy=np.argmax(a=pred,axis=1)

    print(hit_rate_at_1(preddy,test_seq))
    print(hit_rate_at_5(pred, test_seq))
    print(hit_rate_at_10(pred, test_seq))


model_exists=False

if model_exists:
    model = load_model('keras_model.h5')
else:
    model = model_arch()
    model=model_fit(model,train_x,train_seq,total_vocab)

model_predict(model,test_x,test_seq)

print("Done")

