from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.utils import to_categorical
import numpy
from keras import backend as K
from keras import callbacks
from numpy import array
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import h5py

numpy.random.seed(7)
dataset = numpy.genfromtxt("data/spambase.csv", delimiter=",", skip_header=False)#[:, 1:]
X = dataset[:, 0:56]
Y = dataset[:, 57]

#Y=to_categorical(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

#X, Y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=1)

# scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
# scalarX.fit(X)
# scalarY.fit(Y.reshape(100,1))

# X = scalarX.transform(X)
# Y = scalarY.transform(Y.reshape(100,1))
model = Sequential()

#first layer has 12 neurons and expects 8 input variables. The second hidden layer has 8 neurons and finally, the output layer has 1 neuron to predict the class (onset of diabetes or not).

model.add(Dense(56, input_dim=56, activation='relu'))
model.add(Dense(38, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(21, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(11, activation='relu'))
model.add(Dense(8, activation='relu'))
#model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
#model.add(Dense(2, activation='softmax'))

savebest=callbacks.ModelCheckpoint(filepath='checkpoint-{val_acc:.2f}.h5',monitor='val_acc',save_best_only=True)
ReduceLR=callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, verbose=0, mode='auto', cooldown=30, min_lr=0.0001)

class MyCallback(callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        print(K.eval(self.model.optimizer.lr))
PrintLR=MyCallback()
callbacks_list=[ReduceLR,PrintLR,savebest]

#model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=150, validation_data=(x_test, y_test))

scores= model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save('modelR.h5')


model = load_model('modelR.h5')
predictions = model.predict(X)

#print("Prediction number: ", predictions)


#print (model)
#print(h5)

rounded = [round(x[0]) for x in predictions]
print (rounded)

counted =[]
for round in rounded:
    if round > 0.9:
        counted.append(round)

print (len(counted))