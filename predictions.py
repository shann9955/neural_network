from keras.models import load_model
from numpy import array as numpyarray
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras import backend as K
from keras import callbacks
from sklearn.model_selection import train_test_split
import numpy

dataset = numpy.loadtxt("moves.csv", delimiter=",", skiprows=1)
X = dataset[:,0:18]
Y = dataset[:, 18]

Y=to_categorical(Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

model = Sequential()

model.add(Dense(64, input_dim=18, activation='relu'))
model.add(Dense(18, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(4, activation='softmax'))

savebest=callbacks.ModelCheckpoint(filepath='checkpoint-{val_acc:.2f}.h5',monitor='val_acc',save_best_only=True)
ReduceLR=callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.75, verbose=0, mode='auto', cooldown=30, min_lr=0.0001)
class MyCallback(callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        print(K.eval(self.model.optimizer.lr))
PrintLR=MyCallback()
callbacks_list=[ReduceLR,PrintLR,savebest]

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=150, validation_data=(x_test, y_test))

scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save('model.h5')