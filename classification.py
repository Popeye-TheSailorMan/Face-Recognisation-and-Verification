import numpy as np 
from sklearn.model_selection import train_test_split
from keras_vggface.vggface import VGGFace
from keras.engine import  Model
from keras.layers import Input
import numpy as np
import keras  
from keras.layers import Dense

data = np.load('face_data.npz')
x , y  =  data['x'], data['y']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

y_train = keras.utils.to_categorical(y_train, 4)
y_test = keras.utils.to_categorical(y_test, 4)

resnet = VGGFace(model='resnet50',input_shape=(224, 224, 3))

layer_name = resnet.layers[-2].name

out = resnet.get_layer(layer_name).output
out = Dense(4,activation='softmax')(out)
resnet_4 = Model(resnet.input, out)

for layer in resnet_4.layers[:-1]:
	layer.trainable = False

resnet_4.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print (resnet_4.summary())

resnet_4.fit(x_train, y_train,batch_size=16,epochs=5,validation_data=(x_test, y_test),shuffle=True)

scores = resnet_4.evaluate(x_test, y_test, verbose=1)
print('Test accuracy:', scores[1])




