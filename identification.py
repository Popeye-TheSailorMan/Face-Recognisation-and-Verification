import numpy as np 
from sklearn.model_selection import train_test_split
from keras_vggface.vggface import VGGFace
from keras.engine import  Model
from keras.layers import Input
import numpy as np 

data = np.load('face_data.npz')
x , y  =  data['x'], data['y']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

layer_name = 'flatten_1' # edit this line
resnet = VGGFace(model='resnet50',input_shape=(224, 224, 3))
out = resnet.get_layer(layer_name).output
resnet_features = Model(resnet.input, out)

x_train_feats = resnet_features.predict(x_train) 
x_test_feats  = resnet_features.predict(x_test)
np.savez('feats',train=x_train_feats,test=x_test_feats)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3) 
knn.fit(x_train_feats , y_train)
print (knn.score(x_test_feats , y_test))




