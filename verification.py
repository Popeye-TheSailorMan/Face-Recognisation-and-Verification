import numpy as np 
from sklearn.model_selection import train_test_split
from keras_vggface.vggface import VGGFace
from keras.engine import  Model
from keras.layers import Input
import numpy as np 

data = np.load('face_data.npz')
x , y  =  data['x'], data['y']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)


a_x = x_train[y_train == 0]
a_y = y_train[y_train == 0]


layer_name = 'flatten_1' # edit this line
resnet = VGGFace(model='resnet50',input_shape=(224, 224, 3))
out = resnet.get_layer(layer_name).output
resnet_features = Model(resnet.input, out)

a_x_feats = resnet_features.predict(a_x)
#print (a_x_feats.shape)

dists = []
for i in range(400):

	ind1 = np.random.randint(0,a_x_feats.shape[0])
	ind2 = np.random.randint(0,a_x_feats.shape[0])
	dist = np.linalg.norm(a_x_feats[ind1,:]-a_x_feats[ind2,:])
	#print (dist)

	dists.append(dist)

dists = np.array(dists)
alpha = np.average(dists)

x_test_feats = resnet_features.predict(x_test)

for i in range(x_test_feats.shape[0]):
	dists = []
	for j in range(a_x_feats.shape[0]):
		dist = np.linalg.norm(x_test_feats[i,:]-a_x_feats[j,:])
		dists.append(dist)

	dists = np.array(dists)
	min_ = np.min(dists)
	avg_ = np.average(dists)

	if (avg_ <= alpha):
		print ("we verify its abhi predicted class 0 ",y_test[i])

	else:
		print ("we verify its not abhi predicted class not 0 ",y_test[i])










