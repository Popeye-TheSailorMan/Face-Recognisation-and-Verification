import numpy as np
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
import glob 
from sklearn.utils import shuffle 
images  = []
labels  = []

name2label = {'Abhi':0,'Prakhar':1,'Nikhil':2,'Shubham':3}
labe2name = {0:'Abhi',1:'Prakhar',2:'Nikhil',3:'Shubham'}

for sub_folder in glob.glob('./Data/*'):
	
	name = sub_folder.split('/')[-1]

	for file in glob.glob(sub_folder + '/*'):
		img = image.load_img(file, target_size=(224, 224))
		x = image.img_to_array(img)
		images.append(x)
		labels.append(name2label[name])
		print (x.shape , name2label[name])

		
images = np.array(images)
labels = np.array(labels)
x = utils.preprocess_input(images, version=2) 
print (images.shape)
print (labels.shape)

images , labels = shuffle(images , labels)
np.savez('face_data', x=images, y=labels)





