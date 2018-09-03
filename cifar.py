## Saket Joshi
## TCS15B034
## DLCV Assignment 1, CIFAR-10 

# Import required libraries
import pickle
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt

# Data Path
data_path = './cifar-10-batches-py'

############# Get data ##########
# returns data with shape 10000, 3072
def get_batch(n):
	with open(data_path+'/data_batch_{}'.format(n)) as f:
		dat = pickle.load(f)
	return dat

def get_data():
	labels = []
	images = []
	for i in range(1,2):
		with open(data_path+'/data_batch_{}'.format(i)) as f:
			dat = pickle.load(f)
		labels.append(dat['labels'])
		images.append(dat['data'])

	labels = np.concatenate(tuple(labels))
	images = np.concatenate(tuple(images))
	return (images,labels)

def get_test_data():
	with open(data_path+'/test_batch') as f:
			dat = pickle.load(f)
	return (dat['data'],dat['labels'])


# Function to extract histogram feature
def extract_color_histogram(image, bins=(8, 8, 8)):
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
	hist = cv2.normalize(hist,hist)
	return hist.flatten()

# Containers for training data
hists = []
features = []
hogs = []
# Containers for testing data
test_hists = []
test_features = []
test_hogs = []

raw_Images,labels = get_data()
test_raw_Images,test_labels = get_test_data()

################## 
## To get the hog features
cell_size = (8, 8)  # h x w in pixels
block_size = (2, 2)  # h x w in cells
nbins = 4  # number of orientation bins
shape = (32,32)
# winSize is the size of the image cropped to an multiple of the cell size
hog = cv2.HOGDescriptor(_winSize=(shape[1] // cell_size[1] * cell_size[1],
                                  shape[0] // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)

n_cells = (shape[0] // cell_size[0], shape[1] // cell_size[1])


## Computing HOG features, preparing training data
n = len(raw_Images)
for i in range(n):
	img = raw_Images[i]
	img = img.reshape((32,32,3))
	hists.append(extract_color_histogram(img))
	hogs.append(hog.compute(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))\
					.reshape(n_cells[1] - block_size[1] + 1,
                        n_cells[0] - block_size[0] + 1,
                        block_size[0], block_size[1], nbins) \
               		.transpose((1, 0, 2, 3, 4)))

## Computing HOG features for test data
tn = len(test_raw_Images)
for i in range(n):
	img = test_raw_Images[i]
	img = img.reshape((32,32,3))
	test_hists.append(extract_color_histogram(img))
	test_hogs.append(hog.compute(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))\
					.reshape(n_cells[1] - block_size[1] + 1,
                        n_cells[0] - block_size[0] + 1,
                        block_size[0], block_size[1], nbins) \
               		.transpose((1, 0, 2, 3, 4)))


print("done precomputation")

# Iterating over number of PCA dimensions
for nc in range(10,11,10):
	
	pca = PCA(n_components=nc)
	pca.fit(hists)

	pcas = pca.transform(hists)
	features = [np.append(pcas[i],hogs[i].flatten()) for i in range(n)]

	test_pcas = pca.transform(test_hists)
	test_features = [np.append(test_pcas[i],test_hogs[i].flatten()) for i in range(tn)]

	# Iterating over number of KNN neighbours
	for nneb in range(21,22,2):

		#(trainRI, testRI, trainRL, testRL) = train_test_split(rawImages, labels, test_size=0.25, random_state=42)
		#(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.25, random_state=42)
		model = KNeighborsClassifier(n_neighbors=nneb)
		#model.fit(trainFeat, trainLabels)
		model.fit(features, labels)
		acc = model.score(test_features, test_labels)
		print("[INFO] Accuracy for PCA dimensions={}, neighbours={}: {:.2f}%".format(nc,nneb,acc * 100))

		
		image = raw_Images[0]
		img = image.reshape((32,32,3))
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		# Specialized hog configuration since image is small in size
		hog_feats = hog.compute(img)\
		               .reshape(n_cells[1] - block_size[1] + 1,
		                        n_cells[0] - block_size[0] + 1,
		                        block_size[0], block_size[1], nbins) \
		               .transpose((1, 0, 2, 3, 4)) 


# Visualizing on 2d pca
nc = 2

pca = PCA(n_components=nc)
pca.fit(hists)

pcas = pca.transform(hists)
features = [np.append(pcas[i],hogs[i].flatten()) for i in range(n)]

test_pcas = pca.transform(test_hists)
test_features = [np.append(test_pcas[i],test_hogs[i].flatten()) for i in range(tn)]

# Iterating over number of KNN neighbours
nneb = 15

#(trainRI, testRI, trainRL, testRL) = train_test_split(rawImages, labels, test_size=0.25, random_state=42)
#(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.25, random_state=42)
model = KNeighborsClassifier(n_neighbors=nneb)
#model.fit(trainFeat, trainLabels)
model.fit(features, labels)
class_features = [[] for i in range(10)]
for i in range(len(test_features)):
	class_features[model.predict([test_features[i]])[0]].append((test_features[i][0],test_features[i][1]))
colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired  
colorst = [colormap(i) for i in np.linspace(0, 0.9,10)]       
for i in range(10):
	x,y = zip(*class_features[i])
	plt.scatter(x,y,s=1,color=colorst[i])
plt.show()
