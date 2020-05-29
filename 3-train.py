import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
import sys
 
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
 
print('3-train.py Perform PCA / Train ANN')

# Get the target person
if(len(sys.argv) == 2):
    name = sys.argv[1]
else:
    print("no target person specified! taking Chris")
    name= 'chris'

# Load own images
myX = np.loadtxt(open('Data/' + name + '.csv', "rb"), delimiter=",")

# Load LFW Face Database
lfw_dataset = fetch_lfw_people(min_faces_per_person=100)

# Init LFW Dataset 
_, h, w = lfw_dataset.images.shape
xLFW = lfw_dataset.data
yLFW = lfw_dataset.target
target_names = lfw_dataset.target_names

# print('Dataset Stats:')
# print(xLFW.shape)
# print(yLFW.shape)

# Create a new Y for our own Face
# take target_names.shape[0] to get the "next" free index
myY = []
for i in range(myX.shape[0]):
    myY.append(target_names.shape[0])

# Also Add the own Name to List
lst = list(target_names)
lst.append('my ' + name)
target_names = np.asarray(lst)
print(target_names)

# split LFW Database into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(xLFW, yLFW, test_size=0.3)

print('Train Stats:')
print(X_train.shape)
print('Test Stats:')
print(X_test.shape)


# usually we have less Data, so we take a smaller test_size
myX_train, myX_test, myy_train, myy_test = train_test_split(myX, myY, test_size=0.2)

# Add own Face to the Set
X_train = np.concatenate((myX_train, X_train))
X_test = np.concatenate((myX_test, X_test))
y_train = np.concatenate((myy_train, y_train))
y_test = np.concatenate((myy_test, y_test))

print('Train Stats:')
print(X_train.shape)
print('Test Stats:')
print(X_test.shape)


# Set Number of Eigenfaces
k = target_names.shape[0] * target_names.shape[0]

# Compute a PCA 
pca = PCA(n_components = k,whiten=True).fit(X_train)
 
# apply PCA transformation
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# save the PCA 
pk.dump(pca, open('pca.pkl','wb'))

print('PCA Stats:')
print(pca.mean_.shape)
print(pca.components_.shape)

print(pca.mean_)
print(pca.components_[pca.components_.shape[0]-1])

fig1 = plt.figure()
ax1 = fig1.subplots()
eigenfaceMean = pca.mean_
imgMean = eigenfaceMean.reshape(62,47)
ax1.imshow(imgMean,cmap='gray')
ax1.set_title("Mean Eigenface")
ax1.axis('off')

plt.figure()
for i in range(36):
    eigenface = pca.components_[i]
    img = eigenface.reshape(62,47)
    plt.subplot(6,6,i+1)
    plt.imshow(img,cmap='gray')
    plt.axis('off')

# train a neural network
print("Fitting the classifier to the training set")
model = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, verbose=True, early_stopping=True).fit(X_train_pca, y_train)

# save the ANN 
pk.dump(model, open('weights.pkl', 'wb'))

y_pred = model.predict(X_test_pca)
print(classification_report(y_test, y_pred, target_names=target_names))

# Visualization of Test Data
def plot_gallery(images, titles, h, w, rows=5, cols=5):
    plt.figure(3)
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())
 
def titles(y_pred, y_test, target_names):
    for i in range(y_pred.shape[0]):
        pred_name = target_names[y_pred[i]].split(' ')[-1]
        true_name = target_names[y_test[i]].split(' ')[-1]
        yield 'predicted: {0}\ntrue: {1}'.format(pred_name, true_name)
 
prediction_titles = list(titles(y_pred, y_test, target_names))
plot_gallery(X_test, prediction_titles, h, w)

plt.show()

print('3-train.py Done')