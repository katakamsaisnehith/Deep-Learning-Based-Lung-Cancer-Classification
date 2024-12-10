import warnings
warnings.filterwarnings("ignore")
import glob
import cv2
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import seaborn as sns
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from sklearn.metrics import confusion_matrix,classification_report
import os
from tqdm import tqdm

data = r"C:\Users\katak\OneDrive\Desktop\major\LUNG DISEASES CLASSIFICATION BY ANALYSIS OF LUNG TISSUE DENSITIE\7.LUNG DISEASES CLASSIFICATION BY ANALYSIS OF LUNG TISSUE DENSITIES\dataset"
cancer = glob.glob(r'dataset\CANCER\*.jpg')
normal = glob.glob(r'dataset\NORMAL\*.jpg')
cancer
normal
print('Number of images with cancer : {}'.format(len(cancer)))
print('Number of images with normal : {}'.format(len(normal)))

#Threshold segmentation
print('\n')
print('Threshold Segmentation')
image=cv2.imread('dataset\CANCER\cancer (3).jpg')
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
ret, TB = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
ret, TBI = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
ret, TT = cv2.threshold(img, 120, 255, cv2.THRESH_TRUNC)
ret, T_T = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO)
ret, TTI = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO_INV)
 

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, TB, TBI, TT, T_T, TTI]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

 
categories = ['cancer', 'normal']
len_categories = len(categories)
print(len_categories)
image_count = {}
train_data = []

for i , category in tqdm(enumerate(categories)):
    class_folder = os.path.join(data, category)
    label = category
    image_count[category] = []
    
    for path in os.listdir(os.path.join(class_folder)):
        image_count[category].append(category)
        train_data.append(['{}/{}'.format(category, path), i, category])
#show image count
for key, value in image_count.items():
    print('{0} -> {1}'.format(key, len(value)))
#create a dataframe
df = pd.DataFrame(train_data, columns=['file', 'id', 'label'])
df.shape
df.head()

from keras.preprocessing import image
# function to get an image
def read_img(filepath, size):
    img = image.load_img(os.path.join(data, filepath), target_size=size)
    #convert image to array
    img = image.img_to_array(img)
    return img

nb_rows = 3
nb_cols = 5
fig, axs = plt.subplots(nb_rows, nb_cols, figsize=(10, 5))
plt.suptitle('SAMPLE IMAGES')
for i in range(0, nb_rows):
    for j in range(0, nb_cols):
        axs[i, j].xaxis.set_ticklabels([])
        axs[i, j].yaxis.set_ticklabels([])
        axs[i, j].imshow((read_img(df['file'][np.random.randint(120)], (255,255)))/255.)
plt.show()


lst_cancer = []
for x in cancer:
  lst_cancer.append([x,1])
lst_normal = []
for x in normal:
  lst_normal.append([x,0])
lst_complete = lst_cancer + lst_normal
random.shuffle(lst_complete)

df = pd.DataFrame(lst_complete,columns = ['files','target'])
df.head(10)
filepath_img ="CANCER/NORMAL/*.jpg"
df = df.loc[~(df.loc[:,'files'] == filepath_img),:]
df.shape

plt.figure(figsize = (10,10))
sns.countplot(x = "target",data = df)
plt.title("Cancer and Normal") 
plt.show()

def preprocessing_image(filepath):
  img = cv2.imread(filepath) #read
  img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR) #convert
  img = cv2.resize(img,(196,196))  # resize
  img = img / 255 #scale
  return img

def create_format_dataset(dataframe):
  X = []
  y = []
  for f,t in dataframe.values:
    X.append(preprocessing_image(f))
    y.append(t)
  
  return np.array(X),np.array(y)
X, y = create_format_dataset(df)
X.shape,y.shape
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,stratify = y)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


'''CNN'''
CNN = Sequential()

CNN.add(Conv2D(128,(2,2),input_shape = (196,196,3),activation='relu'))
CNN.add(Conv2D(64,(2,2),activation='relu'))
CNN.add(MaxPooling2D())
CNN.add(Conv2D(32,(2,2),activation='relu'))
CNN.add(MaxPooling2D())

CNN.add(Flatten())
CNN.add(Dense(128))
CNN.add(Dense(1,activation= "sigmoid"))
CNN.summary()
CNN.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])
CNN.fit(X_train,y_train,validation_data=(X_test,y_test),epochs = 3,batch_size = 20)
print("Accuracy of the CNN is:",CNN.evaluate(X_test,y_test)[1]*100, "%")
history = CNN.history.history

# Plotting the accuracy
train_loss = history['loss']
val_loss = history['val_loss']
train_acc = history['acc']
val_acc = history['val_acc']
   
# Loss
plt.figure()
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Testing Loss')
plt.title('Loss')
plt.legend()
plt.show()
   
#Accuracy
plt.figure()
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Testing Accuracy')
plt.title('Accuracy')
plt.legend()
plt.show()

y_pred = CNN.predict(X_test)
y_pred = y_pred.reshape(-1)
y_pred[y_pred<0.5] = 0
y_pred[y_pred>=0.5] = 1
y_pred = y_pred.astype('int')
y_pred
print('\n')
classification=classification_report(y_test,y_pred)
print(classification)
print('\n')
plt.figure(figsize = (5,4.5))
cm = confusion_matrix(y_test,y_pred)
print(cm)
sns.heatmap(confusion_matrix(y_test,y_pred),annot = True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


#predictive result for single image
from tkinter import filedialog
from tkinter import *
import tkinter.messagebox
root = Tk()
root.withdraw()
options = {}
options['initialdir'] = 'CT_SCAN'
global fileNo
#options['title'] = title
options['mustexist'] = False
file_selected = filedialog.askopenfilename(title = "Select file",filetypes = (("JPG files","*.jpg"),("PNG files","*.png"),("all files","*.*")))
head_tail = os.path.split(file_selected)
fileNo=head_tail[1].split('.')

InpImg=cv2.imread(file_selected)
InpImg = cv2.cvtColor(InpImg,cv2.COLOR_RGB2BGR) #convert
InpImg = cv2.resize(InpImg,(196,196))
  
cv2.imshow('Input image', InpImg)
plt.imshow(InpImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

X_test_single=np.zeros((1,196,196,3))
X_test_single[0,:,:,:]=InpImg
y_pred = CNN.predict(X_test_single)
y_pred = y_pred.reshape(-1)
y_pred[y_pred<0.5] = 0
y_pred[y_pred>=0.5] = 1
y_predsingle = y_pred.astype('int')
if y_predsingle==1:
    tkinter.messagebox.showinfo('Info','Cancer')
    print('SELECTED CT SCAN IMAGE IS CANCER')
else:
    tkinter.messagebox.showinfo('Info','Normal')
    print('SELECTED CT SCAN IMAGE IS NORMAL')
    


#=============================Prediction====================================================
    
plt.figure(figsize=(12,7))
for i in range(10):
    sample = random.choice(range(len(X_test)))
    image = X_test[sample]
    category = y_test[sample]
    pred_category = y_pred
    
    if category== 0:
        label = "Normal"
    else:
        label = "Cancer"
        
    if pred_category== 0:
        pred_label = "Normal"
    else:
        pred_label = "Cancer"
        
    plt.subplot(2,5,i+1)
    plt.imshow(image)
    plt.xlabel("Actual:{}\nPrediction:{}".format(label,pred_label))
plt.tight_layout() 
