#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import fetch_lfw_pairs
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from deepface import DeepFace
from tqdm import tqdm
import pandas as pd


# In[2]:


# lfw_people = fetch_lfw_people()
fetch_lfw_pairs = fetch_lfw_pairs(subset = 'test', color = True
                                  , resize = 1 #this transform inputs to (125, 94) from (62, 47)
                                 )


# In[3]:


pairs = fetch_lfw_pairs.pairs
labels = fetch_lfw_pairs.target
target_names = fetch_lfw_pairs.target_names


# In[4]:


instances = pairs.shape[0]
print("instances: ", instances)


# In[5]:


from deepface.basemodels import VGGFace, Facenet, OpenFace, FbDeepFace
from deepface.basemodels.DlibResNet import DlibResNet
"""vgg_model = VGGFace.loadModel()
print("VGG-Face loaded")

facenet_model = Facenet.loadModel()
print("FaceNet loaded")

openface_model = OpenFace.loadModel()
print("OpenFace loaded")

deepface_model = FbDeepFace.loadModel()
print("DeepFace loaded")

"""
dlib_model = DlibResNet()
print("Dlib loaded")


# In[6]:


plot = True

actuals = []; predictions = []; distances = []

pbar = tqdm(range(0, instances))

for i in pbar:
    pair = pairs[i]
    img1 = pair[0]; img2 = pair[1]
    plt.imshow(img1/255)
    plt.show()
    img1 = img1[:,:,::-1]; img2 = img2[:,:,::-1] #opencv expects bgr instead of rgb
    plt.imshow(img1/255)
    plt.show()

    
    #obj = DeepFace.verify(img1, img2, model_name = 'VGG-Face', model = vgg_model)
    obj = DeepFace.verify(img1, img2, model_name = 'Dlib', model = dlib_model, distance_metric = 'euclidean',enforce_detection=False)
    prediction = obj["verified"]
    predictions.append(prediction)
    
    distances.append(obj["distance"])
    
    label = target_names[labels[i]]
    actual = True if labels[i] == 1 else False
    actuals.append(actual)
    
    if plot:    
        print(i)
        fig = plt.figure(figsize=(5,2))

        ax1 = fig.add_subplot(1,3,1)
        plt.imshow(img1/255)
        plt.axis('off')

        ax2 = fig.add_subplot(1,3,2)
        plt.imshow(img2/255)
        plt.axis('off')

        ax3 = fig.add_subplot(1,3,3)
        plt.text(0, 0.50, label)
        plt.axis('off')

        plt.show()


# In[7]:


accuracy = 100*accuracy_score(actuals, predictions)
precision = 100*precision_score(actuals, predictions)
recall = 100*recall_score(actuals, predictions)
f1 = 100*f1_score(actuals, predictions)


# In[8]:


print("instances: ",len(actuals))
print("accuracy: " , accuracy, "%")
print("precision: ", precision, "%")
print("recall: ", recall,"%")
print("f1: ",f1,"%")


# In[9]:


cm = confusion_matrix(actuals, predictions)


# In[10]:


# cm


# In[11]:


tn, fp, fn, tp = cm.ravel()


# In[12]:


print ("tn ={}, fp ={}, fn ={}, tp ={}".format(tn, fp, fn, tp))


# In[13]:


# true_negative = 472
# false_positive = 28
# false_negative = 45
# true_positive = 455


# In[ ]:




