import numpy as nm
import pandas as py
import matplotlib.pyplot as plt
#For creating heatmaps
import seaborn as sns;sns.set()

from sklearn.datasets import fetch_20newsgroups
data=fetch_20newsgroups()
data.target_names

#Defining categories
categories=['alt.atheism','comp.graphics','comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware','comp.sys.mac.hardware',
 'comp.windows.x','misc.forsale','rec.autos','rec.motorcycles',
 'rec.sport.baseball','rec.sport.hockey',
 'sci.crypt','sci.electronics','sci.med','sci.space',
 'soc.religion.christian','talk.politics.guns',
 'talk.politics.mideast','talk.politics.misc','talk.religion.misc']

#Training data
train=fetch_20newsgroups(subset='train',categories=categories)
#Test Data
test=fetch_20newsgroups(subset='test',categories=categories)

print(train.data[5])

#Importing packages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
# Creating model in Naive Bayes
model=make_pipeline(TfidfVectorizer(),MultinomialNB())
#Training the model on train data
model.fit(train.data,train.target)
#Creating labels for test data
labels=model.predict(test.data)

#Creating a confusion matrix
from sklearn.metrics import confusion_matrix
mat=confusion_matrix(test.target,labels)
sns.heatmap(mat.T,square=True,annot=True, fmt='d',cbar=False,
            xticklabels=train.target_names,
            yticklabels=train.target_names)
#Plotting a heatmap of confusion matrix
plt.xlabel('true label')
plt.ylabel('predicted label')

#Predicting category on new data based on trained model
def predict_category(s,train=train,model=model):
    pred= model.predict([s])
    return train.target_names[pred[0]]

#Predicting using new values
predict_category('Jesus Christ')
predict_category('Audi')
predict_category('President')