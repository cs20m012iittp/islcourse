import torch
from torch import nn
import torch.optim as optim

# You can import whatever standard packages are required

# full sklearn, full pytorch, pandas, matplotlib, numpy are all available
# Ideally you do not need to pip install any other packages!
# Avoid pip install requirement on the evaluation program side, if you use above packages and sub-packages of them, then that is fine!

###### PART 1 ######
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
from sklearn.datasets import make_blobs, make_circles

def get_data_blobs(n_points=100):
  pass
  # write your code here
  # Refer to sklearn data sets
  X, y =  make_blobs(n_samples=n_points,centers = 5)
  print(X.shape, y.shape)
  return X,y

X , y = get_data_blobs()


def build_kmeans(X,y ,k=10):
  pass
  km = KMeans(n_clusters=k)
  km.fit(X,y)
  return km

def assign_kmeans(km, X):
  pass
  ypred =  km.predict(X)
  return ypred

km = build_kmeans(X,y)
y_pred = assign_kmeans(km, X)
print(y_pred)

def get_data_circles(n_points=100):
  pass
  # write your code here
  # Refer to sklearn data sets
  X, y =  make_circles(n_samples=n_points)
  print(X.shape, y.shape)
  # write your code ...
  return X,y
X , y= get_data_circles()

km2 = build_kmeans(X,y)
y_pred2 = assign_kmeans(km2, X)

from sklearn.metrics.cluster import v_measure_score
print("v score is  :",v_measure_score(y_pred,y_pred2))

from sklearn.metrics.cluster import homogeneity_score
h = homogeneity_score(y_pred,y_pred2)
print("h score is  :",h)

from sklearn.metrics import completeness_score
c = completeness_score(y_pred,y_pred2)
print("c score is  :",c)



###### PART 2 ######


from sklearn.datasets import load_digits 
def get_data_mnist():
  digits = load_digits()
  print(type(digits))
  print(digits.keys())
  print(digits.data.shape)
  X , y = digits.data, digits.target
  print(X.shape, y.shape)
  return X, y



from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score


from sklearn import svm, metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score


x , y = get_data_mnist()

def build_lr_model(X=None, y=None):
  pass
  lr_model = LogisticRegression(random_state=0).fit(X, y)
  # write your code...
  # Build logistic regression, refer to sklearn
  return lr_model

lr_model = build_lr_model(x,y)
print(lr_model)

def build_rf_model(X, y):
  pass
  rf_model = RandomForestClassifier(n_estimators=100)
  rf_model.fit(X,y)
 
  # write your code...
  # Build Random Forest classifier, refer to sklearn
  return rf_model
  # write your code...
  # Build Random Forest classifier, refer to sklearn
  
rf_model = build_rf_model(x,y)
print(rf_model)

def get_metrics(model,X,y):
  pass
  ypred = model.predict(X)
  # Obtain accuracy, precision, recall, f1score, auc score - refer to sklearn metrics
  
  print("Classification_Report : ")
  print(classification_report(y,ypred))
  print("Confusion_Matrix : ")
  print(confusion_matrix(y, ypred))
  print("f1 score is : ",f1_score(y, ypred, average=None))
  print("Precision_score :",precision_score(y, ypred, average=None))
  print("recall_score : ",recall_score(y, ypred, average=None))
  print("accuracy : ", accuracy_score(y, ypred))
  # write your code here...
  
  
x, y = get_data_mnist()
get_metrics(rf_model, x ,y)

def get_paramgrid_lr():
  # you need to return parameter grid dictionary for use in grid search cv
  # penalty: l1 or l2
  lr_param_grid = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}

  # refer to sklearn documentation on grid search and logistic regression
  # write your code here...
  return lr_param_grid

def get_paramgrid_rf():
  # you need to return parameter grid dictionary for use in grid search cv
  # n_estimators: 1, 10, 100
  # criterion: gini, entropy
  # maximum depth: 1, 10, None  
  rf_param_grid =  {
    'n_estimators' : [50, 100],
    'max_features' : ['auto', 'sqrt','log2'],
    'max_depth' : [0, 1, 10],
    'criterion' : ['gini', 'entropy']}

  # refer to sklearn documentation on grid search and random forest classifier
  # write your code here...
  return rf_param_grid

def perform_gridsearch_cv_multimetric(model=None, param_grid=None, cv=5, X=None, y=None, metrics=['accuracy','roc_auc']):
  
  # you need to invoke sklearn grid search cv function
  # refer to sklearn documentation
  # the cv parameter can change, ie number of folds  
  
  # metrics = [] the evaluation program can change what metrics to choose
  
  grid_search_cv = GridSearchCV(estimator = model, param_grid = param_grid, cv = cv)
  # create a grid search cv object
  # fit the object on X and y input above
  new_model = grid_search_cv.fit(X,y)
  # write your code here...
  ypred=new_model.predict(X)
  # metric of choice will be asked here, refer to the-scoring-parameter-defining-model-evaluation-rules of sklearn documentation
  acc = accuracy_score(y, ypred)
  
  prec = precision_score(y, ypred, average=None)
  rec = recall_score(y, ypred, average=None)
  f1 = f1_score(y, ypred, average=None)
  roc = roc_auc_score(y, new_model.predict_proba(X), multi_class='ovr')
  
  # refer to cv_results_ dictonary
  # return top 1 score for each of the metrics given, in the order given in metrics=... list
  a = acc.max()
  b = prec.max()
  c =  f1.max()
  d =rec.max()
  e = roc.max()
  top1_scores = [a,b,c,d,e]
  
  return top1_scores



param_grid=get_paramgrid_lr()
print(param_grid)
param_rf = get_paramgrid_rf()
print(param_rf)

perform_grid = perform_gridsearch_cv_multimetric(rf_model,param_rf,5,x,y)

print(perform_grid)

###### PART 3 ######

class MyNN(nn.Module):
  def __init__(self,inp_dim=64,hid_dim=13,num_classes=10):
    super(MyNN,self)
    
    self.fc_encoder = None # write your code inp_dim to hid_dim mapper
    self.fc_decoder = None # write your code hid_dim to inp_dim mapper
    self.fc_classifier = None # write your code to map hid_dim to num_classes
    
    self.relu = None #write your code - relu object
    self.softmax = None #write your code - softmax object
    
  def forward(self,x):
    x = None # write your code - flatten x
    x_enc = self.fc_encoder(x)
    x_enc = self.relu(x_enc)
    
    y_pred = self.fc_classifier(x_enc)
    y_pred = self.softmax(y_pred)
    
    x_dec = self.fc_decoder(x_enc)
    
    return y_pred, x_dec
  
  # This a multi component loss function - lc1 for class prediction loss and lc2 for auto-encoding loss
  def loss_fn(self,x,yground,y_pred,xencdec):
    
    # class prediction loss
    # yground needs to be one hot encoded - write your code
    lc1 = None # write your code for cross entropy between yground and y_pred, advised to use torch.mean()
    
    # auto encoding loss
    lc2 = torch.mean((x - xencdec)**2)
    
    lval = lc1 + lc2
    
    return lval
    
def get_mynn(inp_dim=64,hid_dim=13,num_classes=10):
  mynn = MyNN(inp_dim,hid_dim,num_classes)
  mynn.double()
  return mynn

def get_mnist_tensor():
  # download sklearn mnist
  # convert to tensor
  X, y = None, None
  # write your code
  return X,y

def get_loss_on_single_point(mynn=None,x0,y0):
  y_pred, xencdec = mynn(x0)
  lossval = mynn.loss_fn(x0,y0,y_pred,xencdec)
  # the lossval should have grad_fn attribute set
  return lossval

def train_combined_encdec_predictor(mynn=None,X,y, epochs=11):
  # X, y are provided as tensor
  # perform training on the entire data set (no batches etc.)
  # for each epoch, update weights
  
  optimizer = optim.SGD(mynn.parameters(), lr=0.01)
  
  for i in range(epochs):
    optimizer.zero_grad()
    ypred, Xencdec = mynn(X)
    lval = mynn.loss_fn(X,y,ypred,Xencdec)
    lval.backward()
    optimzer.step()
    
  return mynn
    
