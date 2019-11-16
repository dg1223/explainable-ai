
# load breast cancer dataset, a well-known small dataset that comes with scikit-learn
from sklearn.datasets import load_breast_cancer
from sklearn import svm
from sklearn.model_selection import train_test_split
breast_cancer_data = load_breast_cancer()
classes = breast_cancer_data.target_names.tolist()

# split data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(breast_cancer_data.data,            
                                                    breast_cancer_data.target,  
                                                    test_size=0.2,
                                                    random_state=0)
clf = svm.SVC(gamma=0.001, C=100., probability=True)
model = clf.fit(x_train, y_train)

import pickle
# save the model to disk
modelname = 'finalized_svm_model.sav'
pickle.dump(model, open(modelname, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(modelname, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)
