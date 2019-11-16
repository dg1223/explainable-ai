#### Initialize (existing) workspace and compute target ####
from azureml.core import Workspace
print("\nimported Workspace\n")
subscription_id = '65d5f74f-ace7-469d-9082-4393b3a7764b'
resource_group = 'ITAS-SCD'
workspace_name = 'ML-ITAS-SCD'
cpu_cluster_name = 'Run-Demos-Fast' # This is also the computer name

ws = Workspace(subscription_id, resource_group, workspace_name, _workspace_id='0ccb4b4c419146808a2c98b8e826aa35')
print("\ninitialized Workspace\n")
#compute_target = ws.compute_targets[compute_name]



# #### Verify that cluster does not exist already ####
# from azureml.core.compute import ComputeTarget, AmlCompute
# from azureml.core.compute_target import ComputeTargetException

# try:
    # cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    # print('Found existing cluster, use it.')
# except ComputeTargetException:
    # compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_F4S_V2',
                                                           # max_nodes=8)
    # cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)



# #### Create a run configuration for the persistent compute target ####

# from azureml.core.runconfig import RunConfiguration
# from azureml.core.conda_dependencies import CondaDependencies
# from azureml.core.runconfig import DEFAULT_CPU_IMAGE

# # Create a new runconfig object
# run_amlcompute = RunConfiguration()

# # Use the cpu_cluster you created above. 
# run_amlcompute.target = cpu_cluster

# # Enable Docker
# run_amlcompute.environment.docker.enabled = True

# # Set Docker base image to the default CPU-based image
# run_amlcompute.environment.docker.base_image = DEFAULT_CPU_IMAGE

# # Use conda_dependencies.yml to create a conda environment in the Docker image for execution
# run_amlcompute.environment.python.user_managed_dependencies = False

# # Auto-prepare the Docker image when used for execution (if it is not already prepared)
# run_amlcompute.auto_prepare_environment = True

# # Specify CondaDependencies obj, add necessary packages
# run_amlcompute.environment.python.conda_dependencies = CondaDependencies.create(conda_packages=['scikit-learn'])



# #### Create an experiment in the workspace ####
# from azureml.core import Experiment
# experiment_name = 'US-census-income-kdd'

# exp = Experiment(workspace=ws, name=experiment_name)


#### Write code to get and split your data into train and test sets here ####

# Import workspace and other libraries
#from azureml import Workspace
import os
print("\nimported os\n")
import numpy as np
print("imported numpy\n")
import pandas as pd
print("imported pandas\n")
from sklearn import preprocessing
print("imported preprocessing\n")
from sklearn.preprocessing import Imputer
print("imported Imputer\n")
from sklearn import svm
print("imported svm\n")
from sklearn.model_selection import train_test_split
print("imported train_test_split\n")
from sklearn.preprocessing import normalize
print("imported normalize\n")
from azureml.core.datastore import Datastore
print("imported Datastore\n")
from azureml.core.dataset import Dataset
print("imported Dataset\n")


#ws = Workspace()
#ds = ws.datasets['census-income-data_all']
#print("initialized dataset\n")
#df = ds.to_pandas_dataframe()
#print("converted dataset to dataframe\n")

script_folder = os.getcwd()
print(script_folder + '\n')
filepath = script_folder + '/census-income-data_all.csv'
print(filepath + '\n')

df = pd.read_csv(filepath)
print("converted dataset to dataframe\n")

## Start preprocessing: Clean up data and normalize ##
print("preprocessing data...\n")
# Replace missing data
df_clean = df.replace('?', np.nan)
print("preprocessing complete\n")

# Separate labels from measures
print("separating label from dataset\n")
y_raw = df_clean.pop('Income_<>_$50000')
y = y_raw.values.reshape((y_raw.shape[0],1))
X = df_clean
print("label separation complete\n")

# Drop columns with too many missing values (all migration-related columns)
print("dropping columns with too many missing values\n")
X.drop(X.columns[[25,26,27,29]], axis = 1, inplace = True)
print("dropped all migration-related columns\n")

# Define feature names and label/classes
print("defining feature names and classes\n")
feature_names = np.array(["Age", "Class Of Worker", "Industry Code", "Occupation Code", "Education", 
						"Wage Per Hour x 100", "Enrolled In Educational Inst.", "Marital Status", "Major Industry Code", 
						"Major Occupation Code", "Race", "Hispanic Origin", "Sex", "Member Of A Labour Union", 
						"Reason For Unemployment", "Full Or Part-Time", "Capital Gains", "Capital Losses", 
						"Divdends From Stocks", "Tax Filer Status", "Region Of Previous Residence", 
						"State Of Previous Residence", "Detailed Household Stat", "Detailed Household Summary", 
						"Instance Weight", "Lived In This House 1 Year Ago", "Num Persons Worked For Employer", 
						"Family Members Under 18", "Country Of Birth - Father", "Country Of Birth - Mother", 
						"Country Of Birth - Self", "Citizenship", "Self Employed", 
						"Fill Inc Questionnaire For Veteran's Admin", "Veterans Benefits", "Weeks Worked In Year", "Year of Survey"])
						
labels = np.array(['-50000', '50000'])
print("features and classes have been defined\n")

# Encode categorical values but preserve nulls
print("encoding categorical values with label encoder\n")
le = preprocessing.LabelEncoder()
X = X.apply(lambda series: pd.Series(le.fit_transform(series[series.notnull()]), index=series[series.notnull()].index))
print("label encoding complete\n")

# Impute missing values using univariate imputer (multivariate is not supported in this version of Scikit-learn 0.18.1)
print("imputing missing values as nan\n")
imp = Imputer(missing_values=np.nan, strategy='most_frequent')
imp.fit(X)
X_clean = imp.transform(X)

imp.fit(y)
y_clean = imp.transform(y)
print("imputation complete\n")

# Flatten label column to fit the SVM classifier requirement (otherwise you get warnings)
y_flat = y_clean.ravel()
print("flattened class label array\n")

# Normalize dataset (exclude labels)
X_norm = normalize(X_clean)
print("normalized dataset\n")

#### write code to train your model here ####

# Use SVM to train the model by splitting data into training and test set 80-20
x_train,x_test,y_train,y_test = train_test_split(X_norm, y_flat, test_size=0.2)
print("split dataset into training and test sets with an 80-20 partition\n")
clf = svm.SVC(gamma=0.001, C=100, probability=True)
print("initialized SVM classifier\n")
print("fitting the training set into the classifier now\n")
model = clf.fit(x_train, y_train)



#### explain predictions on your local machine ####
# "features" and "classes" fields are optional
print("started model explanation \n")
from azureml.explain.model.tabular_explainer import TabularExplainer

explainer = TabularExplainer(model, 
                            x_train, 
                            features=feature_names, 
                            classes=labels)
print("initialized explainer\n")

#### Explain overall model predictions (global explanation) ####
print("initializing global explainer\n")
global_explanation = explainer.explain_global(x_test, batch_size=200)

# uploading global model explanation data for storage or visualization in webUX
# the explanation can then be downloaded on any compute
# multiple explanations can be uploaded
print("uploading global explainer\n")
client.upload_model_explanation(global_explanation, comment='global explanation: all features')
print("uploaded global explainer\n")
# or you can only upload the explanation object with the top k feature info
#client.upload_model_explanation(global_explanation, top_k=2, comment='global explanation: Only top 2 features')


#### Explain each data point in the test set ####
print("initializing local explainer\n")
local_explanation = explainer.explain_local(x_test)

# uploading local model explanation data for storage or visualization in webUX
print("uploading local explainer\n")
client.upload_model_explanation(local_explanation, comment='local explanation: all features')
print("uploaded local explainer\n")
