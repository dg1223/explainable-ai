from azureml.core import Workspace
print("\nimported Workspace\n")

from azureml.core.compute import ComputeTarget, AmlCompute
print("\nimported ComputeTarget, AmlCompute\n")
from azureml.core.compute_target import ComputeTargetException
print("\nimported ComputeTargetException\n")

from azureml.core.runconfig import RunConfiguration
print("\nimported RunConfiguration\n")
from azureml.core.conda_dependencies import CondaDependencies
print("\nimported CondaDependencies\n")
from azureml.core.runconfig import DEFAULT_CPU_IMAGE
print("\nimported DEFAULT_CPU_IMAGE\n")

from azureml.core import Experiment
print("\nimported Experiment\n")

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

import azureml
print("imported azureml\n")

from azureml import explain
print("imported explain\n")

from azureml.explain import model
print("imported model\n")

from azureml.explain.model import tabular_explainer
print("imported tabular_explainer\n")

from azureml.explain.model.tabular_explainer import TabularExplainer
print("imported TabularExplainer\n")