from typing import Optional

import lib.augmentations
from lib.utils import *
from lib.datasets_2 import *
from lib.test_metrics import get_standard_test_metrics
from lib.Networks import get_generator, get_discriminator
from main import *
from main_rough import *

import matplotlib.pyplot as plt
import sys
from lib import augmentations
sys.modules['augmentations'] = augmentations
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, auc
from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss
from sklearn.semi_supervised import LabelPropagation, LabelSpreading, SelfTrainingClassifier
import warnings
from math import floor


from utils import *


data = pd.read_csv("/bs140513_032310.csv") #download banksim data from https://www.kaggle.com/datasets/ealaxi/banksim1

start_ind = 5

risk_groups = {
    'super_high_risk': ["'es_leisure'", "'es_travel'"], # almost 95% to 80% frauds
    'high_risk': ["'es_sportsandtoys'", "'es_hotelservices'"], #31-50
    'risk': ["'es_otherservices'", "'es_home'", "'es_health'"], #10-25
    'medium': ["'es_tech'", "'es_wellnessandbeauty'", "'es_hyper'"], #4-7
    'medium_safe': ["'es_barsandrestaurants'", "'es_fashion'", "'es_transportation'", "'es_food'", "'es_contents'"] #0-2
}
string_to_number = {'medium_safe': 0, 'medium': 1, 'risk': 2, 'high_risk':3, 'super_high_risk':4}

# Invert the dictionary to map categories to risk levels
category_to_risk = {cat: risk for risk, categories in risk_groups.items() for cat in categories}
data = data[~data["gender"].isin(["'E'", "'U'"])]
data['risk_level'] = data['category'].map(category_to_risk)
data['risk_level_num'] = data['risk_level'].map(string_to_number)


# prepare transaction data 
import copy

d = data.drop(columns=["age", "gender", "zipcodeOri", "merchant", "zipMerchant", "category", "risk_level", "merchant_risk_level"]).copy()
grouped = d.groupby("customer")
group_tensors = [
    torch.tensor(group.iloc[ : (start_ind+i),:].drop(columns=['customer']).values, dtype=torch.float32) 
    for _, group in grouped
    for i in range(group.shape[0]-start_ind+1)    
]

x = [g[:,:2] for g in group_tensors]
labels = [g[-1,2].sum()>0 for g in group_tensors] 

labels_orig = copy.deepcopy(labels)
torch.save(x, "/data/x_transactions.pt")
torch.save(labels_orig, "/data/labels_transactions.pt")

d2 = data.copy()
d2["age"] = pd.factorize(d2["age"])[0]
d2["gender"] = pd.factorize(d2["gender"])[0]

g2 = d2.groupby("customer")
emb_list = []
for _,g in g2:
    for i in range(g.shape[0]-start_ind+1):
        g_tmp = g.iloc[ : (start_ind+i),:]
        t1 = torch.tensor([g_tmp["age"].iloc[-1], g_tmp["gender"].iloc[-1]], dtype=torch.int)
        t_tmp = torch.tensor(g_tmp["risk_level_num"].values, dtype=torch.int)
        if t_tmp.sum() == 0:
            t2 = torch.tensor([0])
        else:
            t2 = torch.tensor([torch.round((t_tmp**2).sum() / t_tmp.sum())])
        emb_list.append(torch.cat((t1, t2.to(int)), dim=0))

category_list = []
for _,g in g2:
    for i in range(g.shape[0]-start_ind):
        category_list.append(g.iloc[ -1,:]["category"])

first_fraud = [g[:-1,2].sum()==0 for g in group_tensors]

torch.save(emb_list, "/data/emb_transactions.pt")

import random

aug = [Scale(scale=2, dim=0),
    AddTime(),
    LeadLag(),
    VisiTrans()]

x_norm = normalize_data(x)
x_logsig = sig_list(x_norm, aug)
torch.save(x_logsig, "/data/x_logsig.pt")
