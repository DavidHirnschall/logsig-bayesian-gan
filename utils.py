from typing import Optional

import matplotlib.pyplot as plt
import sys
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

import os
import math
import time
import copy
import random
import pickle
import warnings
import itertools
from abc import abstractmethod
from functools import partial, reduce
from collections import defaultdict
from statistics import median
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils as ut
from torch import optim, autograd
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm
import wfdb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc
import pandas_datareader as pdr
import yfinance as yf
import signatory
from fbm import fbm, MBM
#from torchmetrics.classification import BinaryF1Score, F1Score
from multiprocess import Process
from dataclasses import dataclass

# Reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

import numpy.random as random
def train_test_list(labels, rate, random_state=123):
    np.random.seed(random_state)
    non_zero_ind = labels.nonzero()[:,0]
    l = labels.clone() - 1
    zero_ind = l.nonzero()[:,0]
    zero_ind, non_zero_ind

    zi_test = random.choice(zero_ind, size = int(rate*  len(zero_ind)), replace=False)
    zi_train = zero_ind[~np.isin(zero_ind, zi_test)]
    nzi_test = random.choice(non_zero_ind, size = int(rate * len(non_zero_ind)), replace=False)
    nzi_train = non_zero_ind[~np.isin(non_zero_ind, nzi_test)]
    train_ind = np.append(zi_train, nzi_train)
    test_ind = np.append(zi_test, nzi_test)

    return train_ind, test_ind

def normalize_data(list_in, m0=None, m1=None):
        if m0 == None:
            m0 = max([l[:, 0].diff().max() for l in list_in])
        if m1 == None:
            m1 = max([l[:, 1].max() for l in list_in])
        x_norm = []
        for t in list_in:
            updated_tensor = t.clone()
            updated_tensor[1:, 0] = updated_tensor[:, 0].diff()/m0
            updated_tensor[1:, 1] = (updated_tensor[:, 1]/m1)[1:]
            x_norm.append(updated_tensor[1:,:])
        return x_norm


from functools import reduce
def sig_list(x, augmentations, depth=3, log=True):
    if log==True:
        y = [signatory.logsignature(
            reduce(lambda y, a: a.apply(y), augmentations, xi.clone().unsqueeze(0)), depth=depth)
            for xi in x]
    else:
        y = [signatory.signature(
            reduce(lambda y, a: a.apply(y), augmentations, xi.clone().unsqueeze(0)), depth=depth)
            for xi in x]
    return torch.stack(y).squeeze(1)

from sklearn.metrics import roc_auc_score
def wauc(y_test, y_scores, classes=[0, 1]):
    #y_test_bin = label_binarize(y_test, classes=classes)

    y_test_bin = np.eye(len(classes))[y_test].astype(int)
    # Compute AUC for each class
    auc_per_class = []
    for i in range(y_test_bin.shape[1]):
        auc_value = roc_auc_score(y_test_bin[:, i], y_scores[:, i])
        auc_per_class.append(auc_value)

    # Compute weights (proportion of each class in y_true)
    class_counts = np.bincount(y_test)
    class_weights = class_counts / len(y_test)

    # Compute weighted AUC
    weighted_auc = np.sum(np.array(auc_per_class) * class_weights)
    return weighted_auc

from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, auc
def wprauc(y_test, y_scores, classes=[0, 1]):
    
    y_test_bin = np.eye(len(classes))[y_test].astype(int)
    # Compute AUC for each class
    auc_per_class = []
    for i in range(y_test_bin.shape[1]):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_scores[:, i])
        auc_per_class.append(auc(recall, precision))
        

    # Compute weights (proportion of each class in y_true)
    class_counts = np.bincount(y_test)
    class_weights = class_counts / len(y_test)

    # Compute weighted AUC
    weighted_auc = np.sum(np.array(auc_per_class) * class_weights)
    return weighted_auc

# additional metrics
def precision_at_k(y_true, y_score, k_percent):
    """
    k_percent: e.g. 0.1 for top 0.1% of cases.
    Returns precision, K count, TP count.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    n_pos = int(y_true.sum())
    k = max(1, int(np.ceil(n * (k_percent / 100.0))))
    order = np.argsort(-y_score)  # descending
    top_idx = order[:k]
    tp = y_true[top_idx].sum()
    precision = tp / k
    recall = tp / n_pos if n_pos > 0 else np.nan
    return precision, k, tp, recall

def precision_table(y_true, y_score, k_list=(0.05, 0.1, 0.2, 0.5, 1.0)):
    rows = []
    ps = []
    """for k in k_list:
        p, k_count, tp = precision_at_k(y_true, y_score, k)
        ps.append(p)
        rows.append({"K%": k, "Top K (count)": k_count, "TP in Top K": int(tp), "Precision@K": p})
    #return pd.DataFrame(rows)
    df = pd.DataFrame(ps, columns=k_list)
    return df"""
    ps, rs = {}, {}
    for k in k_list:
        p, k_count, tp, r = precision_at_k(y_true, y_score, k)
        ps[f"Prec@{k}"]   = p
        rs[f"Recall@{k}"] = r

    row = {**ps, **rs}
    return pd.DataFrame([row])

from sklearn.utils import resample
from sklearn.metrics import average_precision_score
def bootstrap_pr_auc(y_true, y_score, n_boot=1000, seed=42):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    boot_scores = []
    n = len(y_true)

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_scores.append(average_precision_score(y_true[idx], y_score[idx]))

    boot_scores = np.array(boot_scores)
    mean = boot_scores.mean()
    lo, hi = np.percentile(boot_scores, [2.5, 97.5])
    return mean, lo, hi, boot_scores

def tail_reliability_plot(y_true, y_score, top_percent=1.0, n_bins=10, title=None):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    k = max(1, int(np.ceil(n * (top_percent / 100.0))))
    order = np.argsort(-y_score)
    top_idx = order[:k]
    y_top = y_true[top_idx]
    s_top = y_score[top_idx]

    # bin the top scores
    bins = np.linspace(s_top.min(), s_top.max(), n_bins + 1)
    bin_ids = np.digitize(s_top, bins) - 1
    xs, ys, counts = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if np.any(mask):
            xs.append(s_top[mask].mean())          # avg predicted prob in bin
            ys.append(y_top[mask].mean())          # observed fraud rate in bin
            counts.append(mask.sum())

    plt.figure(figsize=(5,4))
    plt.plot([0,1],[0,1], linestyle='--')
    plt.scatter(xs, ys, s=np.array(counts))  # marker size ~ bin count
    plt.xlabel("Predicted probability (bin average)")
    plt.ylabel("Observed fraud rate (bin average)")
    if title is None:
        title = f"Tail Reliability (Top {top_percent}% by score)"
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
from sklearn.metrics import precision_recall_curve

def expected_cost_curve(y_true, y_score, amounts, fp_pct=0.02):
    """
    y_true: 0/1 labels
    y_score: predicted probabilities
    amounts: transaction amount per record
    fp_pct: fraction of amount used as FP (ops/customer) cost
    Returns a DataFrame with threshold and expected cost, plus the argmin row.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    amounts = np.asarray(amounts)

    # Use unique thresholds from scores (or PR curve thresholds)
    precision, recall, thr = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns thresholds for all but the first point
    thresholds = np.r_[0.0, thr, 1.0]  # include 0 and 1 boundaries

    rows = []
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        # Missed frauds (FN): y=1 but pred=0 -> cost = amount
        fn_mask = (y_true == 1) & (pred == 0)
        fn_cost = amounts[fn_mask].sum()

        # False positives (FP): y=0 but pred=1 -> cost = fp_pct * amount
        fp_mask = (y_true == 0) & (pred == 1)
        fp_cost = (fp_pct * amounts[fp_mask]).sum()

        total_cost = fn_cost + fp_cost
        rows.append({"threshold": t, "FN_cost": fn_cost, "FP_cost": fp_cost, "Total_cost": total_cost})

    df = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    best_row = df.loc[df["Total_cost"].idxmin()]
    return df, best_row

def expected_cost_at_k(y_true, y_score, amounts, k_percent, fp_pct=0.02, fn_pct=1.0):
    n = len(y_true)
    k = max(1, int(np.ceil(n * (k_percent / 100.0))))
    order = np.argsort(-y_score)
    top = order[:k]
    rest = order[k:]
    fp_cost = (fp_pct * amounts[(y_true==0) & np.isin(np.arange(n), top)]).sum()
    fn_cost = fn_pct * amounts[(y_true==1) & np.isin(np.arange(n), rest)].sum()
    return fp_cost + fn_cost

def expected_cost_table(y_true, y_score, amounts, k_percent=(0.1, 0.2, 0.5, 1.0), fp_pct=0.02, fn_pct=1.0):
    cs = {}
    for k in k_percent:
        c = expected_cost_at_k(y_true, y_score, amounts, k, fp_pct, fn_pct)
        cs[k] = c
        
    return pd.DataFrame([cs])

def partial_pr_auc_at_k(y_true, y_score, k_percent):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    n = len(y_true)
    k = int(np.ceil(n * (k_percent / 100.0)))
    order = np.argsort(-y_score)
    cutoff = y_score[order[k-1]]

    # recall at that cutoff
    pred = (y_score >= cutoff).astype(int)
    recall_at_k = ( (pred & (y_true==1)).sum() ) / y_true.sum()

    # mask PR curve up to recall_at_k
    mask = recall <= recall_at_k
    return auc(recall[mask], precision[mask]), recall_at_k

def partial_pr_auc_at_k_table(y_true, y_score, k_percent=(0.1, 0.2, 0.5, 1.0)):
    cs, rs = {}, {}
    for k in k_percent:
        c, r = partial_pr_auc_at_k(y_true, y_score, k)
        cs[f"PrAUC@{k}"]   = c
        rs[f"Recall@{k}"] = r

    row = {**cs, **rs}
    return pd.DataFrame([row])

def _ensure_point_at_recall_cap(prec, rec, cap, eps=1e-12):
    """
    Clip PR curve to end exactly at recall=cap by linear interpolation.
    Returns arrays with matching lengths.
    """
    m = rec <= cap + eps
    prec_clip = prec[m].astype(float).copy()
    rec_clip  = rec[m].astype(float).copy()

    if rec_clip.size == 0:
        # fallback: one point at (cap, first precision)
        return np.array([prec[0]], dtype=float), np.array([cap], dtype=float)

    if np.isclose(rec_clip[-1], cap, atol=eps):
        return prec_clip, rec_clip

    idx_above = np.searchsorted(rec, cap, side="right")
    if idx_above >= len(rec):
        # extend horizontally
        return np.r_[prec_clip, prec_clip[-1]], np.r_[rec_clip, cap]

    r0, r1 = rec[idx_above-1], rec[idx_above]
    p0, p1 = prec[idx_above-1], prec[idx_above]
    if np.isclose(r1, r0, atol=eps):
        p_cap = p1
    else:
        alpha = (cap - r0) / (r1 - r0)
        p_cap = p0 + alpha * (p1 - p0)

    prec_clip = np.r_[prec_clip, p_cap]
    rec_clip  = np.r_[rec_clip,  cap]
    return prec_clip, rec_clip

def _monotonicize_for_auc(rec, prec):
    """Sort by recall asc and collapse duplicate recall values (keep last precision)."""
    df = pd.DataFrame({"rec": rec, "prec": prec}).sort_values("rec", kind="mergesort")
    df = df.groupby("rec", as_index=False).last()
    return df["rec"].to_numpy(), df["prec"].to_numpy()

def partial_pr_auc_to_recall(y_true, y_score, recall_cap):
    """
    Compute partial area under the PR curve up to a FIXED recall_cap in [0,1].
    Returns:
      pAUC     : area under P-R from recall=0 to recall=recall_cap
      head_AP  : normalized pAUC / recall_cap  (average precision in the head)
      recall_cap: echoed (float)
    """
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    pos = int(y_true.sum())
    if pos == 0 or len(y_true) == 0 or recall_cap <= 0:
        return 0.0, 0.0, float(max(recall_cap, 0.0))

    precision, recall, _ = precision_recall_curve(y_true, y_score)

    # clip / interpolate the curve to end exactly at recall_cap
    prec_clip, rec_clip = _ensure_point_at_recall_cap(precision, recall, float(recall_cap))
    rec_clip, prec_clip = _monotonicize_for_auc(rec_clip, prec_clip)  # <-- important
    
    # integrate area from 0 to recall_cap
    pAUC = float(auc(rec_clip, prec_clip))
    head_AP = float(pAUC / max(recall_cap, 1e-12))
    return pAUC#, float(recall_cap)#, head_AP

def partial_pr_auc_to_recall_table(y_true, y_score, recall_cap=[0.5, 0.6, 0.7, 0.8]):
    cs = {}
    for k in recall_cap:
        c = partial_pr_auc_to_recall(y_true, y_score, k)
        cs[k]   = c

    return pd.DataFrame([cs])

def stack_model_table(df_mean, model_name, metric=None):
    """
    df_mean: rows = train sizes (0..N-1), columns like 'Prec@0.1', 'Recall@1.0'
    returns long df: model, train_size, metric, K, value
    """
    df = df_mean.copy()
    df = df.reset_index(drop=True).rename_axis("train_size").reset_index()
    long = df.melt(id_vars="train_size", var_name="col", value_name="value")
    # split 'Prec@0.1' -> metric='Prec', K='0.1'
    
    if metric == None:
        m = long["col"].str.extract(r'^(?P<metric>[^@]+)@(?P<K>.*)$')
        long["metric"] = m["metric"]
        long["K"] = m["K"].astype(float)
    else:
        long["metric"] = metric
        long["K"] = long["col"].astype(float)
    long["model"] = model_name
    return long[["model","train_size","metric","K","value"]]

def main_table_at_K(tidy, metric, K, model_order):
    sub = tidy[(tidy["metric"]==metric) & (tidy["K"]==K)]
    pt = (sub.pivot(index="train_size", columns="model", values="value")
              .reindex(columns=model_order).sort_index())
    return pt.round(3)

def appendix_table_metric(tidy, metric, model_order, K_order=(0.1,0.2,0.5,1.0)):
    sub = tidy[tidy["metric"]==metric].copy()
    sub["K"] = pd.Categorical(sub["K"], categories=K_order, ordered=True)
    sub = sub.sort_values(["K","train_size","model"])
    # build multirow-like view (K then train_size then models)
    pt = (sub.pivot_table(index=["K","train_size"], columns="model", values="value", aggfunc="mean")
              .reindex(columns=model_order))
    return pt.round(3)  

def save_results(df, path, loss):
    pd.concat(df).to_csv(path + "/" + loss + "_all.csv")
    mean_rs = pd.concat(df).groupby(level=0).mean().round(decimals=3)
    mean_rs.to_csv(path + "/" + loss + ".csv")
    sd_rs = pd.concat(df).groupby(level=0).std().round(decimals=3)
    min_rs = pd.concat(df).groupby(level=0).min().round(decimals=3)
    max_rs = pd.concat(df).groupby(level=0).max().round(decimals=3)
    np.random.seed(123)  #

    mean_rs.set_index("N_l", inplace=True)
    #mean_rs.set_index("Size", inplace=True)

    pl = [1,2,3,4,5,6]#,4,5]
    cols = [1,4,5,6,8,10,11]
    jitter = np.random.uniform(-.2, .2, size=len(cols)) #[1, 6, 7,8,9,10,11,12,13]))          #[4,8,11,12,15,17,18]))#[2,4,5,7,9]))
    for j,column in enumerate(mean_rs.columns[cols]):  #[1, 6, 7,8,9,10,11,12,14]]):#                [[4,8,11,12,15,17,18]]):#[[2,4,5,7,9]]):#[[1,3,4,5,6,7,8]]:#[[2,4,5,6,7,8,9]]:#[1:]:
        i=0
    
        plt.errorbar(
            pl + jitter[j],      # Use the index of the DataFrame as the x-axis
            mean_rs[column],    # Data for the y-axis from df1
            yerr=sd_rs[column],  # Error data from df2
            label=column,   # Column name for the legend
            capsize=3,      # Caps on the error bars
            marker='o',     # Marker for the points on the line
            linestyle='-',  # Line style
            linewidth=1     # Line width

        )
        i=i+1
    plt.xticks(pl, mean_rs.index.astype(int))
    plt.xlabel('number of labeled data')
    plt.ylabel(loss)
    #plt.title('Line Plots with Error Bars')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.title("Fraud Detection")
    plt.tight_layout()
    plt.savefig(path + "/" + loss + ".png")
    plt.show()
    
# Silence warnings
warnings.filterwarnings("ignore")
from os import path as pt

def set_seed(seed: int):
    """ Sets the seed to a specified value. Needed for reproducibility of experiments. """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

def sample_indices(dataset_size, batch_size):
    indices = torch.from_numpy(np.random.choice(dataset_size, size=batch_size, replace=False))#.cuda()
    # functions torch.-multinomial and torch.-choice are extremely slow -> back to numpy
    return indices.long()

def to_numpy(x):
    return x.detach().cpu().numpy()

def plot_signature(sig):
    plt.plot(to_numpy(sig).T, 'o')

def init_weights(m, normal = False, gain=1):
    if isinstance(m, nn.Linear):
        if normal == True:
            nn.init.normal(m.weight.data, np.random.normal(0, 0.01), 0.02)    #sigma größer für SBGAN #0.02 original
        else:
            nn.init.xavier_normal_(m.weight.data, gain=gain)#nn.init.calculate_gain('leaky_relu'))
        try:
            nn.init.zeros_(m.bias.data)#m.bias.zero_()#, gain=nn.init.calculate_gain('relu'))
        except:
            pass

def init_weights_d(gain):
    def initializer(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data, gain=gain)#nn.init.calculate_gain('leaky_relu'))
        try:
            nn.init.zeros_(m.bias.data)#m.bias.zero_()#, gain=nn.init.calculate_gain('relu'))
        except:
            pass
    return initializer


def get_config_path(config, dataset):
    return '/configs/{config}.json'.format(config=config, dataset=dataset)
    

def get_config_path_generator(config, dataset):
    return '/configs/{config}.json'.format(
        dataset=dataset, config=config)


def get_config_path_discriminator(config, dataset):
    return '/configs/{config}.json'.format(config=config, dataset=dataset)
    

def get_wgan_experiment_dir(dataset, discriminator, generator, gan, seed):
    return '/configs/{gan}_{generator}_{discriminator}_{seed}'.format(
        dataset=dataset, gan=gan, generator=generator, discriminator=discriminator, seed=seed)
    

def save_obj(obj: object, filepath: str):
    if filepath.endswith('pkl'):
        saver = pickle.dump
    elif filepath.endswith('pt'):
        saver = torch.save
    else:
        raise NotImplementedError()
    with open(filepath, 'wb') as f:
        saver(obj, f)
    return 0


def load_obj(filepath):
    if filepath.endswith('pkl'):
        loader = pickle.load
    elif filepath.endswith('pt'):
        loader = torch.load
    elif filepath.endswith('json'):
        import json
        loader = json.load
    else:
        raise NotImplementedError()
    with open(filepath, 'rb') as f:
        if filepath.endswith('pt'):
            return torch.load(f, map_location=torch.device('cpu'))
        else:
            return loader(f)
        

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def compute_grad2(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = grad_dout2.view(batch_size, -1).sum(1)
    return reg

def sig_list(x, augmentations, depth=3, log=True):
    if log==True:
        y = [signatory.logsignature(
            reduce(lambda y, a: a.apply(y), augmentations, xi.clone().unsqueeze(0)), depth=depth)
            for xi in x]
    else:
        y = [signatory.signature(
            reduce(lambda y, a: a.apply(y), augmentations, xi.clone().unsqueeze(0)), depth=depth)
            for xi in x]
    return torch.stack(y).squeeze(1)


def get_time_vector(size: int, length: int):
    return torch.linspace(0, 1, length).reshape(1, -1, 1).repeat(size, 1, 1)


def lead_lag_transform(x: torch.Tensor):
    x_rep = torch.repeat_interleave(x, repeats=2, dim=1)
    x_ll = torch.cat([x_rep[:, :-1], x_rep[:, 1:]], dim=2)
    return x_ll

def I_visibility_transform(path: torch.Tensor):
    init_tworows_ = torch.zeros_like(path[:,:1,:])
    init_tworows = torch.cat((init_tworows_, path[:,:1,:]), axis=1)

    a = torch.cat((init_tworows, path), axis=1)

    last_col1 = torch.zeros_like(path[:,:2,:1])
    last_col2 = torch.cat((last_col1, torch.ones_like(path[:,:,:1])), axis=1)

    output = torch.cat((a, last_col2), axis=-1)
    return output

@dataclass
class BaseAug:
    pass

    def apply(self, *args: List[torch.Tensor]):
        raise NotImplementedError('Needs to be implemented by child.')

@dataclass
class Scale(BaseAug):
    scale: float = 1
    dim: int = None

    def apply(self, x: torch.Tensor):
        if self.dim == None:
            return self.scale * x
        else:
            x[...,self.dim] = self.scale * x[...,self.dim]
            return x

@dataclass
class AddTime(BaseAug):
    def apply(self, x: torch.Tensor):
        t = get_time_vector(x.shape[0], x.shape[1]).to(x.device)
        return torch.cat([t, x], dim=-1)

@dataclass
class LeadLag(BaseAug):
    def apply(self, x: torch.Tensor):
        return lead_lag_transform(x)

@dataclass
class VisiTrans(BaseAug):
    def apply(self, x: torch.Tensor):
        return I_visibility_transform(x)
        
def apply_augmentations(x: torch.Tensor, augmentations: Tuple):
    y = x.clone()
    for augmentation in augmentations:
        y = augmentation.apply(y)
    return y

AUGMENTATIONS = {'AddTime': AddTime, 'LeadLag': LeadLag,
        'Scale': Scale,  'VisiTrans': VisiTrans}

def parse_augmentations(list_of_dicts):
    augmentations = list()
    for kwargs in list_of_dicts:
        name = kwargs.pop('name')
        augmentations.append(
            AUGMENTATIONS[name](**kwargs)
        )
    return augmentations


import torch

@torch.no_grad()
def stratified_group_train_test_split_torch(
    y: torch.Tensor,                # shape [N], 0/1 (binary); can extend to multi-class
    groups: torch.Tensor,           # shape [N], group ids (e.g., customer ids)
    test_size: float = 0.2,
    generator: torch.Generator = None,  # for tie-breaking randomness
):
    """
    Greedy splitter: stratifies by labels while enforcing group exclusivity.
    Works entirely in torch (no numpy).

    Returns
    -------
    train_idx : LongTensor
    test_idx  : LongTensor
    """
    assert y.dim() == 1 and groups.dim() == 1 and y.numel() == groups.numel()
    assert 0.0 < test_size < 1.0, "test_size must be in (0,1)."

    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(42)

    # Ensure 1D long tensors
    y = y.view(-1).long()
    groups = groups.view(-1)

    # Unique groups and inverse index per sample
    uniq_groups, inv = torch.unique(groups, return_inverse=True)
    G = uniq_groups.numel()

    # Count per-group sizes and positives
    n_per_group = torch.bincount(inv, minlength=G)                      # total samples per group
    pos_per_group = torch.bincount(inv, weights=y.float(), minlength=G) # positives per group (float)
    pos_per_group = pos_per_group.round().long()

    total_n = int(n_per_group.sum().item())
    total_pos = int(pos_per_group.sum().item())

    target_n_test = int(round(test_size * total_n))
    target_pos_test = int(round(test_size * total_pos))

    # Order groups by "heaviness" (max of pos vs neg in that group), descending
    neg_per_group = n_per_group - pos_per_group
    heaviness = torch.maximum(pos_per_group, neg_per_group)
    order = torch.argsort(-heaviness)

    # Greedy assignment to test
    test_n = 0
    test_pos = 0
    is_test_group = torch.zeros(G, dtype=torch.bool)

    # Helper: random tie-break in torch
    def coin_flip():
        return torch.rand((), generator=generator).item() < 0.5

    for g in order.tolist():
        cand_test_n = test_n + int(n_per_group[g].item())
        cand_test_pos = test_pos + int(pos_per_group[g].item())

        err_if_test = (cand_test_n - target_n_test) ** 2 + (cand_test_pos - target_pos_test) ** 2
        err_if_train = (test_n - target_n_test) ** 2 + (test_pos - target_pos_test) ** 2

        if (err_if_test < err_if_train) or (err_if_test == err_if_train and coin_flip()):
            is_test_group[g] = True
            test_n, test_pos = cand_test_n, cand_test_pos

    # Map groups back to samples
    is_test_sample = is_test_group[inv]
    test_idx = torch.nonzero(is_test_sample, as_tuple=True)[0]
    train_idx = torch.nonzero(~is_test_sample, as_tuple=True)[0]

    return train_idx.long(), test_idx.long()