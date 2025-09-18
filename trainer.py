import os
import sys
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
from sklearn.metrics import roc_auc_score
# Reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Silence warnings
warnings.filterwarnings("ignore")
from os import path as pt

from utils import *
from networks import *
from losses import *


DATA_DIR = 'datasets'

GENERATORS = {'ResFNN_gen': ResFNNGenerator}

def get_generator(generator_type, input_dim, output_dim, **kwargs):
    print(GENERATORS)
    return GENERATORS[generator_type](input_dim=input_dim, output_dim=output_dim, **kwargs)

DISCRIMINATORS = {'ResFNN': ResFNNDiscriminator}

def get_discriminator(discriminator_type, input_dim, emb=False, **kwargs):
    return DISCRIMINATORS[discriminator_type](input_dim=input_dim, emb=emb, **kwargs)

class BaseTrainer:
    def __init__(self, batch_size, G, G_optimizer, n_gradient_steps, foo=lambda x: x):
        self.batch_size = batch_size

        self.G = G
        self.G_optimizer = G_optimizer
        self.n_gradient_steps = n_gradient_steps

        self.losses_history = defaultdict(list)
        self.foo = foo
        
        self.init_time = time.time()

        self.best_G = copy.deepcopy(G.state_dict())
        self.best_cov_err = None
        

class SBGANTrainer(BaseTrainer):
    def __init__(self, netDs, netGs, discriminator_steps_per_generator_step,
            lr_discriminator, lr_generator, depth, x_real: torch.Tensor, len_noise, input_dim, 
            normalise_sig: bool = True, x_labelled = None, x_labelled_sig = None, netD_fullsup = None, netD_fullsup_direct = None, 
            x_fake_test_labelled = None, x_fake_test_labelled_sig = None, reg_param=10., gp = True, lr_g=None, lr_d=None, l_errsup=1, 
            labels=None, labels_test=None, acts=None, acts_labelled=None, acts_test=None, generator_type=None,
            bs=None,x_sig=None, x_logsig=None,
             **kwargs):
        if kwargs.get('augmentations') is not None:
            self.augmentations = kwargs['augmentations']
            del kwargs['augmentations']
        else:
            self.augmentations = None
        self.augmentations = [Scale(scale=2, dim=0),
            AddTime(),
            LeadLag(),
            VisiTrans()]

        self.depth = depth
        self.normalise_sig = normalise_sig
        self.numz = kwargs["numz"]
        del kwargs['numz']
        self.numD = kwargs["numD"]
        del kwargs['numD']
        self.num_mcmc = kwargs["num_mcmc"]
        del kwargs['num_mcmc']
        self.bayes = kwargs["bayes"]
        del kwargs['bayes']
        self.sup = kwargs["sup"]
        del kwargs['sup']
        self.num_classes = kwargs["num_classes"]
        del kwargs['num_classes']
        self.gnoise_alpha = kwargs["gnoise_alpha"]
        del kwargs['gnoise_alpha']
        self.dnoise_alpha = kwargs["dnoise_alpha"]
        del kwargs['dnoise_alpha']
        self.wasserstein = kwargs["wasserstein"]
        del kwargs['wasserstein']
        self.len_noise = len_noise
        self.input_dim = input_dim
        self.l_errsup = l_errsup
        self.generator_type = generator_type
        
        self.stateG, self.stateD = [], []
        self.stateD_max_wauc = None
        self.labels = labels
        self.labels_test = labels_test
        self.acts = acts
        self.acts_labelled = acts_labelled
        self.acts_test = acts_test
        
        self.start_time_loop = time.perf_counter() 
        self.n_gradient_steps = 2 #1000 #kwargs["n_gradient_steps"]  
        x_size = len(x_real)
        
        if x_size < 1024:#kwargs["batch_size"]:
            self.batch_size = x_size
        else:
            self.batch_size = 1024 
        self.batch_size = bs 

        self.losses_history = defaultdict(list)
        
        x = x_real[0].unsqueeze(0)
        self.n_lags = x_labelled_sig.shape[1] + acts_labelled.shape[1]
        
        self.netGs = netGs
        self.netDs = netDs
        self.gnoise_criterion = []
        self.dnoise_criterion = []
        self.expected_signature_nu = [None] * len(self.netGs)
        self.optimizerGs = []
        self.bayes = True 
        
        if lr_d != None:
            lr_discriminator = lr_d
        else:
            lr_discriminator = 0.1
            
        if lr_g != None:
            lr_generator = lr_g
        else:
            lr_generator = 0.1
            
        self.sig = True
    
        self.scheduler_G = []
        
        for netG in self.netGs:
            G_optimizer = torch.optim.Adam(netG.parameters(), lr=lr_generator, betas=(0.0, 0.99), weight_decay=1e-4) 
            self.optimizerGs.append(G_optimizer)
            
        self.D_steps_per_G_step = 5 
        
        self.optimizerDs = []
        self.optimizerDs_sup = []
        self.scheduler = []
        for netD in self.netDs:
            D_optimizer = torch.optim.Adam(netD.parameters(), lr=lr_discriminator, betas=(0.0, 0.99), weight_decay=1e-4) 
            self.optimizerDs.append(D_optimizer)
            self.scheduler.append(torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerDs[-1],factor=.8,patience=200,verbose=True))

        # fully supervised
        self.netD_fullsup = netD_fullsup
        self.netD_fullsup_direct = netD_fullsup_direct
        
        if netD_fullsup != None:
            self.criterion_fullsup = nn.BCEWithLogitsLoss()
            self.optimizerD_fullsup = torch.optim.Adam(netD_fullsup.parameters(), lr=lr_discriminator, betas=(0.9, 0.999), weight_decay=1e-4)
            self.scheduler_fullsup = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizerD_fullsup,factor=.8,patience=100,verbose=True)

        # quasi out of sample data (real and generated withother model)
        if x_fake_test_labelled != None:
            if type(x_fake_test_labelled) == list:
                self.x_fake_test_input, self.x_fake_test_target = x_fake_test_labelled, self.labels_test
            else:
                self.x_fake_test_input, self.x_fake_test_target = x_fake_test_labelled[:,0:-1], x_fake_test_labelled[:,-1] 
        
        self.reg_param = reg_param
        self.gp = gp
        self.x_real, self.x_labelled = x_real, x_labelled
        self.x_labelled_sig = x_labelled_sig
        self.x_fake_test_labelled = x_fake_test_labelled
        self.x_fake_test_labelled_sig = x_fake_test_labelled_sig
        self.init_time = time.time()
        self.best_G = [copy.deepcopy(self.netGs[i].state_dict()) for i in range(len(self.netGs))]
        self.best_cov_err = [None] * len(self.netGs)

        self.lr_generator = lr_generator
        self.lr_discriminator = lr_discriminator

        
        if x_logsig != None:
            self.x_real_signature = x_logsig
        else:
            self.x_real_signature = sig_list(self.x_real, self.augmentations)
        self.min_vals = self.x_real_signature.min(dim=0, keepdim=True).values
        self.max_vals = self.x_real_signature.max(dim=0, keepdim=True).values
        
        self.expected_signature_mu = self.x_real_signature.mean(0)
        self.count = 0
        self.dreal, self.dfake = [], []
        self.losses_D_all_real = []
        self.losses_D_all_fake = []
        self.losses_D_all = []
     
    def fit(self, device):
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        self.acts = self.acts.to(device)
        
        if self.acts_labelled != None:
            self.acts_labelled = self.acts_labelled.to(device)
        if self.acts_test != None:
            self.acts_test = self.acts_test.to(device)

        for netG in self.netGs:
            netG.to(device)
            self.gnoise_criterion.append(NoiseLoss(params=netG.parameters(), scale=math.sqrt(1.5*self.gnoise_alpha/self.lr_generator), observed=self.batch_size, device=device))
            
        for netD in self.netDs:
            netD.to(device)
            self.dnoise_criterion.append(NoiseLoss(params=netD.parameters(), scale=math.sqrt(1.5*self.dnoise_alpha/(10*self.lr_discriminator)), observed=len(self.netGs)*self.batch_size, device=device))

        self.gprior_criterion = PriorLoss(prior_std=1., observed=self.batch_size, device=device) #1000.
        self.dprior_criterion = PriorLoss(prior_std=1., observed=10*self.batch_size, device=device) #50000., device=device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_comp = ComplementCrossEntropyLoss(except_index=0, device=device)
        
        if self.x_labelled is not None and self.x_labelled_sig is not None:
            if type(self.x_labelled) == list:
                self.target_sup = self.labels
            else:
                self.target_sup = self.x_labelled[:,-1]
            class_counts = np.bincount(self.target_sup)  # [5, 4, 2] - counts of classes 0, 1, 2
            total_samples = self.target_sup.shape[0]
            class_weights = total_samples / (len(class_counts) * class_counts)
            class_weights_normalized = class_weights / class_weights.sum()
            self.class_weights = torch.tensor(class_weights_normalized, dtype=torch.float).to(device)
            if type(self.x_labelled) != list:
                self.input_sup_direct = Variable(self.x_labelled[:,0:-1].to(device))
                
            self.target_sup = Variable(self.target_sup.to(device))
            
            if type(self.x_labelled) == list:
                self.input_sup = Variable(self.x_labelled_sig.to(device))
            else:
                self.input_sup = Variable(self.x_labelled_sig[:,0:-1].to(device))
            
            self.input_sup_all = self.input_sup.clone()
            self.target_sup_all = self.target_sup.clone()

        if self.x_fake_test_labelled is not None and self.x_fake_test_labelled_sig is not None:
            if type(self.x_fake_test_labelled) == list:
                self.x_fake_test_target = self.labels_test
            else:
                self.x_fake_test_target = self.x_fake_test_labelled[:,-1]
            if type(self.x_fake_test_labelled) != list:
                self.x_fake_test_input_direct = Variable(self.x_fake_test_labelled[:,0:-1].to(device)) 
            self.x_fake_test_target = Variable(self.x_fake_test_target.to(device))        
            
            if type(self.x_fake_test_labelled) == list:
                self.x_fake_test_input = Variable(self.x_fake_test_labelled_sig.to(device))
                #self.x_fake_test_input = (self.x_fake_test_input - self.min_vals) / (self.max_vals - self.min_vals + 1e-16)
            else:
                self.x_fake_test_input = Variable(self.x_fake_test_labelled_sig[:,0:-1].to(device))
            
            self.x_fake_test_input_all = self.x_fake_test_input.clone()   
            self.x_fake_test_target_all = self.x_fake_test_target.clone() 

        for i in range(self.n_gradient_steps):
            break_loop = False
            self.step(device, g=i)
            if i > 300:
                if i%10 == 0:
                    self.stateG.append([copy.deepcopy(G.state_dict()) for G in self.netGs])
                    self.stateD.append([copy.deepcopy(D.state_dict()) for D in self.netDs])

            for j in range(len(self.netGs)): 
                for k in range(len(self.netDs)):
                    
                    if np.isnan(self.losses_history['D_loss_' + str(k) + "_for_G_" + str(j)][-1]):
                        break_loop = True
                        break
                if np.isnan(self.losses_history['G_loss'+str(j)][-1]) or break_loop:
                    break_loop = True
                    break
            
            if break_loop:
                print("epoch: ", i)
                print('D_loss_' + str(k) + "for_G_" + str(j)+": {:1.2e}".format(self.losses_history['D_loss_' + str(k) + "_for_G_" + str(j)][-1]))
                print('err_sup_' + str(k) + "for_G_" + str(j)+": {:1.2e}".format(self.losses_history['err_sup_' + str(k) + "_for_G_" + str(j)][-1]))
                print("G_loss_"+str(j)+": {:1.2e}".format(self.losses_history['G_loss'+str(j)][-1]))  
                break
            
    def step(self, device, g):
        self.g = g
        if len(self.x_real) < self.batch_size:
            self.batch_size = len(self.x_real)
        for i in range(self.D_steps_per_G_step):
            torch.manual_seed(i)
            random.seed(i)
            np.random.seed(i)
            indices = sample_indices(len(self.x_real), self.batch_size)
            
            x_real_batch = self.x_real_signature[indices,:]
            
            if self.acts!=None:
                acts_batch = self.acts[indices,:].to(device)
            
            self.input_sup = self.input_sup_all
            self.target_sup = self.target_sup_all
            fakes_orig = []
            fakes_gp = []
            fakes = []
            for _idxm in range(self.num_mcmc):
                
                fakes_numz = []
                
                
                if self.generator_type == 'ResFNN_gen':
                    noise_tmp = torch.randn(self.batch_size, self.len_noise, device=device)
                with torch.no_grad():
                    fake = self.netGs[_idxm](z=noise_tmp, embeddings=acts_batch, batch_size=self.batch_size, n_lags=self.n_lags, device=device)
                fakes_numz = fake
                if self.numz>1:
                    fakes.append(torch.cat(fakes_numz))
                else:
                    fakes.append(fakes_numz)
            
                fakes_orig.append(fake)
            del noise_tmp
            self.count = self.count + 1
            if self.netD_fullsup != None:
                if self.netD_fullsup_direct != None:
                    wgan_gp, D_losses, D_losses_real, D_losses_fake, err_fullsup, errD_noise, errD_prior, total_loss, err_fullsup_fake_test, err_fullsup_fake_test_misclass, err_sup, err_sup_fake_test, err_sup_fake_test_misclass, comp_gan_bceloss, comp_gan_misclass, comp_gan_misclass2, comp_fullsup_bceloss, comp_fullsup_misclass, fullsup_f1_fake_test, gan_f1_fake_test, fullsup_f1, gan_f1, err_fullsup_direct, err_fullsup_fake_test_direct, err_fullsup_fake_test_misclass_direct, fullsup_f1_fake_test_direct, fullsup_wauc_fake_test, fullsup_wauc, gan_wauc, gan_wauc_fake_test   = self.D_trainstep(fakes, x_real_batch, device, acts_batch=acts_batch, iteration=i) #in_sample_gan_f1, comp_gan_f1, comp_fullsup_f1, in_sample_fullsup_f1
                else:
                    wgan_gp, D_losses, D_losses_real, D_losses_fake, err_fullsup, errD_noise, errD_prior, total_loss, err_fullsup_fake_test, err_fullsup_fake_test_misclass, err_sup, err_sup_fake_test, err_sup_fake_test_misclass, comp_gan_bceloss, comp_gan_misclass, comp_gan_misclass2, comp_fullsup_bceloss, comp_fullsup_misclass, fullsup_f1_fake_test, gan_f1_fake_test, fullsup_f1, gan_f1, fullsup_wauc_fake_test, fullsup_wauc, gan_wauc, gan_wauc_fake_test = self.D_trainstep(fakes, x_real_batch, device, acts_batch=acts_batch, iteration=i) #in_sample_gan_f1, comp_gan_f1, comp_fullsup_f1, in_sample_fullsup_f1
            else:
                
                wgan_gp, D_losses, D_losses_real, D_losses_fake, errD_noise, errD_prior, total_loss, err_sup, err_sup_fake_test, err_sup_fake_test_misclass, comp_gan_bceloss, comp_gan_misclass, comp_gan_misclass2, gan_f1_fake_test, gan_f1, gan_wauc, gan_wauc_fake_test = self.D_trainstep(fakes, x_real_batch, device, acts_batch=acts_batch, iteration=i)
            self.losses_D_all_fake.append(D_losses_fake[0][0])
            self.losses_D_all_real.append(D_losses_real[0][0])
            self.losses_D_all.append(D_losses[0][0])
            
            if i == self.D_steps_per_G_step-1: 
                if self.netD_fullsup != None:
                    self.losses_history['Fullsup'].append(err_fullsup)
                    self.losses_history['err_fullsup_fake_test'].append(err_fullsup_fake_test)
                    self.losses_history['err_fullsup_fake_test_misclass'].append(err_fullsup_fake_test_misclass)
                    self.losses_history['Fullsup_BCELoss'].append(comp_fullsup_bceloss)
                    self.losses_history['Fullsup_Misclass'].append(comp_fullsup_misclass)
                    self.losses_history['Fullsup_F1_fake_test'].append(fullsup_f1_fake_test)
                    self.losses_history['Fullsup_WAUC_fake_test'].append(fullsup_wauc_fake_test)
                    self.losses_history['Fullsup_F1'].append(fullsup_f1)
                    self.losses_history['Fullsup_WAUC'].append(fullsup_wauc)
                    
                    if self.netD_fullsup_direct != None:
                        self.losses_history['Fullsup_direct'].append(err_fullsup_direct)
                        self.losses_history['err_fullsup_direct_fake_test'].append(err_fullsup_fake_test_direct)
                        self.losses_history['err_fullsup_direct_fake_test_misclass'].append(err_fullsup_fake_test_misclass_direct)
                        self.losses_history['Fullsup_direct_F1_fake_test'].append(fullsup_f1_fake_test_direct)
            
                   
                for j in range(len(self.netGs)):
                    for k in range(len(self.netDs)):
                        self.losses_history['D_loss_fake_' + str(k) + "_for_G_" + str(j)].append(D_losses_fake[j][k])
                        self.losses_history['D_loss_real_' + str(k) + "_for_G_" + str(j)].append(D_losses_real[j][k])
                        self.losses_history['D_loss_' + str(k) + "_for_G_" + str(j)].append(D_losses[j][k])
                        self.losses_history['WGAN_GP_' + str(k) + "_for_G_" + str(j)].append(wgan_gp[j][k])
                        self.losses_history['err_sup_' + str(k) + "_for_G_" + str(j)].append(err_sup[j][k])
                        
                        if j == 0:
                            self.losses_history['err_sup_fake_test' + str(k)].append(err_sup_fake_test[k])
                            self.losses_history['gan_f1_fake_test' + str(k)].append(gan_f1_fake_test[k])
                            self.losses_history['gan_wauc_fake_test' + str(k)].append(gan_wauc_fake_test[k])
                            self.losses_history['gan_f1' + str(k)].append(gan_f1[k])
                            self.losses_history['gan_wauc' + str(k)].append(gan_wauc[k])
                            self.losses_history['err_sup_fake_test_misclass' + str(k)].append(err_sup_fake_test_misclass[k])
                            self.losses_history['GAN_BCELoss' + str(k)].append(comp_gan_bceloss[k])
                            self.losses_history['GAN_Misclass' + str(k)].append(comp_gan_misclass[k])
                            
                        if g>2 and k==0 and self.losses_history['gan_wauc' + str(k)][-1] > self.losses_history['gan_wauc' + str(k)][-2]:
                            self.stateD_max_wauc = [copy.deepcopy(D.state_dict()) for D in self.netDs]  
                        
                        if self.bayes:
                            self.losses_history['errD_noise_' + str(k) + "_for_G_" + str(j)].append(errD_noise[j][k])
                            self.losses_history['errD_prior_' + str(k) + "_for_G_" + str(j)].append(errD_prior[j][k])
                            
                        self.losses_history['total_loss_' + str(k) + "_for_G_" + str(j)].append(total_loss[j][k])
            
            for j in range(len(self.netGs)):
                for k in range(len(self.netDs)):
                    self.losses_history['D_loss_fake_' + str(i) + "_for_G_0_full"].append(D_losses_fake[j][k])
                    self.losses_history['D_loss_real_' + str(i) + "_for_G_0_full"].append(D_losses_real[j][k])
                    self.losses_history['D_loss_' + str(i) + "_for_G_0_full"].append(D_losses[j][k])            
        G_losses, sigw1_loss_list, errG_noise, errG_prior = self.G_trainstep(device, g, acts_batch)
        for m in range(len(G_losses)):
            self.losses_history['G_loss' + str(m)].append(G_losses[m])
            
        for t in range(self.num_mcmc):
            if self.generator_type != 'ResFNN_gen':
                self.losses_history['sigw1_loss_G'+ str(t)].append(sigw1_loss_list[t])
            if self.bayes:
                self.losses_history['errG_noise_for_G_' + str(t)].append(errG_noise[t])
                self.losses_history['errG_prior_for_G_' + str(t)].append(errG_prior[t])
            
    def G_trainstep(self, device, g, acts_batch=None):
        torch.manual_seed(g)
        random.seed(g)
        np.random.seed(g)
                
        self.fakes = []
        self.fakes_eval = []
        sigw1_loss_list = []
        
        fakes_sig_mcmc = []
        errG = []
        errG_noise = []
        errG_prior = []
        for _idxm in range(self.num_mcmc):
            toggle_grad(self.netGs[_idxm], True)
            self.netGs[_idxm].train()
            self.optimizerGs[_idxm].zero_grad()
            
            noise_tmp = torch.randn(self.batch_size, self.len_noise, device=device)
            fakes_sig_numz = []
            
            fake = self.netGs[_idxm](z=noise_tmp, embeddings=acts_batch, batch_size=self.batch_size, n_lags=self.n_lags, device=device) 
            self.fakes_eval.append(fake)  

            x_fake_signature = fake
            
            x_fake_signature.requires_grad_()
            fakes_sig_numz.append(x_fake_signature)
            
            fakes_sig_numz_tensor = x_fake_signature
            fakes_sig_mcmc.append(fakes_sig_numz_tensor)
            
            d_fakes = []
            for netD in self.netDs:
                
                if self.generator_type == 'ResFNN_gen':
                    df = netD(fakes_sig_mcmc[-1].to(device), acts_batch)
                else:
                    df = netD(fakes_sig_mcmc[-1], acts_batch)
                    
                d_fakes.append(df)
                netD.train()
            d_fake = torch.stack(d_fakes, dim=2)
            d_fake = d_fake.mean(2)
            
            with torch.autograd.set_detect_anomaly(True):
                
                errG.append(g_loss(d_fake))
                    
            errG[_idxm].backward(retain_graph=True)
            
            if self.bayes:
                errgnoise = self.gnoise_criterion[_idxm](self.netGs[_idxm].parameters(), seed=g)
                if torch.isnan(errgnoise)!=True:
                    errG_noise.append(errgnoise)
                    errG_noise[_idxm].backward()
                else:
                    errG_noise.append(0)
                    
                errgprior =  self.gprior_criterion(self.netGs[_idxm].parameters(), "recursive")
                if torch.isnan(errgprior)!=True:
                    errG_prior.append(errgprior)
                    errG_prior[_idxm].backward()
                else:
                    errG_prior.append(0)
                errG[_idxm] = errG[_idxm] + errG_prior[_idxm] + errG_noise[_idxm] 
                
            # Clip the norm of the gradients
            ut.clip_grad_norm_(self.netGs[_idxm].parameters(), 3)
            self.optimizerGs[_idxm].step()
            if self.bayes:
                errG[_idxm] = errG[_idxm] - errG_prior[_idxm]
        del noise_tmp
        
        return [l.item() if l!=0 else l for l in errG], sigw1_loss_list, [l.item() if l!=0 else l for l in errG_noise], [l.item() if l!=0 else l for l in errG_prior] #errG.detach()

    def D_trainstep(self, fakes, x_real_signature, device="cpu", acts_batch=None, iteration=None): #, g

        x_real_signature.requires_grad_()
        list_wgan_gp = []
        list_dlosses = []
        list_dlosses_real = []
        list_dlosses_fake = []
        list_err_sup = []
        list_errD_noise = []
        list_errD_prior = []
        list_l = []
        
        err_sup_fake_test = []
        gan_f1_fake_test = []
        gan_wauc_fake_test = []
        err_sup_fake_test_misclass = []
        comp_gan_bceloss = []
        comp_gan_misclass = []
        comp_gan_misclass2 = []
        gan_f1 = []
        gan_wauc = []
        
        for j in range(len(fakes)):
            x_fake_orig = fakes[j]
            
            x_fake_signature = x_fake_orig.to(device)
            x_real_signature = x_real_signature.to(device)
            x_fake_signature.requires_grad_()
            
            # For Labelled data (for semi-supervised learning)
            err_sup = []
                
            if self.bayes:
                errD_prior, errD_noise = [], []
                
            dlosses = []
            dlosses_real = []
            dlosses_fake = []
            errD, wgan_gp = [], []
            total_losses = []
            
                            
            for i in range(len(self.netDs)):
                toggle_grad(self.netDs[i], True)
                self.netDs[i].train()
                self.optimizerDs[i].zero_grad()
                self.netDs[i].zero_grad()
                
                d_real = self.netDs[i](x_real_signature, acts_batch)
                if self.generator_type == 'ResFNN_gen':
                    d_fake = self.netDs[i](x_fake_signature, acts_batch)
                else:
                    d_fake = self.netDs[i](x_fake_signature, acts_batch)
                    
                dloss, dloss_real, dloss_fake = d_loss(d_fake, d_real)      
                
                dlosses.append(dloss)
                dlosses_real.append(dloss_real)
                dlosses_fake.append(dloss_fake)
                
                with torch.backends.cudnn.flags(enabled=False):
                    wgan_gp.append(self.reg_param * self.wgan_gp_reg(x_real_signature, x_fake_signature, i=i, acts_batch=acts_batch)) 
                    
                dloss.backward(retain_graph=True)
                wgan_gp[-1].backward(retain_graph=True)
                
                if self.bayes:
                    errdnoise = 0.5 * self.dnoise_criterion[i](self.netDs[i].parameters(), seed=self.g, pr=False)
                    if torch.isnan(errdnoise)!=True:
                        errD_noise.append(errdnoise)
                        errD_noise[i].backward(retain_graph=True)
                    else:
                        errD_noise.append(errdnoise)
                        
                    errdprior = 0.1 * self.dprior_criterion(self.netDs[i].parameters(), "recursive", pr=False, g=1)
                    if torch.isnan(errdprior)!=True:
                        errD_prior.append(errdprior)
                        errD_prior[i].backward(retain_graph=True)
                    else:
                        errD_prior.append(errdprior)
                num_ii = 1
                for ii in range(num_ii):
                    
                    if self.sup:
                        output_sup = self.netDs[i](self.input_sup, self.acts_labelled) 
                        pred = torch.softmax(output_sup[:,1:], axis=1)
                        errsup = nn.BCELoss()(pred[:,1].float(), self.target_sup.float().to(device)) * self.l_errsup  
                            
                    else:
                        errsup=0
                
                    if self.sup and math.isnan(errsup)!=True and errsup > 0:
                        errsup.backward()
                        if ii == (num_ii-1):
                            if errsup > 5:
                                errsup = 5
                            err_sup.append(errsup)
                    else:
                        err_sup.append(0)
                
                ut.clip_grad_norm_(self.netDs[i].parameters(), 3)
                self.optimizerDs[i].step()  
                if j == 0 and iteration == (self.D_steps_per_G_step-1): 
                    if self.sup:
                        self.netDs[i].eval()
                        out = self.netDs[i](self.x_fake_test_input, self.acts_test)
                        pred = torch.softmax(out[:,1:], axis=1)
                        err_sup_fake_test.append(nn.BCELoss()(pred[:,1].float(), self.x_fake_test_target.float().to(device)).cpu().detach().numpy())
                        err_sup_fake_test_misclass.append(misclass(out[:,1:]-0, self.x_fake_test_target.to(device)).cpu())
                        gan_f1_fake_test.append(0) #F1_score(out-0, self.x_fake_test_target, device).cpu().detach().numpy())
                        prob = torch.softmax(out[:,1:], dim=1).cpu().detach().numpy()
                        gan_wauc_fake_test.append(wauc(self.x_fake_test_target.cpu().flatten(), prob, classes=[0,1]))

                        output_comp_gan = self.netDs[i](self.input_sup, self.acts_labelled) 
                        pred = torch.softmax(output_comp_gan[:,1:], axis=1)
                        comp_gan_bceloss.append(nn.BCELoss()(pred[:,1].float(), self.target_sup.float().to(device)).cpu().detach().numpy())
                        comp_gan_misclass.append(misclass(output_comp_gan[:,1:], self.target_sup).cpu())
                        gan_f1.append(0) #F1_score(output_comp_gan-0, self.target_sup, device).cpu().detach().numpy())
                        prob = torch.softmax(output_comp_gan[:,1:], dim=1).cpu().detach().numpy()
                        gan_wauc.append(wauc(self.target_sup.cpu().flatten(), prob, classes=[0,1]))

                    else:
                        err_sup_fake_test.append(0)
                        err_sup_fake_test_misclass.append(0)
                        gan_f1_fake_test.append(0)
                        gan_wauc_fake_test.append(0)
                        comp_gan_bceloss.append(0)
                        comp_gan_misclass.append(0)
                        gan_f1.append(0)  
                        gan_wauc.append(0)
                
                out = self.netDs[i](self.x_fake_test_input, self.acts_test)
                prob = torch.softmax(out[:,1:], dim=1).cpu().detach().numpy()
                val_wac = wauc(self.x_fake_test_target.cpu().flatten(), prob, classes=[0,1])

                self.scheduler[i].step(val_wac)
                
                loss = dlosses_real[i] + dlosses_fake[i] + wgan_gp[i]
                if self.sup and math.isnan(err_sup[i])!=True:
                    loss = loss + err_sup[i]
                if self.bayes:
                    loss = loss + errD_noise[i] + errD_prior[i]
                total_losses.append(loss) 
                toggle_grad(self.netDs[i], False)
                
            list_wgan_gp.append([gp.cpu().detach().numpy() if gp !=0 else gp for gp in wgan_gp])
            list_dlosses.append([l.item() for l in dlosses])
            list_dlosses_real.append([l.item() for l in dlosses_real])
            list_dlosses_fake.append([l.item() for l in dlosses_fake])
            
            list_err_sup.append([l.cpu().detach().numpy() if (l<5 and l!=0) else l for l in err_sup])
            
            if self.bayes:
                list_errD_noise.append([l.cpu().detach().numpy() if l!=0 else l for l in errD_noise])
                list_errD_prior.append([l.cpu().detach().numpy() if l!=0 else l for l in errD_prior])
                
            list_l.append([loss.item() for loss in total_losses])
            
            if self.netD_fullsup != None and self.input_sup != None and self.target_sup != None:
            
                toggle_grad(self.netD_fullsup, True)
                self.netD_fullsup.train()
                self.netD_fullsup.zero_grad()
                self.optimizerD_fullsup.zero_grad()
                
                output_fullsup = self.netD_fullsup(self.input_sup, self.acts_labelled) 
                pred = torch.softmax(output_fullsup, axis=1)
                err_fullsup = nn.BCELoss()(pred[:,1].float(), self.target_sup.float().to(device))
                            
                err_fullsup.backward()
                self.optimizerD_fullsup.step()
                self.scheduler_fullsup.step(err_fullsup)
                
                if j == 0 and iteration == (self.D_steps_per_G_step-1):
                    self.netD_fullsup.eval()
                    
                    output_fake_fullsup_test = self.netD_fullsup(self.x_fake_test_input, self.acts_test)
                    pred = torch.softmax(output_fake_fullsup_test, axis=1)
                    err_fullsup_fake_test = nn.BCELoss()(pred[:,1].float(), self.x_fake_test_target.float().to(device)).cpu().detach().numpy() 
                
                    err_fullsup_fake_test_misclass = misclass(output_fake_fullsup_test, self.x_fake_test_target).cpu()
                    fullsup_f1_fake_test = 0 #F1_score(output_fake_fullsup_test-0, self.x_fake_test_target, device).cpu().detach().numpy()
                    prob = torch.softmax(output_fake_fullsup_test, dim=1).cpu().detach().numpy()
                    fullsup_wauc_fake_test = wauc(self.x_fake_test_target.cpu().flatten(), prob, classes=[0,1])
                    
                    output_comp_fullsup = self.netD_fullsup(self.input_sup, self.acts_labelled) #[self.ind,:]) 
                    pred = torch.softmax(output_comp_fullsup, axis=1)
                    comp_fullsup_bceloss = nn.BCELoss()(pred[:,1].float(), self.target_sup.float().to(device)).cpu().detach().numpy()
                
                    comp_fullsup_misclass = misclass(output_comp_fullsup-0, self.target_sup).cpu()
                    fullsup_f1 = 0 #F1_score(output_comp_fullsup-0, self.target_sup, device).cpu().detach().numpy()
                    prob = torch.softmax(output_comp_fullsup, dim=1).cpu().detach().numpy()
                    fullsup_wauc = wauc(self.target_sup.cpu().flatten(), prob, classes=[0,1])
                else:
                    err_fullsup_fake_test, err_fullsup_fake_test_misclass, fullsup_f1_fake_test, fullsup_wauc_fake_test = 0,0,0,0
                    comp_fullsup_bceloss, comp_fullsup_misclass, fullsup_f1, fullsup_wauc = 0,0,0,0
                    
                err_fullsup_direct, err_fullsup_fake_test_direct, err_fullsup_fake_test_misclass_direct, fullsup_f1_fake_test_direct = 0,0,0,0
            else:
                err_fullsup = 0
                err_fullsup_fake_test, err_fullsup_fake_test_misclass, fullsup_f1_fake_test, fullsup_wauc_fake_test = 0,0,0,0
                comp_fullsup_bceloss, comp_fullsup_misclass, fullsup_f1, fullsup_wauc = 0,0,0,0
          
        
        if self.netD_fullsup != None:
            
            err_fullsup = err_fullsup.item() if err_fullsup!=0 else 0
            err_fullsup_fake_test = err_fullsup_fake_test.item() if err_fullsup_fake_test!=0 else 0
            
            if self.netD_fullsup_direct != None:
                err_fullsup_direct = err_fullsup_direct.item() if err_fullsup_direct!=0 else 0
                err_fullsup_fake_test_direct = err_fullsup_fake_test_direct.item() if err_fullsup_fake_test_direct!=0 else 0
                return list_wgan_gp, list_dlosses, list_dlosses_real, list_dlosses_fake, err_fullsup, list_errD_noise, list_errD_prior, list_l, err_fullsup_fake_test, err_fullsup_fake_test_misclass, list_err_sup, err_sup_fake_test, err_sup_fake_test_misclass, comp_gan_bceloss, comp_gan_misclass, comp_gan_misclass2, comp_fullsup_bceloss, comp_fullsup_misclass, fullsup_f1_fake_test, gan_f1_fake_test, fullsup_f1, gan_f1, err_fullsup_direct, err_fullsup_fake_test_direct, err_fullsup_fake_test_misclass_direct, fullsup_f1_fake_test_direct, fullsup_wauc_fake_test, fullsup_wauc, gan_wauc, gan_wauc_fake_test  #, in_sample_gan_f1, comp_gan_f1, comp_fullsup_f1, in_sample_fullsup_f1 #err_fullsup_fake_test.item()
            return list_wgan_gp, list_dlosses, list_dlosses_real, list_dlosses_fake, err_fullsup, list_errD_noise, list_errD_prior, list_l, err_fullsup_fake_test, err_fullsup_fake_test_misclass, list_err_sup, err_sup_fake_test, err_sup_fake_test_misclass, comp_gan_bceloss, comp_gan_misclass, comp_gan_misclass2, comp_fullsup_bceloss, comp_fullsup_misclass, fullsup_f1_fake_test, gan_f1_fake_test, fullsup_f1, gan_f1, fullsup_wauc_fake_test, fullsup_wauc, gan_wauc, gan_wauc_fake_test  #, in_sample_gan_f1, comp_gan_f1, comp_fullsup_f1, in_sample_fullsup_f1 #err_fullsup_fake_test.item()
      
        else:
            return list_wgan_gp, list_dlosses, list_dlosses_real, list_dlosses_fake, list_errD_noise, list_errD_prior, list_l, list_err_sup, err_sup_fake_test, err_sup_fake_test_misclass, comp_gan_bceloss, comp_gan_misclass, comp_gan_misclass2, gan_f1_fake_test, gan_f1, gan_wauc, gan_wauc_fake_test
            
    def wgan_gp_reg(self, x_real, x_fake, i, acts_batch=None, center=1.):
        batch_size = x_real.size(0)
        if self.sig:
            eps = torch.rand(batch_size, device=x_real.device).view(batch_size, 1)
        else:
            eps = torch.rand(batch_size, device=x_real.device).view(batch_size, 1, 1) 
        if self.generator_type == 'ResFNN_gen':
            x_fake = x_fake
        x_interp = (1 - eps) * x_real + eps * x_fake
        x_interp.requires_grad_()
        if self.generator_type == 'ResFNN_gen':
            d_out = self.netDs[i](x_interp, acts_batch)
        else:
            d_out = self.netDs[i](x_interp, acts_batch)
        d_out = 1/math.sqrt(d_out.shape[1]) *(d_out[:,0]-torch.sum(d_out[:,1:], dim=1))
        grad = compute_grad2(d_out, x_interp)
        reg = (grad.sqrt() - center).pow(2).mean()
        return reg



def train_gru(model, X_train, y_train, epochs=10, batch_size=64, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n = X_train.shape[0]
    for ep in range(1, epochs+1):
        # shuffle
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start+batch_size]
            xb, yb = X_train[idx].to(device), y_train[idx].to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
def update_teacher(student_net, teacher_net, alpha):
    """ teacher = alpha*teacher + (1-alpha)*student """
    for teacher_param, student_param in zip(teacher_net.parameters(), student_net.parameters()):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=(1 - alpha))

def train_mean_teacher(student, teacher, X_lab, emb_lab, y_lab, X_unlab, emb_unlab, 
                       optimizer, batch_size=32, steps_per_epoch=100):
    student.train()
    teacher.eval()  # Teacher often kept in eval mode (no dropout, BN updates, etc.)

    n_lab   = len(X_lab)
    n_unlab = len(X_unlab)
    
    noise_std=0.05  
    consistency_weight=0.5  
    for step in range(steps_per_epoch):
        # 1) Sample a random batch from labeled data
        idx_lab = np.random.choice(n_lab, batch_size, replace=True)
        data_l, data_l_emb  = X_lab[idx_lab], emb_lab[idx_lab]
        labels_l = y_lab[idx_lab]

        # 2) Sample a random batch from unlabeled data
        idx_unlab = np.random.choice(n_unlab, batch_size, replace=True)
        data_u, data_u_emb = X_unlab[idx_unlab], emb_unlab[idx_unlab]

        # 3) Add Gaussian noise for perturbations
        data_l_noisy = data_l + noise_std * torch.randn_like(data_l)
        data_u_student = data_u + noise_std * torch.randn_like(data_u)
        data_u_teacher = data_u  # teacher sees clean input

        # 4) Compute student predictions
        logits_l = student(data_l_noisy, data_l_emb)
        logits_u_student = student(data_u_student, data_u_emb)

        # 5) Compute teacher predictions (fixed at this step)
        with torch.no_grad():
            logits_u_teacher = teacher(data_u_teacher, data_u_emb)

        # 6) Convert to probabilities
        probs_u_student = F.log_softmax(logits_u_student, dim=1)
        probs_u_teacher = F.softmax(logits_u_teacher, dim=1)

        # 7) Compute losses
        loss_sup = criterion_sup(logits_l, labels_l)
        loss_cons = F.kl_div(probs_u_student, probs_u_teacher, reduction="batchmean")

        loss = loss_sup + consistency_weight * loss_cons

        # 8) Update student
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 9) Update teacher using EMA
        update_teacher(student, teacher, alpha=ema_decay)