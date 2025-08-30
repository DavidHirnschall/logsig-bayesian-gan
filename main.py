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
from trainer import *

def run_sbgan(
    datasets = ('fraud',),
    discriminators=('ResFNN',),
    generators=('ResFNN_gen',),
    n_seeds=10,
    device='cuda:0',
    num = 0,
    logret = False, 
    x_real = None,
    y_real = None,
    x_labelled = None,
    fullsup = False, 
    x_fake_test_labelled = None,
    numG = None,
    numD = None,
    sup = False,
    rate=None,
    lr_g=None,
    lr_d=None,
    l_errsup=1,
    random_state=None,
    labels=None,
    labels_test=None,
    acts=None,
    acts_labelled=None, 
    acts_test=None,
    bs=None,
    x_sig=None,
    x_logsig=None,
    x_labelled_logsig=None, 
    x_fake_test_labelled_logsig=None
):
    
    gan_algo='SBGAN'
    seeds = list(range(n_seeds))
    
    grid = itertools.product(datasets, discriminators, generators, seeds)
    
    for dataset, discriminator, generator, seed in grid:
        data_config = load_obj(get_config_path(dataset, dataset))
        discriminator_config = load_obj(get_config_path_discriminator(discriminator, dataset))
        gan_config = load_obj(get_config_path('SBGAN', dataset))
        generator_config = load_obj(get_config_path_generator(generator, dataset))

        if gan_config.get('augmentations') is not None:
            gan_config['augmentations'] = parse_augmentations(gan_config.get('augmentations'))

        if generator_config.get('augmentations') is not None:
            generator_config['augmentations'] = parse_augmentations(generator_config.get('augmentations'))

        if generator_config['generator_type'] == 'LogSigRNN':
            generator_config['n_lags'] = data_config['n_lags']

        experiment_dir = get_wgan_experiment_dir(dataset, discriminator, generator, 'SBGAN', seed)

        if not pt.exists(experiment_dir):
            os.makedirs(experiment_dir)

        save_obj(data_config, pt.join(experiment_dir, 'data_config.pkl'))
        save_obj(discriminator_config, pt.join(experiment_dir, 'discriminator_config.pkl'))
        save_obj(gan_config, pt.join(experiment_dir, 'gan_config.pkl'))
        save_obj(generator_config, pt.join(experiment_dir, 'generator_config.pkl'))

        print('Training: %s' % experiment_dir.split('/')[-2:])
        
    set_seed(seed)
    if numG != None and numD == None:
        experiment_dir = experiment_dir +"_"+ str(numG) + "G1D" + "_" + str(num) + "_" + str(logret) + "_" + str(sup)
    if numD != None and numG == None:
        experiment_dir = experiment_dir +"_"+ "1G" + str(numD) + "D" + "_" + str(num) + "_" + str(logret) + "_" + str(sup)
    if numD != None and numG != None:
        experiment_dir = experiment_dir +"_"+ str(numG) + "G" + str(numD) + "D" + "_" + str(num) + "_" + str(logret) + "_" + str(sup)
    else:
        experiment_dir = experiment_dir + "_" + str(num) + "_" + str(logret) + "_" + str(sup)
        
    if rate!=None:
        experiment_dir = experiment_dir + "_" + str(rate)
    if lr_g!=None:
        experiment_dir = experiment_dir + "_G" + str(lr_g)
    if lr_d!=None:
        experiment_dir = experiment_dir + "_D" + str(lr_d)
    if random_state!=None:
        experiment_dir = experiment_dir + "_" + str(random_state)
    if bs != None:
        experiment_dir = experiment_dir + "_" + str(bs)
    if l_errsup != None:
        experiment_dir = experiment_dir + "_" + str(l_errsup)
        
    print(experiment_dir)
    
    if gan_algo == 'SBGAN':
        gan_config["numD"] = numD
        gan_config["num_mcmc"] = numG
        if gan_config["sup"] != 0 and gan_config["sup"] != 1:
            gan_config["sup"] = sup
    
    if not pt.exists(experiment_dir):
        os.makedirs(experiment_dir)
        
    save_obj(data_config, pt.join(experiment_dir, 'data_config.pkl'))
    save_obj(discriminator_config, pt.join(experiment_dir, 'discriminator_config.pkl'))
    save_obj(gan_config, pt.join(experiment_dir, 'gan_config.pkl'))
    save_obj(generator_config, pt.join(experiment_dir, 'generator_config.pkl'))
    
    ##############
    gan_config['augmentations'] = [Scale(scale=2, dim=0),
     AddTime(),
     LeadLag(),
     VisiTrans()]

    ##############
    n_lags = data_config.pop("n_lags")
        
    n_lags=24   
    if x_real == None:
        
        data_config["n_lags"]=20
    else:
        x_real = x_real

    x_real = [x.to(device) for x in x_real]
    n_lags = int( sum([x.shape[0] for x in x_real]) / len(x_real) )
    x_real_rolled = x_real
    x_real_train, x_real_test = x_real[0].unsqueeze(0), x_real[0].unsqueeze(0)
    x_real_dim: int = x_real[0].shape[1]
        
    # Get generator
    #set_seed(seed)
    generator_config.update(output_dim=x_real_dim)
    if generator_config['generator_type'] == 'LogSigRNN':
        generator_config.update(n_lags=n_lags)
    if generator_config['generator_type'] == 'LSTM':
        generator_config.update(len_noise=n_lags)
    generator_config.update(init_fixed=False)
    if generator_config['generator_type'] == 'ResFNN_gen':
        generator_config.update(len_noise=256)
        generator_config.update(output_dim=[140,3])
        generator_config.update(hidden_dims=[256,256])
        generator_config.update(input_dim=271) #256+15 
    G = get_generator(**generator_config).to(device)

    if gan_algo == 'SBGAN':
        set_seed(seed)
        aug = gan_config['augmentations']
        depth = gan_config['depth']
        
        if sup == True:
            gan_config['sup'] = True
        if type(x_real) == list:
            
            if x_sig != None:
                x_real_signature = x_sig
                x_real_logsig = x_logsig
            else:
                x_real_logsig = sig_list(x_real_rolled, aug)
                x_real_signature = sig_list(x_real_rolled, aug, log=False)
        else:
            x_real_augmented = apply_augmentations(x_real_rolled, aug)
            x_real_logsig = signatory.logsignature(x_real_augmented, depth=depth)
            x_real_signature = signatory.signature(x_real_augmented, depth=depth)
        discriminator_config.update(input_dim=140)
        if acts_labelled != None:
            discriminator_config.update(emb=True)
            input_d = 155
            discriminator_config.update(input_dim=input_d)
        discriminator_config.update(hidden_dims=[32,32])
        netDs = []
        D = get_discriminator(**discriminator_config).to(device)
        for _id in range(gan_config["numD"]):
            netDs.append(D)
                        
        if fullsup == True:
            discriminator_config.update(output_dim=2)
            netD_fullsup = get_discriminator(**discriminator_config).to(device)
            discriminator_config.update(input_dim=int(n_lags))
            discriminator_config.update(output_dim=3)
            netD_fullsup_direct = get_discriminator(**discriminator_config).to(device)
            if acts_labelled!=None:
                discriminator_config.update(input_dim=input_d)
        else: 
            netD_fullsup = None
            netD_fullsup_direct = None
        
        netGs = []
        for _idxm in range(gan_config["num_mcmc"]): 
            set_seed(_idxm)
            generator_config.update(output_dim=x_real_dim)
            if generator_config['generator_type'] == 'LogSigRNN':
                generator_config.update(n_lags=n_lags)
            if generator_config['generator_type'] == 'ResFNN_gen':
                generator_config.update(output_dim=[140,3])
                generator_config.update(hidden_dims=[256,256])
            generator_config.update(init_fixed=False)
            print("dim: ", generator_config['output_dim'])
            G = get_generator(**generator_config).to(device)
            netGs.append(G)
            
        set_seed(seed)
        if x_labelled != None:
            if type(x_labelled) == list:
                input_sup_direct, target_sup = [x.to(device) for x in x_labelled], labels.to(device)
                if x_labelled_logsig != None:
                    input_sup = x_labelled_logsig
                else:    
                    input_sup = sig_list(input_sup_direct, gan_config['augmentations'])
                x_labelled_sig = input_sup
            else:
                input_sup_direct, target_sup = x_labelled[:,0:-1].to(device), x_labelled[:,-1:].to(device)
                input_sup_a = apply_augmentations(input_sup_direct.unsqueeze(-1), gan_config['augmentations'])
                input_sup = signatory.logsignature(input_sup_a, depth=4)
                x_labelled_sig = torch.cat((input_sup, target_sup), dim=1)
        else:
            x_labelled_sig = None
            
        if x_fake_test_labelled != None:
            if type(x_fake_test_labelled) == list:
                x_fake_test_input_direct, x_fake_test_target = [x.to(device) for x in x_fake_test_labelled], labels_test.to(device)
                if x_fake_test_labelled_logsig != None:
                    x_fake_test_input = x_fake_test_labelled_logsig
                else:    
                    x_fake_test_input = sig_list(x_fake_test_input_direct, gan_config['augmentations'])
                x_fake_test_labelled_sig = x_fake_test_input
            else:
                x_fake_test_input_direct, x_fake_test_target = x_fake_test_labelled[:,0:-1].to(device), x_fake_test_labelled[:,-1:].to(device)
                x_fake_test_input_a = apply_augmentations(x_fake_test_input_direct.unsqueeze(-1), gan_config['augmentations'])
                x_fake_test_input = signatory.logsignature(x_fake_test_input_a, depth=4)
                x_fake_test_labelled_sig = torch.cat((x_fake_test_input, x_fake_test_target), dim=1)
                
        else:
            x_fake_test_labelled_sig=None
        if type(x_real) == list:
            x_real_sig = x_real_logsig #sig_list(x_real, gan_config['augmentations'])
        else:
            x_real_a = apply_augmentations(x_real, gan_config['augmentations'])
            x_real_sig = signatory.logsignature(x_real_a, depth=4)
        
        print("setup trainer")
        trainer = SBGANTrainer(netDs, netGs,
                              x_real=x_real_rolled,
                              len_noise = generator_config["len_noise"],  
                              input_dim = generator_config["input_dim"],  
                              x_labelled = x_labelled,
                              x_labelled_sig = x_labelled_sig,
                              netD_fullsup = netD_fullsup,
                              netD_fullsup_direct = netD_fullsup_direct,
                              x_fake_test_labelled = x_fake_test_labelled,
                              x_fake_test_labelled_sig = x_fake_test_labelled_sig,
                              lr_g=lr_g, 
                              lr_d=lr_d,  
                              l_errsup=l_errsup,
                              foo = lambda x: x.exp(),
                              labels=labels,
                 			  labels_test=labels_test,
                              acts=acts,
                              acts_labelled=acts_labelled, 
                              acts_test=acts_test,
                              generator_type = generator_config['generator_type'],
                              bs=bs,
                              x_sig=x_real_signature, x_logsig=x_real_logsig,
                              **gan_config
      )
    else:
        raise NotImplementedError()

    # Start training
    set_seed(seed)
    print("train")
    trainer.fit(device=device)
    for i in range(len(trainer.netGs)):
        trainer.netGs[i].eval()    
    for i in range(len(trainer.netDs)):
        trainer.netDs[i].eval()   

    # Store relevant training results
    save_obj([to_numpy(x) for x in x_real], pt.join(experiment_dir, 'x_real.pkl'))
    save_obj(to_numpy(labels), pt.join(experiment_dir, 'labels.pkl'))
    save_obj(to_numpy(labels_test), pt.join(experiment_dir, 'labels_test.pkl'))
    if acts_labelled != None and acts_test != None:
        save_obj(to_numpy(acts_labelled), pt.join(experiment_dir, 'acts_labelled.pkl'))
        save_obj(to_numpy(acts_test), pt.join(experiment_dir, 'acts_test.pkl'))
    save_obj(to_numpy(x_real_test), pt.join(experiment_dir, 'x_real_test.pkl'))
    save_obj(to_numpy(x_real_train), pt.join(experiment_dir, 'x_real_train.pkl'))
    save_obj(trainer.losses_history, pt.join(experiment_dir, 'losses_history.pkl'))  # dev of losses / metrics

    if gan_algo == "SBGAN":
        torch.save(trainer.stateG, pt.join(experiment_dir, 'generator_posterior_state_dict.pth'))
        torch.save(trainer.stateD, pt.join(experiment_dir, 'discriminator_posterior_state_dict.pth'))
        torch.save(trainer.stateD_max_wauc, pt.join(experiment_dir, 'discriminator_state_max_wauc.pth'))
        if trainer.min_vals != None:
            torch.save(trainer.min_vals, pt.join(experiment_dir, 'min_vals.pth'))
            torch.save(trainer.max_vals, pt.join(experiment_dir, 'max_vals.pth'))
            
        save_obj(to_numpy(x_real_sig), pt.join(experiment_dir, 'x_real_sig.pkl'))
        if x_labelled != None:
            if type(x_real)!=list:
                save_obj(to_numpy(x_labelled), pt.join(experiment_dir, 'x_labelled.pkl'))
            else:
                save_obj([to_numpy(x) for x in x_labelled], pt.join(experiment_dir, 'x_labelled.pkl'))
            save_obj(to_numpy(x_labelled_sig), pt.join(experiment_dir, 'x_labelled_sig.pkl'))
        if y_real != None:
            save_obj(y_real, pt.join(experiment_dir, 'y_real.pkl'))
        if x_fake_test_labelled != None:
            if type(x_real)!=list:
                save_obj(to_numpy(x_fake_test_labelled), pt.join(experiment_dir, 'x_fake_test_labelled.pkl'))
            else:
                save_obj([to_numpy(x) for x in x_fake_test_labelled], pt.join(experiment_dir, 'x_fake_test_labelled.pkl'))
            save_obj(to_numpy(x_fake_test_labelled_sig), pt.join(experiment_dir, 'x_fake_test_labelled_sig.pkl'))
            
        for i in range(len(trainer.netGs)):
            save_obj(trainer.netGs[i].state_dict(), pt.join(experiment_dir, 'generator_' + str(i) + '_state_dict.pt'))
        for i in range(len(trainer.netDs)):
            save_obj(trainer.netDs[i].state_dict(), pt.join(experiment_dir, 'discriminator_' + str(i) + '_state_dict.pt'))
        if fullsup:
            save_obj(trainer.netD_fullsup.state_dict(), pt.join(experiment_dir, 'fullsup_discriminator_' + '_state_dict.pt'))
            if netD_fullsup_direct != None:
                save_obj(trainer.netD_fullsup_direct.state_dict(), pt.join(experiment_dir, 'fullsup2_discriminator_' + '_state_dict.pt'))
        
    save_obj(generator_config, pt.join(experiment_dir, 'generator_config.pkl'))
    
    if gan_algo == 'SBGAN':
        plt.plot(trainer.losses_D_all_real, label="D_loss_real")
        plt.plot(trainer.losses_D_all_fake, label="D_loss_fake")
        plt.legend()
        plt.savefig(pt.join(experiment_dir, 'All_D_losses_real_and_fake.png'))
        plt.close()
        plt.plot(trainer.losses_D_all, label="D_loss")
        plt.savefig(pt.join(experiment_dir, 'All_D_losses.png'))
        plt.close()
        for i in range(len(trainer.netDs)):
            plt.plot(trainer.losses_history['D_loss_fake_' + str(i) + "_for_G_0"], label="D_loss_fake"+str(i) + "for_G_0")
            plt.plot(trainer.losses_history['D_loss_real_' + str(i) + "_for_G_0"], label="D_loss_real"+str(i) + "for_G_0")
            plt.plot(trainer.losses_history['G_loss' + str(i)], label="G loss")
            plt.plot(trainer.losses_history['D_loss_' + str(i) + "_for_G_0"], label="D loss"+str(i))
            plt.legend(loc="upper right")
            plt.savefig(pt.join(experiment_dir, 'Losses_wie_SWGAN.png'))
            plt.close()  
        for i in range(len(trainer.netDs)):
            plt.plot(trainer.losses_history['D_loss_fake_' + str(i) + "_for_G_0_full"], label="D_loss_fake"+str(i) + "for_G_0_full")
            plt.plot(trainer.losses_history['D_loss_real_' + str(i) + "_for_G_0_full"], label="D_loss_real"+str(i) + "for_G_0_full")
            plt.plot(trainer.losses_history['D_loss_' + str(i) + "_for_G_0_full"], label="D loss"+str(i)+"_full")
            plt.legend(loc="upper right")
            plt.savefig(pt.join(experiment_dir, 'D_Losses_full.png'))
            plt.close()  
        for i in range(len(trainer.netDs)):
            plt.plot(trainer.losses_history['total_loss' + str(i)], label="total_loss"+str(i))
        plt.legend(loc="upper right")
        plt.savefig(pt.join(experiment_dir, 'sbgan_loss_all_discriminators.png'))
        plt.close()
        for i in range(len(trainer.netGs)):
            plt.plot(trainer.losses_history['sigw1_loss_G' + str(i)], label="sigw1_loss"+str(i), alpha=0.8)
        plt.grid()
        plt.yscale('log')
        plt.legend(loc="upper right")
        plt.savefig(pt.join(experiment_dir, 'sig_losses.png'))
        plt.close()
        if fullsup:
            plt.plot(trainer.losses_history['Fullsup'], label="Fullsup")
            if netD_fullsup_direct != None:
                plt.plot(trainer.losses_history['Fullsup_direct'], label="Fullsup_direct")
        for i in range(len(trainer.netDs)):
            plt.plot(trainer.losses_history['err_sup_' + str(i) + "_for_G_0"], label="Labelled"+str(i))
        plt.legend(loc="upper right")
        plt.savefig(pt.join(experiment_dir, 'BCEwithlogits_TRAIN.png'))
        plt.close()

        for i in range(len(trainer.netDs)):
            if fullsup:
                plt.plot(trainer.losses_history['err_fullsup_fake_test'], label="fullsup_test")
                if netD_fullsup_direct != None:
                    plt.plot(trainer.losses_history['err_fullsup_direct_fake_test'], label="fullsup_direct_test")
            plt.plot(trainer.losses_history['err_sup_fake_test' + str(i)], label='sup_test'+str(i))
            plt.legend(loc="upper right")
            plt.savefig(pt.join(experiment_dir, 'BCEwithlogits_OOS_D'+str(i)+' .png'))
            plt.close()
            if fullsup:
                plt.plot(trainer.losses_history['Fullsup_F1'], label="fullsup_f1")
            plt.plot(trainer.losses_history['gan_f1' + str(i)], label='gan_f1'+str(i))
            plt.legend(loc="lower right")
            plt.savefig(pt.join(experiment_dir, 'F1_IS_D'+str(i)+' .png'))
            plt.close()
            if fullsup:
                plt.plot(trainer.losses_history['Fullsup_WAUC'], label="fullsup_wauc")
            plt.plot(trainer.losses_history['gan_wauc' + str(i)], label='gan_wauc'+str(i))
            plt.legend(loc="lower right")
            plt.savefig(pt.join(experiment_dir, 'WAUC_IS_D'+str(i)+' .png'))
            plt.close()
            if fullsup:
                plt.plot(trainer.losses_history['Fullsup_F1_fake_test'], label="fullsup_f1_test")
                if netD_fullsup_direct != None:
                    plt.plot(trainer.losses_history['Fullsup_direct_F1_fake_test'], label="fullsup_direct_f1_test")
            plt.plot(trainer.losses_history['gan_f1_fake_test' + str(i)], label='gan_f1_test'+str(i))
            plt.legend(loc="lower right")
            plt.savefig(pt.join(experiment_dir, 'F1_OOS_D'+str(i)+' .png'))
            plt.show()
            plt.close()
            if fullsup:
                plt.plot(trainer.losses_history['Fullsup_WAUC_fake_test'], label="fullsup_wauc_test")
                if netD_fullsup_direct != None:
                    plt.plot(trainer.losses_history['Fullsup_direct_WAUC_fake_test'], label="fullsup_direct_wauc_test")
            plt.plot(trainer.losses_history['gan_wauc_fake_test' + str(i)], label='gan_wauc_test'+str(i))
            plt.legend(loc="lower right")
            plt.savefig(pt.join(experiment_dir, 'WAUC_OOS_D'+str(i)+' .png'))
            plt.show()
            plt.close()

            if fullsup:
                plt.plot(trainer.losses_history['err_fullsup_fake_test_misclass'], label="fullsup_test_misclass")
                if netD_fullsup_direct != None:
                    plt.plot(trainer.losses_history['err_fullsup_direct_fake_test_misclass'], label="fullsup_direct_test_misclass")
            plt.plot(trainer.losses_history['err_sup_fake_test_misclass' + str(i)], label='sup_test_misclass'+str(i))
            plt.legend(loc="upper right")
            plt.savefig(pt.join(experiment_dir, 'Misclass_OOS_D'+str(i)+' .png'))
            plt.close()
       
            if fullsup:
                plt.plot(trainer.losses_history['Fullsup_BCELoss'], label="Fullsup_BCELoss")
            plt.plot(trainer.losses_history['GAN_BCELoss' + str(i)], label='GAN_BCELoss'+str(i))
            plt.legend(loc="upper right")
            plt.savefig(pt.join(experiment_dir, 'BCELoss_IS_D'+str(i)+' .png'))
            plt.close()

            if fullsup:
                plt.plot(trainer.losses_history['Fullsup_Misclass'], label="Fullsup_Misclass")
            plt.plot(trainer.losses_history['GAN_Misclass' + str(i)], label='GAN_Misclass'+str(i))
            plt.legend(loc="upper right")
            plt.savefig(pt.join(experiment_dir, 'Misclass_IS_D'+str(i)+' .png'))
            plt.close()
            
        for l in range(len(trainer.netGs)):
            for k in range(len(trainer.netDs)):
                plt.plot(trainer.losses_history['D_loss_fake_' + str(k) + "_for_G_" + str(l)], label="D_loss_fake"+str(k) + "for_G_" + str(l))
                plt.plot(trainer.losses_history['D_loss_real_' + str(k) + "_for_G_" + str(l)], label="D_loss_real"+str(k) + "for_G_" + str(l))
                plt.plot(trainer.losses_history['total_loss_' + str(k) + "_for_G_" + str(l)], label="Total_loss"+str(k) + "for_G_" + str(l))
                plt.legend(loc="upper right")
                plt.savefig(pt.join(experiment_dir, 'D_' + str(k) + '_all_big_losses_for_G_'+ str(l)+'.png'))
                plt.close()           
                plt.plot(trainer.losses_history['err_sup_' + str(k) + "_for_G_" + str(l)], label="err_sup"+str(k) + "for_G_" + str(l))
                plt.plot(trainer.losses_history['WGAN_GP_' + str(k) + "_for_G_" + str(l)], label="WGAN_GP"+str(k) + "for_G_" + str(l))
                if trainer.bayes:
                    plt.plot(trainer.losses_history['errD_noise_' + str(k) + "_for_G_" + str(l)], label="errD_noise_"+str(k) + "_for_G_" + str(l))
                    plt.plot(trainer.losses_history['errD_prior_' + str(k) + "_for_G_" + str(l)], label="errD_prior_"+str(k) + "_for_G_" + str(l))
                plt.legend(loc="upper right")
                plt.savefig(pt.join(experiment_dir, 'D_' + str(k) + '_all_additional_losses_for_G_'+ str(l)+'.png'))
                plt.close()
                plt.plot(trainer.losses_history['err_sup_' + str(k) + "_for_G_" + str(l)], label="err_sup"+str(k) + "for_G_" + str(l))
                if trainer.bayes:
                    plt.plot(trainer.losses_history['errD_noise_' + str(k) + "_for_G_" + str(l)], label="errD_noise_"+str(k) + "_for_G_" + str(l))
                    plt.plot(trainer.losses_history['errD_prior_' + str(k) + "_for_G_" + str(l)], label="errD_prior_"+str(k) + "_for_G_" + str(l))
                plt.legend(loc="upper right")
                plt.savefig(pt.join(experiment_dir, 'D_' + str(k) + '_all_additional_without_gp_losses_for_G_'+ str(l)+'.png'))
                plt.close()
                if trainer.bayes:
                    plt.plot(trainer.losses_history['errD_noise_' + str(k) + "_for_G_" + str(l)], label="errD_noise_"+str(k) + "_for_G_" + str(l))
                    plt.plot(trainer.losses_history['errD_prior_' + str(k) + "_for_G_" + str(l)], label="errD_prior_"+str(k) + "_for_G_" + str(l))
                    plt.legend(loc="upper right")
                    plt.savefig(pt.join(experiment_dir, 'D_' + str(k) + '_bayes_losses_for_G_'+ str(l)+'.png'))
                    plt.close()
            if trainer.bayes:
                plt.plot(trainer.losses_history['errG_noise_for_G_' + str(l)], label='errG_noise_for_G_' + str(l))
                plt.plot(trainer.losses_history['errG_prior_for_G_' + str(l)], label='errG_prior_for_G_' + str(l))
            plt.legend(loc="upper right")
            plt.savefig(pt.join(experiment_dir, 'G_all_additional_losses_for_G_'+ str(l)+'.png'))
            plt.close()


    if gan_algo == "SBGAN": 
        if generator_config['generator_type'] != 'ResFNN_gen':
            x_fake_all_list = []
            for ind in range(len(netGs)):
                with torch.no_grad():
                    x_fake = netGs[ind](1024, n_lags, device)
                
                random_indices = torch.randint(0, x_fake.shape[0], (250,))
                x_fake_all_list.append(x_fake[random_indices, :, :])
                for i in range(x_real_dim):
                    plt.plot(to_numpy(x_fake.cpu()[0:250, :, i]).T, 'C%s' % i, alpha=0.1)
                plt.savefig(pt.join(experiment_dir, 'x_fake_' + str(ind) + '.png'))
                plt.close()
            x_fake_all = torch.cat(x_fake_all_list)
            for i in range(x_real_dim):
                plt.plot(to_numpy(x_fake_all.cpu()[:, :, i]).T, 'C%s' % i, alpha=0.1)
            plt.savefig(pt.join(experiment_dir, 'x_fake_all.png'))
            plt.close()
    
            for i in range(len(x_fake_all_list)):
                plt.plot(to_numpy(x_fake_all_list[i].cpu()[:, :, 0]).T, 'C%s' % i, alpha=0.1, label="G "+str(i))
                plt.legend(loc="upper right")
            plt.savefig(pt.join(experiment_dir, 'x_fake_all_1.png'))
            plt.show()
            plt.close()
    
    if gan_algo == 'SBGAN':
        save_obj(discriminator_config, pt.join(experiment_dir, 'discriminator_config.pkl'))
        if generator_config['generator_type'] != 'ResFNN_gen':
            x_fake_augmented = apply_augmentations(x_fake, aug)
            x_fake_logsig = signatory.logsignature(x_fake_augmented, depth=depth)
            x_fake_signature = signatory.signature(x_fake_augmented, depth=depth)
        else:
            x_fake = netGs[0](acts_test.shape[0], n_lags, device, embeddings=acts_test.to(device))
            x_fake_logsig = x_fake
            print("x_fake: ", x_fake.shape, x_fake[:10, :5])
            if len(netGs)>1:
                x_fake2 = netGs[1](acts_test.shape[0], n_lags, device, embeddings=acts_test.to(device))
                print("x_fake2: ", x_fake2.shape, x_fake2[:10, :5])
        
        plot_signature(x_real_signature.mean(0))
        plt.savefig(pt.join(experiment_dir, 'sig_real.png'))
        plt.close()
        if generator_config['generator_type'] != 'ResFNN_gen':
            plot_signature(x_real_signature.mean(0))
            plot_signature(x_fake_signature.mean(0))
            plt.savefig(pt.join(experiment_dir, 'sig_real_fake.png'))
            plt.close()
        aug = gan_config['augmentations']
        depth = gan_config['depth']
        if sup == True:
            gan_config['sup'] = True
        if type(x_real) == list:
            x_real_logsig = x_real_logsig #sig_list(x_real_rolled, aug)
            
        else:
            x_real_augmented = apply_augmentations(x_real_rolled, aug)
            x_real_logsig = signatory.logsignature(x_real_augmented, depth=depth)
            
        plot_signature(x_real_logsig.mean(0))
        plt.savefig(pt.join(experiment_dir, 'logsig_real.png'))
        plt.close()
        
        plot_signature(x_real_logsig.mean(0))
        plot_signature(x_fake_logsig.mean(0))
        plt.savefig(pt.join(experiment_dir, 'logsig_real_fake.png'))
        plt.close()

        plt.plot(x_real_logsig[:100,:].cpu().detach().numpy().T, color="blue", alpha=0.3, label="real")
        plt.plot(x_fake_logsig[:100,:].cpu().detach().numpy().T, color="orange", alpha=0.3, label="fake")
        plt.savefig(pt.join(experiment_dir, 'logsig_real_fake_2.png'))
        plt.show()
        plt.close()

        plt.plot(x_fake_logsig[:100,:].cpu().detach().numpy().T, color="orange", alpha=0.3)
        plt.savefig(pt.join(experiment_dir, 'logsig_fake_2.png'))
        plt.close()

        print(x_real_logsig.shape)
        xmin,_ = x_real_logsig.min(dim=0)
        xmax,_ = x_real_logsig.max(dim=0)
        dx = xmax-xmin 
        plot_signature((x_real_logsig.mean(0)-xmin)/dx)
        plt.savefig(pt.join(experiment_dir, 'logsig_real_norm.png'))
        plt.close()
        
        plot_signature((x_real_logsig.mean(0)-xmin)/dx)
        plot_signature((x_fake_logsig.mean(0).cpu()-xmin)/dx)
        plt.savefig(pt.join(experiment_dir, 'logsig_real_fake_norm.png'))
        plt.close()
        


##############################
experiment_dir = "/results" #path to save results
datadir = "/data" #path to data


size = 256
def train_test_list(labels, rate, random_state=123):
    np.random.seed(random_state)
    random.seed(random_state)
    l = labels.clone() - 1
    non_zero_ind, zero_ind = labels.nonzero()[:,0], l.nonzero()[:,0]
    
    zi_test = np.random.choice(zero_ind, size = int(rate*  len(zero_ind)), replace=False)
    zi_train = zero_ind[~np.isin(zero_ind, zi_test)]
    nzi_test = np.random.choice(non_zero_ind, size = int(rate * len(non_zero_ind)), replace=False)
    nzi_train = non_zero_ind[~np.isin(non_zero_ind, nzi_test)]
    train_ind, test_ind = np.append(zi_train, nzi_train), np.append(zi_test, nzi_test)
    return train_ind, test_ind

def normalize_data(list_in):
    m0 = max([l[:, 0].diff().max() for l in list_in])
    m1 = max([l[:, 1].max() for l in list_in])
    x_norm = []
    for t in list_in:
        updated_tensor = t.clone()
        updated_tensor[1:, 0] = updated_tensor[:, 0].diff()/m0
        updated_tensor[1:, 1] = (updated_tensor[:, 1]/m1)[1:]
        x_norm.append(updated_tensor[1:,:])
    return x_norm

x = torch.load(datadir + '/x_transactions.pt')
#x_sig = torch.load(datadir + '/x_sig.pt')
x_sig = torch.load(datadir + '/x_logsig.pt')

x_logsig = torch.load(datadir + '/x_logsig.pt')
x = normalize_data(x)
acts = torch.stack(torch.load(datadir + '/emb_transactions.pt'))
labels = torch.stack(torch.load(datadir + '/labels_transactions.pt')).to(int)


indices_train, indices_test = train_test_list(labels, 0.1, random_state=42)
x_test, y_test, acts_test = [x[i] for i in indices_test], torch.stack([labels[i] for i in indices_test]), acts[indices_test,:]
x_train, y_train, acts_train = [x[i] for i in indices_train], torch.stack([labels[i] for i in indices_train]), acts[indices_train,:]
x_sig_train, x_logsig_train = x_sig[indices_train,:], x_logsig[indices_train,:]  #x_sig[indices_train,:], x_logsig[indices_train,:]

numG = [1] 
numD = [1] 

for j in range(len(numD)): 
    for i in range(len(numG)):
        torch.cuda.empty_cache()
        rates = np.array([0.999, 0.995])#, 0.9925, 0.95])# #0.999, 0.995, 0.9925, 0.99, 0.975, 0.95])#, 0.95])#, 0.8])#, 0.95, 0.90, 0.85])#, 0.95, 0.9])#, 0.95, 0.9]) #, 0.95, 0.85]) # 0.95
        for r in rates:
            random_state = [42, 54]#, 54, 65]#, 71, 25] 
            for rand in random_state:
                
                indices_labeled, _ = train_test_list(y_train, r, random_state=rand)
                x_labelled, acts_labelled = [x_train[i] for i in indices_labeled], acts_train[indices_labeled,:]
                labels = torch.stack([y_train[i] for i in indices_labeled])
                x_labelled_test = x_test
                labels_test = y_test
                x_labelled_logsig = x_logsig_train[indices_labeled,:]
                x_labelled_test_logsig = x_logsig[indices_test,:]
                
                lr_d_list = np.array([0.001])#0.001])#, 0.003])
                lr_g_list = np.array([0.0001])
                batch_sizes = [2048]
                errsups = [10] 
                for es in errsups:
                    for bs in batch_sizes:
                        for lr_d in lr_d_list:
                            for lr_g in lr_g_list:
                                print("start training: ", r, rand, es, bs, lr_d, lr_g)
                                run_sbgan(datasets=('fraud',), generators=('ResFNN_gen',), n_seeds=1, device='cpu', logret=False, numG=numG[i], numD=numD[j], x_labelled=x_labelled, fullsup=True, x_fake_test_labelled=x_labelled_test, sup=True, x_real=x_train, rate=r, y_real=None, lr_d=lr_d, lr_g=lr_g, l_errsup=es, random_state=rand, labels=labels, labels_test=labels_test, acts=acts, acts_labelled=acts_labelled, acts_test=acts_test, bs=bs, x_sig=x_sig_train, x_logsig=x_logsig_train, x_labelled_logsig=x_labelled_logsig, x_fake_test_labelled_logsig=x_labelled_test_logsig) 
                                    