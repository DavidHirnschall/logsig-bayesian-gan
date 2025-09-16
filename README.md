# Semi-Supervised Bayesian GANs with Log-Signatures for Uncertainty-Aware Credit Card Fraud Detection
This is the github repository for the corresponding article [arXiv](https://arxiv.org/abs/2509.00931)

# Overview
We present a novel deep generative semi-supervised framework for credit card fraud detection, formulated as time series classification task.
	As financial transaction data streams grow in scale and complexity, traditional methods often require large labeled datasets, struggle with time series of irregular sampling frequencies and varying sequence lengths.
	To address these challenges, we extend conditional Generative Adversarial Networks (GANs) for targeted data augmentation, integrate Bayesian inference to obtain predictive distributions and quantify uncertainty, and leverage log-signatures for robust feature encoding of transaction histories. We introduce a novel Wasserstein distance-based loss to align generated and real unlabeled samples while simultaneously maximizing classification accuracy on labeled data. Our approach is evaluated on the BankSim dataset, a widely used simulator for credit card transaction data, under varying proportions of labeled samples, demonstrating consistent improvements over benchmarks in both global statistical and domain-specific metrics.
	These findings highlight the effectiveness of GAN-driven semi-supervised learning with log-signatures for irregularly sampled time series and emphasize the importance of uncertainty-aware predictions.
