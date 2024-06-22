import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import wandb

def estimate_distributions(hist_pos, hist_neg):
    # Estimate the distributions using Kernel Density Estimation
    kde_pos = gaussian_kde(hist_pos)
    kde_neg = gaussian_kde(hist_neg)
    
    return kde_pos, kde_neg

def inference_reliability(hist_pos, hist_neg):
    # Estimate the distributions
    kde_pos, kde_neg = estimate_distributions(hist_pos, hist_neg)
    
    # Define the range of similarity scores
    min_score = min(hist_pos.min(), hist_neg.min())
    max_score = max(hist_pos.max(), hist_neg.max())
    scores = np.linspace(min_score, max_score, 1000)
    
    # Evaluate the KDEs on the scores
    pos_density = kde_pos(scores)
    neg_density = kde_neg(scores)
    
    # Calculate class priors
    prior_pos = len(hist_pos) / (len(hist_pos) + len(hist_neg))
    prior_neg = len(hist_neg) / (len(hist_pos) + len(hist_neg))
    
    # Calculate reliability as the posterior probability with priors
    pos_prob = pos_density * prior_pos
    neg_prob = neg_density * prior_neg
    reliability = pos_prob / (pos_prob + neg_prob)
    
    return scores, reliability

def log_inference_reliability(hist_pos, hist_neg, title):
    scores, reliability = inference_reliability(hist_pos, hist_neg)
    table = wandb.Table(data=[[scores[i], reliability[i]] for i in range(len(scores))], columns=["score", "reliability"])
    plot = wandb.plot.line(table, x='score', y='reliability', title=title)
    wandb.log({title: plot})  

if __name__=='__main__':
    hist_pos = np.random.normal(0, 0.1, 100)
    hist_neg = np.random.normal(0, 0.1, 300)

    scores, reliability = inference_reliability(hist_pos, hist_neg)

    plt.figure(figsize=(10, 6))
    plt.plot(scores, reliability, label='Reliability')
    plt.title('Inference Reliability Distribution')
    plt.xlabel('Similarity Score')
    plt.ylabel('Reliability')
    plt.legend()
    plt.show()
