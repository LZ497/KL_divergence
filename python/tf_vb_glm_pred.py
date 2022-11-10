#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
#from django.test import TestCase
from six.moves import urllib
from tqdm import tqdm
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
import math

def load_and_preprocess_radon_dataset(radon_data, state):
  df = radon_data[radon_data.state==state.encode()].copy()
  df['radon'] = df.activity.apply(lambda x: x if x > 0. else 0.1)
  # Make county names look nice.
  df['county'] = df.county.apply(lambda s: s.decode()).str.strip().str.title()
  # Remap categories to start from 0 and end at max(category).
  df['county'] = df.county.astype(pd.api.types.CategoricalDtype())
  df['county_code'] = df.county.cat.codes
  # Radon levels are all positive, but log levels are unconstrained
  df['log_radon'] = df['radon'].apply(np.log)
  # Drop columns we won't use and tidy the index
  columns_to_keep = ['log_radon', 'floor', 'county', 'county_code']
  df = df[columns_to_keep].reset_index(drop=True)
  return df

### That part was copied from the tensorflow tutorial.

def Normal(a,b,name=None):
    return tfd.Normal(loc=a, scale=b, name=name)

def Laplace(a,b,name=None):
    return tfd.Laplace(loc=a, scale=b, name=name)

def LogNormal(a,b,name=None):
    return tfd.LogNormal(loc=a,scale=b, name=name)

### Define the priors
def define_priors(f):
    global priors
    priors = {}
# This is the random effects variance
# This would match the tutorial:
    priors['county_scale'] = f(0,1,'scale_prior')
    # But I want to use lognormal
    priors['intercept'] = f(0,1, name='intercept')
    # This is the regression coefficient
    priors['floor_weight'] = f(0,1, name='floor_weight')
    
def define_priors_hb(f):
    priors['county_scale'] = f(0,1, name='scale_prior')
    priors['obs_lik'] = f(0,1, name='obs_lik')

### Define the likelihood
# y - The log_radon level, the output to predict. (DATA)
# floor - A binary variable: first floor or second floor? (DATA)
# county - An integer coding for  the county of the observation? (DATA)
# y, floor and county should be vectors of the same length.
# ps - A dictionary with elements giving a sample of each parameter.
#def log_lik(y, floor, county, county_scale, intercept, floor_weight, county_prior):
def log_lik(y, floor, county, ps):
    # Get the appropriate county random effect for each observation.
    random_effect = tf.gather(ps['county_prior'], county, axis=-1)
    # Get fixed effects for each observation.
    fixed_effect = ps['intercept'][:,tf.newaxis] + ps['floor_weight'][:,tf.newaxis] * floor[tf.newaxis,:]
    # Combine.
    linear_response = fixed_effect + random_effect
    # For some reason, this tutorial fixed the variance of the response to 1.
    obs_lik = tfd.Normal(loc=linear_response, scale= ps['obs_lik'][:,tf.newaxis], name='likelihood')
    obs_ll = tf.reduce_sum(obs_lik.log_prob(y), axis = 1)
    ret = obs_ll 
    return ret

## Variational-Prior KL divergence for county_prior is hierarchical; need to estimate with Monte Carlo.
def re_vp(ps):
    # This is the vector of random effects for county effects.
    re_prior = tfd.Normal(loc=tf.zeros(df.county.nunique()), scale=ps['county_scale'][:,tf.newaxis], name='county_prior')
    re_lp = tf.reduce_sum(re_prior.log_prob(ps['county_prior']), axis = 1)
    re_lv = tf.reduce_sum(vd['county_prior'].log_prob(ps['county_prior']), axis = 1)
    return(re_lv - re_lp)

#### Define variational parameters
def define_vp():
    global vp
    vp = {}
    zero = tf.constant(0, dtype = tf.float32)
    one = tf.constant(1, dtype = tf.float32)
    vp['county_scale_loc'] = tf.Variable(zero)
    vp['county_scale_scale'] = tfp.util.TransformedVariable(one, bijector=tfb.Softplus())
    vp['intercept_loc'] = tf.Variable(zero)
    vp['intercept_scale'] = tfp.util.TransformedVariable(one, bijector=tfb.Softplus())
    vp['floor_weight_loc'] = tf.Variable(zero)
    vp['floor_weight_scale'] = tfp.util.TransformedVariable(one, bijector=tfb.Softplus())
    vp['floor_weight_loc'] = tf.Variable(zero)
    vp['floor_weight_scale'] = tfp.util.TransformedVariable(one, bijector=tfb.Softplus())
    vp['county_prior_loc'] = tf.Variable(tf.zeros(df.county.nunique()))
    vp['county_prior_scale'] = tfp.util.TransformedVariable(tf.ones(df.county.nunique()), bijector=tfb.Softplus())
    vp['obs_lik_loc'] = tf.Variable(zero)
    vp['obs_lik_scale'] = tfp.util.TransformedVariable(one, bijector=tfb.Softplus())

## Transformed variables can't be targets of autodiff; we need to extract the underlying untransformed vars.
def define_train_vars():
    global train_vars
    train_vars = {}
    for v in vp.keys():
        # It it's a scale parameter, extract the underlying trainable variable.
        if v.split('_')[-1]=='scale':
            train_vars[v] = vp[v].trainable_variables[0]
        else:
            train_vars[v] = vp[v]


#### Define variational distributions
def define_vd(f):
    global vd
    vd = {}
    vd['intercept'] = f(a=vp['intercept_loc'], b=vp['intercept_scale'],name='intercept')
    vd['floor_weight'] = f(a=vp['floor_weight_loc'], b=vp['floor_weight_scale'],name = 'floor_weight')
    vd['county_prior'] = f(a=vp['county_prior_loc'], b=vp['county_prior_scale'],name='county_prior')
    
def define_vd_hb(f):
    global vd
    vd['county_scale'] = f(a= vp['county_scale_loc'], b= vp['county_scale_scale'],name='county_scale')
    vd['obs_lik'] = f(a=vp['obs_lik_loc'],b=vp['obs_lik_scale'],name='obs_lik')

def optimization(features,labels):
    global vd
    #### Main optimization loop.
    opt = tf.optimizers.Adam(learning_rate=1e-2)
    M = 3 # Monte Carlo sample size
    iters = 30
    costs = np.zeros(iters)
    for it in tqdm(range(iters)):
        # Compute stochastic gradient estimate.
        with tf.GradientTape() as gt:
            ## Sample some parameters from our variational distribution
            # Each elemnt of ps will be of shape M times the shape of the variable.
            ps = {}
            for v in vd.keys():
                ps[v] = vd[v].sample(M)
            
            ## Form a Monte-Carlo estimate of the likelihood.
            ll_draws = log_lik(labels, features['floor'], features['county_code'], ps)
            ll_mc = tf.reduce_mean(ll_draws)

            ## Compute closed-form Variational-Prior KL divergence
            klvp = 0.
            for v in priors.keys():
                klvp += vd[v].kl_divergence(priors[v])

            ## Variational-Prior KL divergence for county_prior is hierarchical; need to estimate with Monte Carlo.
            klvp += tf.reduce_mean(re_vp(ps))

            if ll_mc and klvp:
                cost = -ll_mc + klvp
                grad = gt.gradient(cost, train_vars)

        # Gradient descent step.
                gnv = [(grad[v], train_vars[v]) for v in train_vars.keys()]
                opt.apply_gradients(gnv)

                costs[it] = cost.numpy()
            else:
                pass
    print("costs:")
    print(costs)

def predict(floor, county, vd):
    random_effect = tf.gather(vd['county_prior'], county, axis=-1)
    # Get fixed effects for each observation.
    fixed_effect = vd['intercept'][:,tf.newaxis] + vd['floor_weight'][:,tf.newaxis] * floor[tf.newaxis,:]
    linear_response = fixed_effect + random_effect
    # For some reason, this tutorial fixed the variance of the response to 1.
    obs_lik = tfd.Normal(loc=linear_response, scale= vd['obs_lik'][:,tf.newaxis], name='likelihood')
    preds = obs_lik.sample(1000)
    return preds

def get_rmse(state,priors_dis_ub,vd_dis_ub, priors_dis_hb='LogNormal' ,vd_dis_hb='LogNormal' ):
    ds = tfds.load('radon', split='train')
    radon_data = tfds.as_dataframe(ds)
    radon_data.rename(lambda s: s[9:] if s.startswith('feat') else s, axis=1, inplace=True)
    global df
    df = load_and_preprocess_radon_dataset(radon_data,state)

    #shuffle and split
    shuffle_df = df.sample(frac=1)
    train_size = int(0.7 * len(shuffle_df))
    # Split your dataset 
    df = shuffle_df[:train_size]
    test_df = shuffle_df[train_size:]

    # get features and labels
    features = df[['county_code', 'floor']].astype(int)
    labels = df[['log_radon']].astype(np.float32).values.flatten()

    if priors_dis_ub=='Normal':
        define_priors(Normal)
    if priors_dis_ub=='Laplace':
        define_priors(Laplace)
    if priors_dis_hb =='LogNormal':
        define_priors_hb(LogNormal)

    define_vp()
    define_train_vars()

    if vd_dis_ub=='Normal':
        define_vd(Normal)
    if vd_dis_ub=='Laplace':
        define_vd(Laplace)
    if vd_dis_hb=='LogNormal':
        define_vd_hb(LogNormal)
        
    optimization(features,labels)

    test_features = test_df[['county_code', 'floor']].astype(int)
    test_labels= test_df[['log_radon']].astype(np.float32).values.flatten()

    rmses = []
    m = 3
    for i in range(1000):
        samp = {}
        for v in vd.keys():
            samp[v] = vd[v].sample(m)

        prediction = predict(test_features['floor'], test_features['county_code'], samp)
        ##prediction shape (obs_lik.sample_number,m,len(labels))
        prediction = tf.reduce_mean(prediction, axis = 1)

        # Evaluate point estimate
        pred_point = tf.reduce_mean(prediction, axis = 0)
        mse = np.square(np.subtract(test_labels,pred_point)).mean() 
        rmse = math.sqrt(mse)
        rmses.append(rmse)

        # Evaluate confidence interval coverage
        conf_level = 0.95 #Nominal level
        lb = np.quantile(prediction, (1-conf_level)/2, axis = 0)
        ub = np.quantile(prediction, conf_level+(1-conf_level)/2, axis = 0)
        covered = np.logical_and((lb <= test_labels), (ub >= test_labels))
        obs_coverage = np.mean(covered)

    return sum(rmses)/len(rmses), obs_coverage

def main():
    states = ['AZ','IN','MA','MN','MO','ND','PA','R5']
    Normal_Normal, Laplace_Normal, Laplace_Laplace,Normal_Laplace =[],[],[],[]
    for state in states:
        try:
            Normal_Normal.append(get_rmse(state, priors_dis_ub='Normal' ,vd_dis_ub= 'Normal') )
        except:
            print('Normal_Normal_Failed:', state)
        try:
            Laplace_Normal.append(get_rmse(state, priors_dis_ub='Laplace' ,vd_dis_ub= 'Normal'))
        except:
            print('Laplace_Normal_Failed:', state)
        try:
            Laplace_Laplace.append( get_rmse(state, priors_dis_ub='Laplace' ,vd_dis_ub= 'Laplace'))
        except:
            print('Laplace_Laplace_Failed:', state)
        try:
            Normal_Laplace.append(get_rmse(state, priors_dis_ub='Normal' ,vd_dis_ub= 'Laplace'))
        except:
            print('Normal_Laplace_Failed:', state)

    print('Normal_Normal', np.array(Normal_Normal).mean(axis = 0))
    print('Laplace_Normal', np.array(Laplace_Normal).mean(axis = 0))
    print('Laplace_Laplace',np.array(Laplace_Laplace).mean(axis = 0))
    print('Normal_Laplace',np.array(Normal_Laplace).mean(axis = 0))
    
    ## Normal_Normal [1.77168336 0.6766805 ]
    ## Laplace_Normal [1.81702558 0.73415348]
    ## Laplace_Laplace [2.19840665 0.88227967]
    ## Normal_Laplace [2.11436831 0.80541125]

if __name__ == "__main__":
    main()
