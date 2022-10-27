#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
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

### This part copied from the tensorflow tutorial at https://www.tensorflow.org/probability/examples/Linear_Mixed_Effects_Model_Variational_Inference
def load_and_preprocess_radon_dataset(state='MN'):
  """Load the Radon dataset from TensorFlow Datasets and preprocess it.

  Following the examples in "Bayesian Data Analysis" (Gelman, 2007), we filter
  to Minnesota data and preprocess to obtain the following features:
  - `county`: Name of county in which the measurement was taken.
  - `floor`: Floor of house (0 for basement, 1 for first floor) on which the
    measurement was taken.

  The target variable is `log_radon`, the log of the Radon measurement in the
  house.
  """
  ds = tfds.load('radon', split='train')
  radon_data = tfds.as_dataframe(ds)
  radon_data.rename(lambda s: s[9:] if s.startswith('feat') else s, axis=1, inplace=True)
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

df = load_and_preprocess_radon_dataset()

features = df[['county_code', 'floor']].astype(int)
labels = df[['log_radon']].astype(np.float32).values.flatten()
### That part was copied from the tensorflow tutorial.

### Define the priors
priors = {}
# This is the random effects variance
# This would match the tutorial:
#priors['county_scale'] = tfd.HalfNormal(scale=1., name='scale_prior')
priors['county_scale'] = tfd.LogNormal(loc=0., scale=1., name='scale_prior')
# But I want to use lognormal
priors['intercept'] = tfd.Normal(loc=0., scale=1., name='intercept')
# This is the regression coefficient
priors['floor_weight'] = tfd.Normal(loc=0., scale=1., name='floor_weight')

priors['obs_lik'] = tfd.LogNormal(loc=0., scale=1., name='obs_lik')

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
    obs_lik = tfd.Normal(loc=linear_response, scale= ps['obs_lik'], name='likelihood')
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
train_vars = {}
for v in vp.keys():
    # It it's a scale parameter, extract the underlying trainable variable.
    if v.split('_')[-1]=='scale':
        train_vars[v] = vp[v].trainable_variables[0]
    else:
        train_vars[v] = vp[v]

#### Define variational distributions
vd = {}
vd['county_scale'] = tfd.LogNormal(loc=vp['county_scale_loc'], scale = vp['county_scale_scale'])
vd['intercept'] = tfd.Normal(loc=vp['intercept_loc'], scale = vp['intercept_scale'])
vd['floor_weight'] = tfd.Normal(loc=vp['floor_weight_loc'], scale = vp['floor_weight_scale'])
vd['county_prior'] = tfd.Normal(loc=vp['county_prior_loc'], scale = vp['county_prior_scale'])
vd['obs_lik'] = tfd.LogNormal(loc=vp['obs_lik_loc'], scale = vp['obs_lik_scale'])

#### Main optimization loop.
opt = tf.optimizers.Adam(learning_rate=1e-2)
M = 3 # Monte Carlo sample size
iters = 3000

costs = np.zeros(iters)
for it in tqdm(range(iters)):
    # Compute stochastic gradient estimate.
    with tf.GradientTape() as gt:
        ## Sample some parameters from our variational distribution
        # Each elemnt of ps will be of shape M times the shape of the variable.
        ps = {}
        for v in vd.keys():
            ps[v] = vd[v].sample(M)
        ps['obs_lik'] = vd['obs_lik'].sample(1)
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

#### Model viz and assessment
# Costs
fig = plt.figure()
plt.plot(costs)
plt.xlabel("Opt Iteration")
plt.ylabel("MC KL Estimate")
plt.title("Cost")
plt.tight_layout()
plt.savefig("costs.pdf")
plt.close()

# posterior hists
HS = 1000

fig = plt.figure()
plot_these = ['floor_weight','intercept','county_scale','obs_lik']
for vi,v in enumerate(plot_these):
    plt.subplot(int(np.ceil(len(plot_these)/2)),2,vi+1)
    samp = vd[v].sample(HS)
    plt.hist(samp)
    plt.title(v)
plt.tight_layout()
plt.savefig("hist.pdf")
plt.close()