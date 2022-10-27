#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tfpd = tfp.distributions

distq = tfpd.Normal(4., 1.4)
distp = tfpd.Normal(1., 1.1)

# Analytic Solution
distq.kl_divergence(distp)

# Numerical Solution (Monte Carlo)
M = 1000
xsamp  = distq.sample(M)

tf.reduce_mean(distq.log_prob(xsamp) - distp.log_prob(xsamp))
