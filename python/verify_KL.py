import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import scipy.stats as stats

## comparison 
def comparison(distq, distp):
    """
    Numerical Solution (Monte Carlo)
    analytic: distq.kl_divergence(distp).numpy()
    p_value: 0.1
    """
    MC= []
    for i in range(1000):
        M = 1000
        xsamp  = distq.sample(M)
        MC.append(tf.reduce_mean(distq.log_prob(xsamp) - distp.log_prob(xsamp)).numpy())
    if stats.ttest_1samp( MC, popmean= analytic)[1]> p_value:
        print('they are same!')
        print(' ')
    else:
         print('they are not the same!')


p_value = 0.1
tfpd = tfp.distributions
a_loc, a_scale = 1,4
b_loc, b_scale =4,1

#### q: normal- p:normal
distq = tfpd.Normal(a_loc, a_scale)
distp = tfpd.Normal(b_loc, b_scale)
analytic = distq.kl_divergence(distp).numpy()
print('q: normal- p:normal')
comparison(distq, distp)

#### q: Cauchy- p:Cauchy
distq = tfpd.Cauchy(a_loc, a_scale)
distp = tfpd.Cauchy(b_loc, b_scale)
analytic = distq.kl_divergence(distp).numpy()
print('q: Cauchy- p:Cauchy')
comparison(distq, distp)

#### q: Laplace- p:Laplace
distq = tfpd.Laplace(a_loc, a_scale)
distp = tfpd.Laplace(b_loc, b_scale)
analytic = distq.kl_divergence(distp).numpy()
print('q: Laplace- p:Laplace')
comparison(distq, distp)

#### q: HalfNormal- p:HalfNormal
distq = tfpd.HalfNormal(a_loc, a_scale)
distp = tfpd.HalfNormal(b_loc, b_scale)
analytic = distq.kl_divergence(distp).numpy()
print('q: HalfNormal- p:HalfNormal')
comparison(distq, distp)


#### q: Gamma- p:Gamma
distq = tfpd.Gamma(concentration=3.0, rate=2.0)
distp = tfpd.Gamma(concentration=2, rate=3)
analytic = distq.kl_divergence(distp).numpy()
xsamp  = tf.random.gamma([30], alpha=2, beta=3)
MC= []
MC.append(tf.reduce_mean(distq.log_prob(xsamp) - distp.log_prob(xsamp)).numpy())
if stats.ttest_1samp( MC, popmean= analytic)[1]> p_value:
    print('q: Gamma- p:Gamma')
    print('they are same!')


#### q: LogNormal- p:LogNormal
distq = tfpd.LogNormal(a_loc, a_scale)
distp = tfpd.LogNormal(b_loc, b_scale)
analytic = distq.kl_divergence(distp).numpy()
print('q: LogNormal- p:LogNormal')
comparison(distq, distp)

#### q: Beta- p:Beta
distq = tfpd.Beta(a_loc, a_scale)
distp = tfpd.Beta(b_loc, b_scale)
analytic = distq.kl_divergence(distp).numpy()
MC= []
for i in range(1000):
    xsamp  = np.random.beta(5,6)
    MC.append(tf.reduce_mean(distq.log_prob(xsamp) - distp.log_prob(xsamp)).numpy())
if stats.ttest_1samp( MC, popmean= analytic)[1]> p_value:
    print('q: Beta- p:Beta')
    print('they are same!')

#### q: LogitNormal- p:LogitNormal
distq = tfpd.LogitNormal(a_loc, a_scale)
distp = tfpd.LogitNormal(4., 1.4)
analytic = distq.kl_divergence(distp).numpy()
print('q: LogitNormal- p:LogitNormal')
comparison(distq, distp)

#### q: Uniform- p:Uniform
distq = tfpd.Uniform(a_loc, a_scale)
distp = tfpd.Uniform(b_loc, b_scale)
analytic = distq.kl_divergence(distp).numpy()
print('q: Uniform- p:Uniform')
comparison(distq, distp)


#### a:Normal -p:Laplace  KL(a || b)
import math
import scipy.stats
#### a:Normal -p:Laplace  KL(a || b)
distq = tfpd.Normal(a_loc, a_scale)
distp = tfpd.Laplace(b_loc, b_scale)
p = scipy.stats.norm(0, a_scale).cdf(abs(a_loc-b_loc))- scipy.stats.norm(0, a_scale).cdf(-abs(a_loc-b_loc))
analytic = -0.5*(math.log(2*math.pi*(a_scale**2))+1) + math.log(2*b_scale)+(abs(a_loc-b_loc)/b_scale)* p + \
    (2*a_scale)*(math.exp(-((a_loc-b_loc)**2)/(2*(a_scale**2))))/(b_scale*((2*math.pi)**0.5))
print('a:Normal -p:Laplace')
comparison(distq, distp)

#### q:Normal -p:Cauchy  KL(a || b)
distq = tfpd.Normal(a_loc, a_scale)
distp = tfpd.Laplace(b_loc, b_scale)
analytic = math.log(math.sqrt(0.5*math.pi))+0.5*( a_scale+a_loc**2)-0.5*(1+math.log(2*math.pi*(a_scale**2)))
print('q:Normal -p:Cauchy')
comparison(distq, distp)

#### q:Laplace - p: Normal  KL(a || b)
distq = tfpd.Laplace(a_loc, a_scale)
distp = tfpd.Normal(b_loc, b_scale)
analytic = -1-math.log(2*a_scale)+ ((((a_loc-b_loc)**2)+(2*a_scale**2))/ (2*(b_scale**2))) + math.log(math.sqrt(2*math.pi)* b_scale)
print('q:Laplace - p: Normal')
comparison(distq, distp)

#### q:Laplace - p:Cauchy
distq = tfpd.Laplace(a_loc, a_scale)
distp = tfpd.Cauchy(b_loc, b_scale)
x, aloc, ascale, bloc, bscale = symbols('x aloc ascale bloc bscale')
analytic = log(pi)+ integrate((log(bscale/((x-bloc)**2+bscale**2)))*exp(-1*abs(x-aloc)/a_scale)/(2*ascale), x)
analytic
#print('q:Laplace - p: Normal')
#comparison(distp, distq)