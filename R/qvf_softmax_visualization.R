# QVF SCRIPT FOR VISUALIZING THE SOFTMAX AND CHANGES IN IT

# Setup ----
rm(list = ls()); # clear the workspace
setwd('/Users/sokolhessner/Documents/gitrepos/qvf/R/');

subj_val_diff = seq(from = -.5, to = .5, by = .001)

softmax_inv_temp_preveasy = 26.8 # the baseline mu value from CGE = 26.8; the baseline mu value from CGT = 25.0

softmax_inv_temp_fractions = c(.1, .2, .3, .4, .5, .6, .7, .8) # percent decrease in mu
n_alternatives = length(softmax_inv_temp_fractions)
softmax_inv_temp_values = array(dim = length(softmax_inv_temp_fractions))

for(t in 1:n_alternatives){
  softmax_inv_temp_values[t] = softmax_inv_temp_preveasy * (1-softmax_inv_temp_fractions[t])
}

softmax <- function(xvals,mu){
  yvals = 1/(1 + exp(-mu * xvals))
  return(yvals)
}

prisky_preveasy = softmax(xvals = subj_val_diff, mu = softmax_inv_temp_preveasy)

prisky_alternatives = array(dim = c(n_alternatives,length(subj_val_diff)))
for(t in 1:n_alternatives){
  prisky_alternatives[t,] = softmax(xvals = subj_val_diff, mu = softmax_inv_temp_values[t])
}

plot(subj_val_diff, prisky_preveasy, col = 'blue', type = 'l', lwd = 2, 
     xlab = 'Subjective Value Difference (risky - safe)',
     ylab = 'p(choose risky)')
for(t in 1:n_alternatives){
  lines(subj_val_diff, prisky_alternatives[t,], col = rgb(t/n_alternatives,0,0), lwd = 2)
}

prisky_differences_by_mu = array(dim = dim(prisky_alternatives))
for(t in 1:n_alternatives){
  prisky_differences_by_mu[t,] = abs(prisky_preveasy - prisky_alternatives[t,])
}

prisky_differences_by_mu_norm = array(dim = dim(prisky_alternatives))
for(t in 1:n_alternatives){
  prisky_differences_by_mu_norm[t,] = prisky_differences_by_mu[t,]/max(prisky_differences_by_mu[t,])
}

plot(prisky_preveasy, prisky_differences_by_mu[1,],
     type = 'l', col = rgb(1/n_alternatives,0,0), lwd = 2, ylim = c(0,.31),
     main = 'NOT normalized', xlab = 'p(risky) at baseline', ylab = 'CHANGE in p(risky)')
for(t in 2:n_alternatives){
  lines(prisky_preveasy, prisky_differences_by_mu[t,], col = rgb(t/n_alternatives,0,0), lwd = 2)
}

plot(prisky_preveasy, prisky_differences_by_mu_norm[1,],
     type = 'l', col = rgb(1/n_alternatives,0,0), lwd = 2, ylim = c(0,1), main = 'NORMALIZED')
for(t in 2:n_alternatives){
  lines(prisky_preveasy, prisky_differences_by_mu_norm[t,], col = rgb(t/n_alternatives,0,0), lwd = 2)
}




tmp_var = (prisky_differences_by_mu_norm[1,] > .75)*.75
tmp_var[tmp_var == 0] = NA

lines(prisky_preveasy, tmp_var, lwd = 8, col = 'green')

plot(prisky_preveasy, colSums(prisky_differences_by_mu_norm), type = 'l')

prisky_differences_weighted = array(dim = dim(prisky_differences_by_mu_norm))
for (t in 1:n_alternatives){
  prisky_differences_weighted[t,] = prisky_differences_by_mu_norm[t,] * (1/softmax_inv_temp_fractions[t])
}

summed_weighted_behavior_change = colSums(prisky_differences_weighted)/max(colSums(prisky_differences_weighted))
plot(prisky_preveasy, summed_weighted_behavior_change, type = 'l')
tmp_var = (summed_weighted_behavior_change > .9) * .9
tmp_var[tmp_var == 0] = NA
lines(prisky_preveasy, tmp_var, lwd = 8, col = 'green')



plot(NA, NA, xlab = 'prisky', ylab = 'frequency', ylim = c(0,10), xlim = c(0,1), xaxs = "i");
abline(v = 0.05)
abline(v = 0.07)
abline(v = 0.23)
abline(v = 0.45)
abline(v = 0.55)
abline(v = 0.77)
abline(v = 0.92)
abline(v = 0.95)


