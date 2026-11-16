# QVF SCRIPT FOR VISUALIZING THE SOFTMAX AND CHANGES IN IT

# Setup ----
rm(list = ls()); # clear the workspace
setwd('/Users/sokolhessner/Documents/gitrepos/qvf/R/');
# setwd('/Users/shlab/Documents/GitHub/qvf/qvf_StudyB')


# Baseline Setup ##############################################
# This is about setting up where we expect most folks to be
softmax_inv_temp_preveasy = 26.8 # the baseline mu value from CGE = 26.8; the baseline mu value from CGT = 25.0
# softmax_inv_temp_preveasy = 5.1 # CGE min
# softmax_inv_temp_preveasy = 80 # CGE max
# Turns out this process below is insensitive to baseline softmax temp

# Set up the baseline prisky values
# base_p_vals = c(0.0000001, 0.00001, seq(from = 0.0001, to = 0.9999, by = 0.0001), 0.99999, 0.9999999) # you can do this at super-high resolution, but it doesn't change anything, and ultimately doesn't facilitate later binning when creating choice sets
base_p_vals = c(.Machine$double.eps, seq(from = 0.01, to = 0.99, by = 0.01), 1-.Machine$double.eps)
# 0 and 1 give -Inf and Inf values, so can't use those!

# Calculate the implied value differences, given the CGE-observed softmax value
val_diffs = log((1/base_p_vals-1))/-softmax_inv_temp_preveasy

# Make the softmax function
softmax <- function(xvals,mu){
  yvals = 1/(1 + exp(-mu * xvals))
  return(yvals)
}


# Alternative Softmaxes ##############################################
# Make the new (flatter) softmaxes
softmax_inv_temp_fractions = c(.1, .2, .3, .4, .5, .6, .7, .8) # percent decrease in mu
n_alternatives = length(softmax_inv_temp_fractions)
softmax_inv_temp_values = array(dim = length(softmax_inv_temp_fractions))

for(t in 1:n_alternatives){
  softmax_inv_temp_values[t] = softmax_inv_temp_preveasy * (1-softmax_inv_temp_fractions[t])
}

# Calculate the new prisky values given different softmax temperatures
new_p_vals = array(dim = c(length(softmax_inv_temp_values), length(val_diffs)))
for(t in 1:n_alternatives){
  new_p_vals[t,] = softmax(xvals = val_diffs, mu = softmax_inv_temp_values[t])
}

# Visualize these new softmaxes
plot(val_diffs, base_p_vals, col = 'blue', type = 'l', lwd = 2, 
     main = '1. Softmaxes', xlab = 'Value difference', ylab = 'p(risky)')
for(t in 1:n_alternatives){
  lines(val_diffs, new_p_vals[t,], col = rgb(t/n_alternatives,0,0), lwd = 2)
}

# Differences Between Softmaxes ##############################################
# Look at how the new softmaxes differ from our predicted baseline

# Calculate the differences between the red lines and the blue line
p_diffs = array(dim = c(length(softmax_inv_temp_values), length(val_diffs)))
for(t in 1:n_alternatives){
  p_diffs[t,] = abs(base_p_vals - new_p_vals[t,])
}

# ... and plot them in change-in-p(risky) space
plot(base_p_vals, p_diffs[1,], col = rgb(1/n_alternatives,0,0), type = 'l', lwd = 2, ylim = c(0, .31),
     main = '2. Diff. in p(risky) (baseline vs new softmax value)', xlab = 'Original p(risky)', ylab = 'Change in p(risky)')
for(t in 2:n_alternatives){
  lines(base_p_vals, p_diffs[t,], col = rgb(t/n_alternatives,0,0), lwd = 2)
}

# Normalize the change in p(risky) values to better illustrate where the changes are occurring
p_diffs_norm = p_diffs
for(t in 1:n_alternatives){
  p_diffs_norm[t,] = p_diffs[t,]/max(p_diffs[t,])
}

# Plot the normalized difference values
plot(base_p_vals, p_diffs_norm[1,], col = rgb(1/n_alternatives,0,0), type = 'l', lwd = 2, ylim = c(0, 1),
     main = '3. NORMALIZED Diff. in p(risky) (baseline vs new softmax value)', xlab = 'Original p(risky)', ylab = 'Change in p(risky)')
for(t in 2:n_alternatives){
  lines(base_p_vals, p_diffs_norm[t,], col = rgb(t/n_alternatives,0,0), lwd = 2)
}

# Moving Toward Definitions of Intermediate ##############################################
# Want to figure out where to select trials - so weight the different p(risky) differences
# by what we (roughly) think to be their likelihood - small differences more likely! 
# Then sum up the various difference-in-p(risky) curves to help us see where the 
# biggest differences in behavior will be. Focus intermediate trials there!

# Now calculate the scaled sum of where we think this is most going to go:
weighted_p_diffs = p_diffs_norm
for(t in 1:n_alternatives){
  weighted_p_diffs[t,] = p_diffs_norm[t,] * (1/softmax_inv_temp_fractions[t])
}
weighted_p_diffs_sum = colSums(weighted_p_diffs)/max(colSums(weighted_p_diffs)) # normalized to 1

plot(softmax_inv_temp_fractions, 1/softmax_inv_temp_fractions, pch = 16, col = rgb(1:n_alternatives/n_alternatives,0,0), cex = 2,
     xlab = 'Fractional decrease from baseline softmax inv. temp',
     ylab = 'Weight', main = '4.1. Weights related to rough expectations of behavioral effect sizes')

plot(base_p_vals, weighted_p_diffs_sum, col = 'magenta', type = 'l', xlab = 'p(risky)',
     ylab = 'likelihood of observing a behavioral difference', main = '4.2. Summed Weighted Behavioral Difference Expectations')

threshold = 0.9 # between 0 and 1. Smaller values = narrower range of "intermediate" trials.
# Arbitrary! Need to pick a value from 0-1 corresponding to which points
# from the above plot to retain. Will always roughly emphasize the zone of greatest difference
# in behavior between the baseline softmax & a lower softmax, with an emphasis on
# the smaller softmax inv. temp differences over larger effect sizes. 

# which(weighted_p_diffs_sum > threshold)
# base_p_vals[which(weighted_p_diffs_sum > threshold)]

easy_rej = c(0.00, 0.02)
easy_acc = c(0.98, 1.00)
inte_rej = c(min(base_p_vals[(weighted_p_diffs_sum > threshold) & (base_p_vals < .5)]), 
             max(base_p_vals[(weighted_p_diffs_sum > threshold) & (base_p_vals < .5)]))
inte_acc = c(min(base_p_vals[(weighted_p_diffs_sum > threshold) & (base_p_vals > .5)]),
             max(base_p_vals[(weighted_p_diffs_sum > threshold) & (base_p_vals > .5)]))
diff_bnd = c(0.45, 0.55)

lines(x = inte_rej, y = c(threshold,threshold), col = 'green', lwd = 10)
lines(x = inte_acc, y = c(threshold,threshold), col = 'green', lwd = 10)


plot(NA, NA, xlab = 'p(risky)', ylab = '', ylim = c(0,1), xlim = c(0,1), xaxs = "i", yaxt = "n",
     main = sprintf('5. Trial Types: Easy (blue), Intermediate (pink), & Difficult (red) from baseline mu = %.1f', softmax_inv_temp_preveasy))
polygon(x = c(easy_rej[1], easy_rej[1], easy_rej[2], easy_rej[2]), y = c(-1, 100, 100, -1), col = rgb(0,0,1))
polygon(x = c(easy_acc[1], easy_acc[1], easy_acc[2], easy_acc[2]), y = c(-1, 100, 100, -1), col = rgb(0,0,1))

polygon(x = c(inte_rej[1], inte_rej[1], inte_rej[2], inte_rej[2]), y = c(-1, 100, 100, -1), col = rgb(1,0,1))
polygon(x = c(inte_acc[1], inte_acc[1], inte_acc[2], inte_acc[2]), y = c(-1, 100, 100, -1), col = rgb(1,0,1))

polygon(x = c(diff_bnd[1], diff_bnd[1], diff_bnd[2], diff_bnd[2]), y = c(-1, 100, 100, -1), col = rgb(1,0,0))

cat(sprintf('Bounds for: Intermediate Reject = [%.2f, %.2f]; Intermediate Accept = [%.2f, %.2f]\n',
            inte_rej[1], inte_rej[2], inte_acc[1], inte_acc[2]))



# Plots for Presenting Softmax-related Design Elements & Hypotheses ####

## Plot A: Predictions of Disengagement vs. Adaptation/Facilitation ####
x_vals = seq(from = -5, to = 5, by = .01)
baseline_temp = 1.5
steeper_temp = 4
flatter_temp = .7

y_baseline = softmax(xvals = x_vals, mu = baseline_temp)
y_steeper = softmax(xvals = x_vals, mu = steeper_temp)
y_flatter = softmax(xvals = x_vals, mu = flatter_temp)

plot(x_vals, y_baseline, lwd = 10, col = 'black', type = 'l',
     xaxt = 'n', xlab = 'Value Difference Between Options', ylab = 'p(Risky)', main = 'Adaptation')
lines(x_vals, y_steeper, lwd = 5, lty = 'dashed', col = 'darkgrey')

plot(x_vals, y_baseline, lwd = 10, col = 'black', type = 'l',
     xaxt = 'n', xlab = 'Value Difference Between Options', ylab = 'p(Risky)', main = 'Disengagement')
lines(x_vals, y_flatter, lwd = 5, lty = 'dashed', col = 'darkgrey')


## Plot B: Identifying Easy/Inte/Diff & highlighting zones of maximum difference ####

x_vals = seq(from = -5, to = 5, by = .01)
baseline_temp = 1.5
steeper_temp = 4
flatter_temp = .9

y_baseline = softmax(xvals = x_vals, mu = baseline_temp)
y_steeper = softmax(xvals = x_vals, mu = steeper_temp)
y_flatter = softmax(xvals = x_vals, mu = flatter_temp)


baseline_p_vals_easy_acc = seq(from = 0.001, to = 0.02, by = 0.001)
baseline_p_vals_inte_acc = seq(from = 0.08, to = 0.22, by = 0.022)
baseline_p_vals_diff = seq(from = 0.45, to = 0.55, by = 0.001)
baseline_p_vals_inte_rej = seq(from = 0.78, to = 0.92, by = 0.022)
baseline_p_vals_easy_rej = seq(from = 0.98, to = 0.999, by = 0.001)

baseline_x_vals_easy_acc = log((1/baseline_p_vals_easy_acc-1))/-baseline_temp
baseline_x_vals_inte_acc = log((1/baseline_p_vals_inte_acc-1))/-baseline_temp
baseline_x_vals_diff = log((1/baseline_p_vals_diff-1))/-baseline_temp
baseline_x_vals_inte_rej = log((1/baseline_p_vals_inte_rej-1))/-baseline_temp
baseline_x_vals_easy_rej = log((1/baseline_p_vals_easy_rej-1))/-baseline_temp

steeper_p_vals_easy_acc = softmax(baseline_x_vals_easy_acc, steeper_temp)
steeper_p_vals_inte_acc = softmax(baseline_x_vals_inte_acc, steeper_temp)
steeper_p_vals_diff = softmax(baseline_x_vals_diff, steeper_temp)
steeper_p_vals_inte_rej = softmax(baseline_x_vals_inte_rej, steeper_temp)
steeper_p_vals_easy_rej = softmax(baseline_x_vals_easy_rej, steeper_temp)

flatter_p_vals_easy_acc = softmax(baseline_x_vals_easy_acc, flatter_temp)
flatter_p_vals_inte_acc = softmax(baseline_x_vals_inte_acc, flatter_temp)
flatter_p_vals_diff = softmax(baseline_x_vals_diff, flatter_temp)
flatter_p_vals_inte_rej = softmax(baseline_x_vals_inte_rej, flatter_temp)
flatter_p_vals_easy_rej = softmax(baseline_x_vals_easy_rej, flatter_temp)


plot(NULL, xlim = c(min(x_vals), max(x_vals)), ylim = c(0,1),
     xaxt = 'n', xlab = 'Value Difference Between Options', ylab = 'p(Risky)')
lines(x_vals, y_baseline, lwd = 6, col = 'black')
lines(x_vals, y_steeper, lwd = 2, col = 'darkgrey')
lines(x_vals, y_flatter, lwd = 2, col = 'darkgrey')

polygon(x = c(baseline_x_vals_easy_acc,rev(baseline_x_vals_easy_acc)),
        y = c(flatter_p_vals_easy_acc,rev(steeper_p_vals_easy_acc)),
        col = rgb(0,0,1,.8), border = NA)

polygon(x = c(baseline_x_vals_inte_acc,rev(baseline_x_vals_inte_acc)),
        y = c(flatter_p_vals_inte_acc,rev(steeper_p_vals_inte_acc)),
        col = rgb(.5,0,1,.8), border = NA)

polygon(x = c(baseline_x_vals_diff,rev(baseline_x_vals_diff)),
        y = c(flatter_p_vals_diff,rev(steeper_p_vals_diff)),
        col = rgb(1,0,0,.8), border = NA)

polygon(x = c(baseline_x_vals_inte_rej,rev(baseline_x_vals_inte_rej)),
        y = c(flatter_p_vals_inte_rej,rev(steeper_p_vals_inte_rej)),
        col = rgb(.5,0,1,.8), border = NA)

polygon(x = c(baseline_x_vals_easy_rej,rev(baseline_x_vals_easy_rej)),
        y = c(flatter_p_vals_easy_rej,rev(steeper_p_vals_easy_rej)),
        col = rgb(0,0,1,.8), border = NA)

## Plot C: Expected Magnitude of Behavioral Change Graph ####

# Making a version of an earlier graph that highlights the different trial types better. 
plot(base_p_vals, weighted_p_diffs_sum, 
     col = 'black', type = 'l', lwd = 1,
     xlab = 'Baseline p(risky)',
     ylab = 'Expected Behavioral Difference', 
     main = 'Expected Magnitude of Behavioral Change',
     xaxs = "i",yaxs = "i", bty = 'n')

p_vals_easyacc = base_p_vals < .02
p_vals_inteacc = (base_p_vals > .08) & (base_p_vals < .22)
p_vals_diff = (base_p_vals > .45) & (base_p_vals < .55)
p_vals_interej = (base_p_vals > .78) & (base_p_vals < .92)
p_vals_easyrej = base_p_vals > .98

polygon(x = c(base_p_vals[p_vals_easyacc], rev(base_p_vals[p_vals_easyacc])),
        y = c(rep(0,sum(p_vals_easyacc)), rev(weighted_p_diffs_sum[p_vals_easyacc])),
        col = rgb(0,0,1), border = NA)

polygon(x = c(base_p_vals[p_vals_inteacc], rev(base_p_vals[p_vals_inteacc])),
        y = c(rep(0,sum(p_vals_inteacc)), rev(weighted_p_diffs_sum[p_vals_inteacc])),
        col = rgb(.5,0,1), border = NA)

polygon(x = c(base_p_vals[p_vals_diff], rev(base_p_vals[p_vals_diff])),
        y = c(rep(0,sum(p_vals_diff)), rev(weighted_p_diffs_sum[p_vals_diff])),
        col = rgb(1,0,0), border = NA)

polygon(x = c(base_p_vals[p_vals_interej], rev(base_p_vals[p_vals_interej])),
        y = c(rep(0,sum(p_vals_interej)), rev(weighted_p_diffs_sum[p_vals_interej])),
        col = rgb(.5,0,1), border = NA)

polygon(x = c(base_p_vals[p_vals_easyrej], rev(base_p_vals[p_vals_easyrej])),
        y = c(rep(0,sum(p_vals_easyrej)), rev(weighted_p_diffs_sum[p_vals_easyrej])),
        col = rgb(0,0,1), border = NA)
