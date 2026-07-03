# QVF SCRIPT FOR VISUALIZING THE SOFTMAX AND CHANGES IN IT

# Setup ----
rm(list = ls()); # clear the workspace
setwd('/Users/sokolhessner/Documents/gitrepos/qvf/R/');


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
     main = 'Softmaxes', xlab = 'Value difference', ylab = 'p(risky)')
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
     main = 'Diff. in p(risky) (baseline vs new softmax value)', xlab = 'Original p(risky)', ylab = 'Change in p(risky)')
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
     main = 'NORMALIZED Diff. in p(risky) (baseline vs new softmax value)', xlab = 'Original p(risky)', ylab = 'Change in p(risky)')
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

plot(base_p_vals, weighted_p_diffs_sum, col = 'magenta', type = 'l')

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


plot(NA, NA, xlab = 'prisky', ylab = '', ylim = c(0,1), xlim = c(0,1), xaxs = "i", yaxt = "n",
     main = sprintf('Trial Types: Easy (blue), Intermediate (pink), & Difficult (red) from baseline mu = %.1f', softmax_inv_temp_preveasy),
     xlab = 'p(risky)')
polygon(x = c(easy_rej[1], easy_rej[1], easy_rej[2], easy_rej[2]), y = c(-1, 100, 100, -1), col = rgb(0,0,1))
polygon(x = c(easy_acc[1], easy_acc[1], easy_acc[2], easy_acc[2]), y = c(-1, 100, 100, -1), col = rgb(0,0,1))

polygon(x = c(inte_rej[1], inte_rej[1], inte_rej[2], inte_rej[2]), y = c(-1, 100, 100, -1), col = rgb(1,0,1))
polygon(x = c(inte_acc[1], inte_acc[1], inte_acc[2], inte_acc[2]), y = c(-1, 100, 100, -1), col = rgb(1,0,1))

polygon(x = c(diff_bnd[1], diff_bnd[1], diff_bnd[2], diff_bnd[2]), y = c(-1, 100, 100, -1), col = rgb(1,0,0))

cat(sprintf('Intermediate Reject = [%.2f, %.2f]; Intermediate Accept = [%.2f, %.2f]\n',
            inte_rej[1], inte_rej[2], inte_acc[1], inte_acc[2]))

