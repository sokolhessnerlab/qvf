# QVF SCRIPT FOR FITTING PARTICIPANTS' CHOICES AND CREATING NOVEL CHOICE SETS

# Setup ----
rm(list = ls()); # clear the workspace
setwd('/Users/sokolhessner/Documents/gitrepos/qvf/R/');

library(tictoc)
library(doParallel)
library(doRNG)

# Create function to calculate choice probabilities ----
choice_probability <- function(parameters, choiceset) {
  # A function to calculate the probability of taking a risky option
  # using a prospect theory model.
  # Assumes parameters are [rho, mu] as used in S-H 2009, 2013, 2015, etc.
  # Assumes choiceset has columns riskyoption1, riskyoption2, and safeoption
  #
  # PSH & AR June 2022
  
  # extract  parameters
  rho = as.double(parameters[1]); # risk attitudes
  mu = as.double(parameters[2]); # choice consistency
  
  # Correct parameter bounds
  if(rho <= 0){
    rho = .Machine$double.eps;
  }
  
  if(mu < 0){
    mu = 0;
  }
  
  # calculate utility of the two options
  utility_risky_option = 0.5 * choiceset$riskyoption1^rho + 
    0.5 * choiceset$riskyoption2^rho;
  utility_safe_option = choiceset$safeoption^rho;
  
  # normalize values using this term
  div <- 30^rho; # decorrelates rho & mu
  
  # calculate the probability of selecting the risky option
  p = 1/(1+exp(-mu/div*(utility_risky_option - utility_safe_option)));
  
  return(p)
}


# Configuring choice set creation ----

## Defining how many choice sets to make ----
n_rho_values = 200; # SET THIS TO THE DESIRED DEGREE OF FINENESS
n_mu_values = 201; # IBID

cat(sprintf('You have decided to make %i choice sets!\n\n',n_rho_values*n_mu_values))

rho_values = seq(from = 0.35, to = 2.2, length.out = n_rho_values); # the range of fit-able values
mu_values = seq(from = 11, to = 80, length.out = n_mu_values); # the range of fit-able values

## Defining Choice Set contents ----
# Set up variables defining choice set creation
total_number_difficult = 80; # total number of choices in each type
total_number_intermediate = 84; # total number of choices in each type
total_number_easy = 80;

number_dynamic_blocks = 2

number_difficult_perDynBlk = total_number_difficult/number_dynamic_blocks;
number_intermediate_perDynBlk = total_number_intermediate/number_dynamic_blocks;
number_easy_perDynBlk = total_number_easy/number_dynamic_blocks;

# Probability ranges for easy & difficult categories
choiceP_range_difficult = c(0.45, 0.55);
choiceP_range_int_lower = c(0.08, 0.22);
choiceP_range_int_upper = c(0.78, 0.92);
choiceP_range_easy_lower = c(0, 0.02);
choiceP_range_easy_upper = c(0.98, 1);

# Bin Edges
bin_edges_difficult = seq(from = choiceP_range_difficult[1], to = choiceP_range_difficult[2], by = 0.01)
bin_edges_easy_lower = seq(from = choiceP_range_easy_lower[1], to = choiceP_range_easy_lower[2], by = 0.01)
bin_edges_easy_upper = seq(from = choiceP_range_easy_upper[1], to = choiceP_range_easy_upper[2], by = 0.01)
bin_edges_int_lower = seq(from = choiceP_range_int_lower[1], to = choiceP_range_int_lower[2], by = 0.02) # Need bins of width 2% to make the trial numbers work out evenly
bin_edges_int_upper = seq(from = choiceP_range_int_upper[1], to = choiceP_range_int_upper[2], by = 0.02)

# Number of Bins
nbins_difficult = length(bin_edges_difficult) - 1
nbins_int_lower = length(bin_edges_int_lower) - 1
nbins_int_upper = length(bin_edges_int_upper) - 1
nbins_easy_lower = length(bin_edges_easy_lower) - 1
nbins_easy_upper = length(bin_edges_easy_upper) - 1

# Number of trials/bin/dynamic block
num_difficult_perBin_perDynblk = total_number_difficult/(nbins_difficult * 2) # 2 = num of dynamic blocks
num_int_lower_perBin_perDynblk = total_number_intermediate/(nbins_int_lower * 2 * 2) # 2 = num of dynamic blocks; 2 = upper/lower
num_int_upper_perBin_perDynblk = total_number_intermediate/(nbins_int_upper * 2 * 2) # 2 = num of dynamic blocks; 2 = upper/lower
num_easy_lower_perBin_perDynblk = total_number_easy/(nbins_easy_lower * 2 * 2) # 2 = num of dynamic blocks; 2 = upper/lower
num_easy_upper_perBin_perDynblk = total_number_easy/(nbins_easy_upper * 2 * 2) # 2 = num of dynamic blocks; 2 = upper/lower

# allowable $ values
possible_risky_value_range = c(0.01, 30); 
possible_safe_value_range = c(0.01, 21);

all_possible_safe_values = seq(from = possible_safe_value_range[1], 
                               to = possible_safe_value_range[2], by = 0.01)
all_possible_risky_values = seq(from = possible_risky_value_range[1], 
                                to = possible_risky_value_range[2], by = 0.01)

nvals_safe = length(all_possible_safe_values)
nvals_risky = length(all_possible_risky_values)
npairs = nvals_risky * nvals_safe

full_possible_choiceset = array(dim = c(npairs, 4))
full_possible_choiceset[,1] = rep(all_possible_risky_values, times = nvals_safe) # riskyoption1
full_possible_choiceset[,2] = 0 # riskyoption2
full_possible_choiceset[,3] = rep(all_possible_safe_values, each = nvals_risky) # safe

colnames(full_possible_choiceset) <- c('riskyoption1', 
                                       'riskyoption2', 
                                       'safeoption', 
                                       'choiceP');
full_possible_choiceset = as.data.frame(full_possible_choiceset)

colnames_out = c('riskyoption1', 'riskyoption2', 'safeoption', 
                 'choiceP', 'type_e0i1d2', 'reject0accept1', 'dynamicblocknum');
ncols_out = length(colnames_out)

setwd('/Users/sokolhessner/Documents/gitrepos/qvf/R/bespoke_choicesets/');

# Set up the parallelization
n.cores <- parallel::detectCores() - 1; # Use 1 less than the full number of cores.

my.cluster <- parallel::makeCluster(
  n.cores,
  type = "FORK"
)
doParallel::registerDoParallel(cl = my.cluster)

# Loop through and create choice sets ----

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
#                                          #
#               May take 9 hrs?            #
#                                          #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

tic();
# for(r in 1:n_rho_values){ # for sequential
#   for(m in 1:n_mu_values){ 

# for(r in seq(from = 1, by = 9, to = 200)){ # for testing sequential
#   for(m in seq(from = 1, by = 9, to = 200)){ 

# foreach(r=seq(from = 1, by = 9, to = 200)) %dorng% { # for testing parallelization
#   for(m in seq(from = 1, by = 9, to = 200)){ 

foreach(r=1:n_rho_values) %dorng% { # for parallelization
  for(m in 1:n_mu_values){ 
    
    ## Carry out the subject loop ----
    temp_parameters = c(rho_values[r],mu_values[m]);
    cat(sprintf('\u03C1 = %.2f   \u03BC = %.2f', temp_parameters[1], temp_parameters[2]))
    
    # Calculate all possible choice probabilities for all possible choices
    full_possible_choiceset$choiceP = choice_probability(temp_parameters,full_possible_choiceset) 
    
    new_choiceset = array(dim = c(0,length(colnames_out)))
    colnames(new_choiceset) <- colnames_out
    new_choiceset = as.data.frame(new_choiceset); 
    
    ### Cycle through DYNAMIC BLOCKS ----
    for (numDynBlk in 1:number_dynamic_blocks){
      
      #### Make DIFFICULT choices ----
      for (binN in 1:nbins_difficult){
        # Set up bin edges (in choice probability space)
        tmp_lower_bin_edge = bin_edges_difficult[binN] # define lower bin edge
        tmp_upper_bin_edge = bin_edges_difficult[binN + 1] # define upper bin edge
        
        ind_meet_criteria = which((full_possible_choiceset$choiceP > tmp_lower_bin_edge) & 
                                    (full_possible_choiceset$choiceP <= tmp_upper_bin_edge))
        ind_meet_criteria_selected = sample(ind_meet_criteria, 
                                            size = num_difficult_perBin_perDynblk)
        
        tmp_choiceset = array(dim = c(num_difficult_perBin_perDynblk,
                                      length(colnames_out)))
        colnames(tmp_choiceset) <- colnames_out
        tmp_choiceset = as.data.frame(tmp_choiceset); 
        
        tmp_choiceset$riskyoption1 = full_possible_choiceset$riskyoption1[ind_meet_criteria_selected]
        tmp_choiceset$riskyoption2 = full_possible_choiceset$riskyoption2[ind_meet_criteria_selected]
        tmp_choiceset$safeoption = full_possible_choiceset$safeoption[ind_meet_criteria_selected]
        tmp_choiceset$choiceP = full_possible_choiceset$choiceP[ind_meet_criteria_selected]
        tmp_choiceset$type_e0i1d2 = 2; # 2 = difficult
        tmp_choiceset$reject0accept1 = 2; # 2 = neither accept nor reject (it's difficult)
        tmp_choiceset$dynamicblocknum = numDynBlk; # number of the dynamic block
        
        new_choiceset = rbind(new_choiceset,tmp_choiceset)
      } # end of bin FOR
      
      #### Make INTERMEDIATE choices ----
      ##### INT. LOWER choices (i.e. reject) ----
      for (binN in 1:nbins_int_lower){
        tmp_lower_bin_edge = bin_edges_int_lower[binN]
        tmp_upper_bin_edge = bin_edges_int_lower[binN + 1]
        
        ind_meet_criteria = which((full_possible_choiceset$choiceP > tmp_lower_bin_edge) & 
                                    (full_possible_choiceset$choiceP <= tmp_upper_bin_edge))
        ind_meet_criteria_selected = sample(ind_meet_criteria, 
                                            size = num_int_lower_perBin_perDynblk)
        
        tmp_choiceset = array(dim = c(num_int_lower_perBin_perDynblk,
                                      length(colnames_out)))
        colnames(tmp_choiceset) <- colnames_out
        tmp_choiceset = as.data.frame(tmp_choiceset); 
        
        tmp_choiceset$riskyoption1 = full_possible_choiceset$riskyoption1[ind_meet_criteria_selected]
        tmp_choiceset$riskyoption2 = full_possible_choiceset$riskyoption2[ind_meet_criteria_selected]
        tmp_choiceset$safeoption = full_possible_choiceset$safeoption[ind_meet_criteria_selected]
        tmp_choiceset$choiceP = full_possible_choiceset$choiceP[ind_meet_criteria_selected]
        tmp_choiceset$type_e0i1d2 = 1; # 1 = intermediate
        tmp_choiceset$reject0accept1 = 0; # reject
        tmp_choiceset$dynamicblocknum = numDynBlk; # number of the dynamic block
        
        new_choiceset = rbind(new_choiceset,tmp_choiceset)
      } # end of bin FOR
      
      ##### INT. UPPER choices (i.e. accept) ----
      for (binN in 1:nbins_int_upper){
        tmp_lower_bin_edge = bin_edges_int_upper[binN]
        tmp_upper_bin_edge = bin_edges_int_upper[binN + 1]
        
        ind_meet_criteria = which((full_possible_choiceset$choiceP > tmp_lower_bin_edge) & 
                                    (full_possible_choiceset$choiceP <= tmp_upper_bin_edge))
        ind_meet_criteria_selected = sample(ind_meet_criteria, 
                                            size = num_int_upper_perBin_perDynblk)
        
        tmp_choiceset = array(dim = c(num_int_upper_perBin_perDynblk,
                                      length(colnames_out)))
        colnames(tmp_choiceset) <- colnames_out
        tmp_choiceset = as.data.frame(tmp_choiceset); 
        
        tmp_choiceset$riskyoption1 = full_possible_choiceset$riskyoption1[ind_meet_criteria_selected]
        tmp_choiceset$riskyoption2 = full_possible_choiceset$riskyoption2[ind_meet_criteria_selected]
        tmp_choiceset$safeoption = full_possible_choiceset$safeoption[ind_meet_criteria_selected]
        tmp_choiceset$choiceP = full_possible_choiceset$choiceP[ind_meet_criteria_selected]
        tmp_choiceset$type_e0i1d2 = 1; # 1 = intermediate
        tmp_choiceset$reject0accept1 = 1; # accept
        tmp_choiceset$dynamicblocknum = numDynBlk; # number of the dynamic block
        
        new_choiceset = rbind(new_choiceset,tmp_choiceset)        
      } # end of bin FOR
      
      #### Make EASY choices ----
      ##### Easy LOWER (i.e. reject) ----
      for (binN in 1:nbins_easy_lower){
        tmp_lower_bin_edge = bin_edges_easy_lower[binN]
        tmp_upper_bin_edge = bin_edges_easy_lower[binN + 1]
        
        ind_meet_criteria = which((full_possible_choiceset$choiceP > tmp_lower_bin_edge) & 
                                    (full_possible_choiceset$choiceP <= tmp_upper_bin_edge))
        ind_meet_criteria_selected = sample(ind_meet_criteria, 
                                            size = num_easy_lower_perBin_perDynblk)
        
        tmp_choiceset = array(dim = c(num_easy_lower_perBin_perDynblk,
                                      length(colnames_out)))
        colnames(tmp_choiceset) <- colnames_out
        tmp_choiceset = as.data.frame(tmp_choiceset); 
        
        tmp_choiceset$riskyoption1 = full_possible_choiceset$riskyoption1[ind_meet_criteria_selected]
        tmp_choiceset$riskyoption2 = full_possible_choiceset$riskyoption2[ind_meet_criteria_selected]
        tmp_choiceset$safeoption = full_possible_choiceset$safeoption[ind_meet_criteria_selected]
        tmp_choiceset$choiceP = full_possible_choiceset$choiceP[ind_meet_criteria_selected]
        tmp_choiceset$type_e0i1d2 = 0; # 0 = easy
        tmp_choiceset$reject0accept1 = 0; # reject
        tmp_choiceset$dynamicblocknum = numDynBlk; # number of the dynamic block
        
        new_choiceset = rbind(new_choiceset,tmp_choiceset)
      } # end of bin FOR
      
      ##### Easy UPPER (i.e. accept) ----
      for (binN in 1:nbins_easy_upper){
        tmp_lower_bin_edge = bin_edges_easy_upper[binN]
        tmp_upper_bin_edge = bin_edges_easy_upper[binN + 1]
        
        ind_meet_criteria = which((full_possible_choiceset$choiceP > tmp_lower_bin_edge) & 
                                    (full_possible_choiceset$choiceP <= tmp_upper_bin_edge))
        ind_meet_criteria_selected = sample(ind_meet_criteria, 
                                            size = num_easy_upper_perBin_perDynblk)
        
        tmp_choiceset = array(dim = c(num_easy_upper_perBin_perDynblk,
                                      length(colnames_out)))
        colnames(tmp_choiceset) <- colnames_out
        tmp_choiceset = as.data.frame(tmp_choiceset); 
        
        tmp_choiceset$riskyoption1 = full_possible_choiceset$riskyoption1[ind_meet_criteria_selected]
        tmp_choiceset$riskyoption2 = full_possible_choiceset$riskyoption2[ind_meet_criteria_selected]
        tmp_choiceset$safeoption = full_possible_choiceset$safeoption[ind_meet_criteria_selected]
        tmp_choiceset$choiceP = full_possible_choiceset$choiceP[ind_meet_criteria_selected]
        tmp_choiceset$type_e0i1d2 = 0; # 0 = easy
        tmp_choiceset$reject0accept1 = 1; # accept
        tmp_choiceset$dynamicblocknum = numDynBlk; # number of the dynamic block
        
        new_choiceset = rbind(new_choiceset,tmp_choiceset)
      } # end of bin FOR
    } # end of Dynamic Block FOR
    cat('; done.\n')
    
    ## Save out the new choice set ----
    colnames(new_choiceset) <- colnames_out
    new_choiceset = new_choiceset[sample(nrow(new_choiceset)),]; # Randomly sort the choiceset
    new_choiceset = as.data.frame(new_choiceset); # make it a dataframe for saving
    
    fname = sprintf('qvf_bespoke_choiceset_rhoInd%03i_muInd%03i.csv', r, m); # Use of %03i creates a three-digit text string with leading 0's as needed for the relevant index; this standardizes file name length
    # Files are ~14 KB in size. 40,200 such files should be ~560MB (half a gig).
    
    write.csv(new_choiceset, file = fname, row.names = F);
  } # End of mu FOR
  
  cat(sprintf('Finished \u03C1 %i/%i.\n',r, n_rho_values)) # only works with parallel implementation
} # End of rho FOR
stopCluster(my.cluster)
x = toc()

sec_elapsed = x$toc-x$tic # seconds

cat(sprintf('\n\nTook %.1f hours. Whew!\n',sec_elapsed/60/60))

# expected_hours = sec_elapsed/529*40200/60/60
# 
# cat(sprintf('\n\nExpected total time for 40,200 choice sets = %.1f hours. Plan accordingly!\n', expected_hours))

# All finished!






# APPENDIX ----

## Likelihood function ----
# negLLprospect_qvf <- function(parameters,choiceset,choices) {
#   # A negative log likelihood function for a prospect-theory estimation.
#   # Assumes parameters are [rho, mu] as used in S-H 2009, 2013, 2015, etc.
#   # Assumes choiceset has columns riskyoption1, riskyoption2, and safeoption
#   # Assumes choices are binary/logical, with 1 = risky, 0 = safe.
#   #
#   # Peter Sokol-Hessner
#   # July 2021
#   
#   choiceP = choice_probability(parameters, choiceset);
#   
#   likelihood = choices * choiceP + (1 - choices) * (1-choiceP);
#   likelihood[likelihood == 0] = 0.000000000000001; # 1e-15, i.e. 14 zeros followed by a 1
#   
#   nll <- -sum(log(likelihood));
#   return(nll)
# }



## Visualization of Choice Probability surface given different parameters ----
# 
# riskyvals = seq(from = possible_risky_value_range[1], to = possible_risky_value_range[2],
#                 length.out = 100);
# safevals = seq(from = possible_safe_value_range[1], to = possible_safe_value_range[2],
#                length.out = 101);
# 
# choiceP_matrix = array(dim = c(length(riskyvals),length(safevals)));
# 
# tempchoiceoption = array(dim = c(1,3));
# colnames(tempchoiceoption) <- c('riskyoption1','riskyoption2','safeoption');
# tempchoiceoption = as.data.frame(tempchoiceoption);
# 
# visualization_rho = 2.2; # range is 0.3 - 1.89
# visualization_mu = 7; # expected range is 0-50?
# 
# for(i in 1:length(riskyvals)){
#   for(j in 1:length(safevals)){
#     tempchoiceoption[] = c(riskyvals[i], 0, safevals[j]);
# 
#     choiceP_matrix[i,j] = choice_probability(c(visualization_rho, visualization_mu), tempchoiceoption);
#   }
# }
# 
# # Heatmap of the choiceProbability values
# pdf(file=sprintf('choice_probability_surface_rho%g_mu%g.pdf', visualization_rho, visualization_mu));
# image(riskyvals, safevals, choiceP_matrix,
#       col = hcl.colors(100, palette = "red-green", rev = F), 
#       breaks = seq(from = 0, to = 1, length.out = 101),
#       main = sprintf('Rho = %g, Mu = %g\n min(p) = %.2f, max(p) = %.2f', visualization_rho, visualization_mu, min(choiceP_matrix), max(choiceP_matrix)),
#       xlab = 'Risky values ($)', ylab = 'Safe values ($)')
# points(choiceset$riskyoption1[choiceset$ischecktrial == 0], choiceset$safeoption[choiceset$ischecktrial == 0])
# dev.off();


## Example code to simulate and fit one person's choices ----
# true_vals = c(0.8, 20); # rho (risk attitudes), mu (choice consistency)
# 
# choiceP = choice_probability(true_vals, choiceset)
# simulatedchoices = as.integer(runif(n = length(choiceP)) < choiceP);
# 
# choiceset_temp = list();
# choiceset_temp$riskyoption1 = c(5, 8, 10, 12, 18, 4, 9);
# choiceset_temp$riskyoption2 = c(0, 0,  0,  0,  0, 0, 0);
# choiceset_temp$safeoption =   c(1, 5,  3,  8, 10, 2, 4);
# simulatedchoices =            c(1, 0,  1,  1,  0, 0, 1);
# choiceset = as.data.frame(choiceset_temp);

# NOTE: may want to consider specifying mu values in log space to account for nonlinearity/skewness
#   i.e. using exp(seq(from = log(3), to = log(100), length.out = 50)) or something like it.
# NOTE: in Python, `numpy.linspace` may accomplish this identical operation.

# grid_nll_values = array(dim = c(n_rho_values, n_mu_values));
# 
# tic();
# for(r in 1:n_rho_values){
#   for(m in 1:n_mu_values){
#     grid_nll_values[r,m] = negLLprospect_qvf(c(rho_values[r],mu_values[m]), choiceset, simulatedchoices)
#   }
# }
# toc()
# 
# min_nll = min(grid_nll_values); # identify the single best value
# indexes = which(grid_nll_values == min_nll, arr.ind = T); # Get indices for that single best value
# 
# best_rho = rho_values[indexes[1]]; # what are the corresponding rho & mu values?
# best_mu = mu_values[indexes[2]];
# 
# sprintf('The best R index is %i while the best M indx is %i, with an NLL of %f', indexes[1], indexes[2], min_nll)
# 
# c(best_rho, best_mu)
# true_vals
# 
# fname = sprintf('bespoke_choiceset_rhoInd%03i_muInd%03i.csv', indexes[1], indexes[2]); # Use of %03i creates a three-digit text string with leading 0's as needed for the relevant index; this standardizes file name length


## Example Optimization code ----
# 
# negLLprospect_qvf(c(1.2, 20), choiceset, simulatedchoices)
# # It works!
# 
# eps = .Machine$double.eps;
# lower_bounds = c(eps, 0); # R, M
# upper_bounds = c(2,50); 
# number_of_parameters = length(lower_bounds);
# 
# # Create placeholders for parameters, errors, NLL (and anything else you want)
# number_of_iterations = 200; # 100 or more
# temp_parameters = array(dim = c(number_of_iterations,number_of_parameters));
# temp_hessians = array(dim = c(number_of_iterations,number_of_parameters,number_of_parameters));
# temp_NLLs = array(dim = c(number_of_iterations,1));
# 
# # tic() # start the timer
# 
# for(iter in 1:number_of_iterations){
#   # Randomly set initial values within supported values
#   # using uniformly-distributed values. Many ways to do this!
#   
#   initial_values = runif(number_of_parameters, min = lower_bounds, max = upper_bounds)
#   
#   temp_output = optim(initial_values, negLLprospect_qvf,
#                       choiceset = choiceset,
#                       choices = simulatedchoices,
#                       lower = lower_bounds,
#                       upper = upper_bounds,
#                       method = "L-BFGS-B",
#                       hessian = T)
#   
#   # Store the output we need access to later
#   temp_parameters[iter,] = temp_output$par; # parameter values
#   temp_hessians[iter,,] = temp_output$hessian; # SEs
#   temp_NLLs[iter,] = temp_output$value; # the NLLs
# }
# 
# # toc() # stop the timer; how long did it take? Use this to plan!
# 
# # How'd we do? Look at the NLLs to gauge quality of fit
# unique(temp_NLLs) # they look the same but are not...
# 
# # Compare output; select the best one
# sim_nll = min(temp_NLLs); # the best NLL for this person
# sim_best_ind = which(temp_NLLs == sim_nll)[1]; # the index of that NLL
# 
# sim_parameters = temp_parameters[sim_best_ind,] # the parameters
# sim_parameter_errors = sqrt(diag(solve(temp_hessians[sim_best_ind,,]))); # the SEs
# 
# true_vals
# sim_parameters
# sim_parameter_errors
# 

## Visualization of Estimated parameters, likelihoods, & easy/difficult lines ----
# 
# # Plot actual choices
# plot(choiceset$riskyoption1[simulatedchoices == 0], choiceset$safeoption[simulatedchoices == 0],col = 'red',
#      xlim = c(0,30), ylim = c(0,12))
# points(choiceset$riskyoption1[simulatedchoices == 1], choiceset$safeoption[simulatedchoices == 1],col = 'green')
# 
# # Plot the probability of various choices + lines that define the probability regions
# pal = colorRampPalette(c('red','white','green'))
# choiceset$pal = pal(100)[as.numeric(cut(choiceP, breaks = 100))]
# # choiceset$pal = pal(100)[0:100/100]
# plot(choiceset$riskyoption1, choiceset$safeoption, col = 'black', bg = choiceset$pal, 
#      xlim = c(0,30), ylim = c(0,12), pch = 21)
# 
# xval = 35;
# r = true_vals[1];
# m = true_vals[2];
# pval = c(choiceP_range_easy_lower[2], choiceP_range_difficult, choiceP_range_easy_upper[1])
# yval_true = array(dim = c(length(pval),1));
# 
# for (i in 1:length(pval)){
#   yval_true[i] = ((log(1/pval[i] - 1)/(m/(max(choiceset[,1:3])^r)))+0.5*(xval^r))^(1/r)
#   lines(x = c(0,xval), y = c(0,yval_true[i]))
# }
# 
# # Plot the new choice set (THIS MAY NOT WORK, GIVEN EDITS TO CHOICE SET CREATION ABOVE)
# plot(new_choiceset$riskyoption1[new_choiceset$easy0difficult1==0],
#      new_choiceset$safeoption[new_choiceset$easy0difficult1==0], col = 'blue',
#      xlim = c(0,30), ylim = c(0,12))
# points(new_choiceset$riskyoption1[new_choiceset$easy0difficult1==1],new_choiceset$safeoption[new_choiceset$easy0difficult1==1], col = 'red')
# 
# r = sim_parameters[1];
# m = sim_parameters[2];
# yval_sim = array(dim = c(length(pval),1));
# 
# for (i in 1:length(pval)){
#   yval_sim[i] = ((log(1/pval[i] - 1)/(m/(max(choiceset[,1:3])^r)))+0.5*(xval^r))^(1/r)
#   lines(x = c(0,xval), y = c(0,yval_sim[i]), lty = 'dashed')
# }
