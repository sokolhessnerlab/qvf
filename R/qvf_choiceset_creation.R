# CGT SCRIPT FOR FITTING PARTICIPANTS' CHOICES AND CREATING NOVEL CHOICE SETS

#### Setup ####
rm(list = ls()); # clear the workspace
setwd('/Users/sokolhessner/Documents/gitrepos/qvf/R/');

library(tictoc)

tic()

#### Function Creation ####

# Function to calculate choice probabilities
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
  div <- max(choiceset[,1:3])^rho; # decorrelates rho & mu
  
  # calculate the probability of selecting the risky option
  p = 1/(1+exp(-mu/div*(utility_risky_option - utility_safe_option)));
  
  return(p)
}

# Likelihood function
negLLprospect_cgt <- function(parameters,choiceset,choices) {
  # A negative log likelihood function for a prospect-theory estimation.
  # Assumes parameters are [rho, mu] as used in S-H 2009, 2013, 2015, etc.
  # Assumes choiceset has columns riskyoption1, riskyoption2, and safeoption
  # Assumes choices are binary/logical, with 1 = risky, 0 = safe.
  #
  # Peter Sokol-Hessner
  # July 2021
  
  choiceP = choice_probability(parameters, choiceset);
  
  likelihood = choices * choiceP + (1 - choices) * (1-choiceP);
  likelihood[likelihood == 0] = 0.000000000000001; # 1e-15, i.e. 14 zeros followed by a 1
  
  nll <- -sum(log(likelihood));
  return(nll)
}


# Simulate one person's choices
true_vals = c(0.8, 20); # rho (risk attitudes), mu (choice consistency)

choiceP = choice_probability(true_vals, choiceset)
simulatedchoices = as.integer(runif(n = length(choiceP)) < choiceP);

choiceset_temp = list();
choiceset_temp$riskyoption1 = c(5, 8, 10, 12, 18, 4, 9);
choiceset_temp$riskyoption2 = c(0, 0,  0,  0,  0, 0, 0);
choiceset_temp$safeoption =   c(1, 5,  3,  8, 10, 2, 4);
simulatedchoices =            c(1, 0,  1,  1,  0, 0, 1);
choiceset = as.data.frame(choiceset_temp);

#### Do Grid Search Method of identifying the best parameters ####

n_rho_values = 200; # SET THIS TO THE DESIRED DEGREE OF FINENESS
n_mu_values = 201; # IBID

print(sprintf('You have decided to make %i choice sets!',n_rho_values*n_mu_values))

rho_values = seq(from = 0.3, to = 2.2, length.out = n_rho_values); # the range of fit-able values
mu_values = seq(from = 7, to = 80, length.out = n_mu_values); # the range of fit-able values
# NOTE: may want to consider specifying mu values in log space to account for nonlinearity/skewness
#   i.e. using exp(seq(from = log(3), to = log(100), length.out = 50)) or something like it.
# NOTE: in Python, `numpy.linspace` may accomplish this identical operation.

grid_nll_values = array(dim = c(n_rho_values, n_mu_values));

tic();
for(r in 1:n_rho_values){
  for(m in 1:n_mu_values){
    grid_nll_values[r,m] = negLLprospect_cgt(c(rho_values[r],mu_values[m]), choiceset, simulatedchoices)
  }
}
toc()

min_nll = min(grid_nll_values); # identify the single best value
indexes = which(grid_nll_values == min_nll, arr.ind = T); # Get indices for that single best value

best_rho = rho_values[indexes[1]]; # what are the corresponding rho & mu values?
best_mu = mu_values[indexes[2]];

sprintf('The best R index is %i while the best M indx is %i, with an NLL of %f', indexes[1], indexes[2], min_nll)

c(best_rho, best_mu)
true_vals

fname = sprintf('bespoke_choiceset_rhoInd%03i_muInd%03i.csv', indexes[1], indexes[2]); # Use of %03i creates a three-digit text string with leading 0's as needed for the relevant index; this standardizes file name length


#### Choice Set Creation ####

# Set up variables defining choice set creation
total_number_difficult = 80; # total number of choices in each type
total_number_intermediate = 80; # total number of choices in each type
total_number_easy = 80;

# Probability ranges for easy & difficult categories
choiceP_range_difficult = c(0.45, 0.55);
choiceP_range_easy_lower = 0.10; # implicitly between 0 and this value
choiceP_range_easy_upper = 0.9; # implicitly between this value and 1

# allowable $ values
possible_risky_value_range = c(0.05, 30); 
possible_safe_value_range = c(0.05, 12);

setwd('/Users/sokolhessner/Documents/gitrepos/cgt/choiceset/bespoke_choicesets/');

tic();
for(r in 1:n_rho_values){
  for(m in 1:n_mu_values){
    temp_parameters = c(rho_values[r],mu_values[m]);
    
    newchoices_difficult = array(dim = c(total_number_difficult,5)); # 5 -> riskyoption1, riskyoption2, safeoption, choiceP, easy/difficult
    newchoices_easy = array(dim = c(total_number_easy,5));
    
    choiceP_difficult = array(dim = c(total_number_difficult,1));
    choiceP_easy = array(dim = c(total_number_easy,1));
    
    number_difficult = 0;
    number_easy = 0;
    
    newchoiceoption = array(dim = c(1,5));
    colnames(newchoiceoption) <- c('riskyoption1','riskyoption2','safeoption','choiceP','easy0difficult1');
    newchoiceoption = as.data.frame(newchoiceoption);
    
    number_iterations = 0;
    
    # Make DIFFICULT choices
    while (number_difficult < total_number_difficult){
      number_iterations = number_iterations + 1;
      
      newchoiceoption[1:3] = c(runif(1, min = possible_risky_value_range[1], max = possible_risky_value_range[2]),
                            0,
                            runif(1, min = possible_safe_value_range[1], max = possible_safe_value_range[2]));
      
      choiceP_temporary = choice_probability(temp_parameters,newchoiceoption);
      newchoiceoption[4] = choiceP_temporary;
      newchoiceoption[5] = 1;
      
      if((choiceP_temporary > choiceP_range_difficult[1]) & (choiceP_temporary < choiceP_range_difficult[2])){
        number_difficult = number_difficult + 1;
        newchoices_difficult[number_difficult,] = as.numeric(newchoiceoption);
        choiceP_difficult[number_difficult] = choiceP_temporary;
      }
    }
    print(sprintf('Difficult iterations: %i',number_iterations))
    
    # Make EASY choices
    number_iterations = 0;
    while (number_easy < (total_number_easy/2)){
      number_iterations = number_iterations + 1;
      
      newchoiceoption[1:3] = c(runif(1, min = possible_risky_value_range[1], max = possible_risky_value_range[2]),
                            0,
                            runif(1, min = possible_safe_value_range[1], max = possible_safe_value_range[2]));
      
      choiceP_temporary = choice_probability(temp_parameters,newchoiceoption);
      newchoiceoption[4] = choiceP_temporary;
      newchoiceoption[5] = 0;
      
      if(choiceP_temporary < choiceP_range_easy_lower){
        number_easy = number_easy + 1;
        newchoices_easy[number_easy,] = as.numeric(newchoiceoption);
        choiceP_easy[number_easy] = choiceP_temporary;
      }
    }
    
    while (number_easy < total_number_easy){
      number_iterations = number_iterations + 1;
      
      newchoiceoption[1:3] = c(runif(1, min = possible_risky_value_range[1], max = possible_risky_value_range[2]),
                            0,
                            runif(1, min = possible_safe_value_range[1], max = possible_safe_value_range[2]));
      
      choiceP_temporary = choice_probability(temp_parameters,newchoiceoption);
      newchoiceoption[4] = choiceP_temporary;
      newchoiceoption[5] = 0;
      
      if(choiceP_temporary > choiceP_range_easy_upper){
        number_easy = number_easy + 1;
        newchoices_easy[number_easy,] = as.numeric(newchoiceoption);
        choiceP_easy[number_easy] = choiceP_temporary;
      }
    }
    print(sprintf('Easy iterations: %i',number_iterations))
    
    new_choiceset = rbind(newchoices_easy,newchoices_difficult)
    colnames(new_choiceset) <- c('riskyoption1','riskyoption2','safeoption','choiceP','easy0difficult1');
    new_choiceset = new_choiceset[sample(nrow(new_choiceset)),];
    new_choiceset = as.data.frame(new_choiceset);
    
    fname = sprintf('bespoke_choiceset_rhoInd%03i_muInd%03i.csv', r, m); # Use of %03i creates a three-digit text string with leading 0's as needed for the relevant index; this standardizes file name length
    # Files are roughly 7.5 KB per. 2500 files would be ~18 MB.
    
    write.csv(new_choiceset, file = fname);
    
  }
}
toc()


#### Visualization of Choice Probability surface given different parameters ####

riskyvals = seq(from = possible_risky_value_range[1], to = possible_risky_value_range[2],
                length.out = 100);
safevals = seq(from = possible_safe_value_range[1], to = possible_safe_value_range[2],
               length.out = 101);

choiceP_matrix = array(dim = c(length(riskyvals),length(safevals)));

tempchoiceoption = array(dim = c(1,3));
colnames(tempchoiceoption) <- c('riskyoption1','riskyoption2','safeoption');
tempchoiceoption = as.data.frame(tempchoiceoption);

visualization_rho = 2.2; # range is 0.3 - 1.89
visualization_mu = 7; # expected range is 0-50?

for(i in 1:length(riskyvals)){
  for(j in 1:length(safevals)){
    tempchoiceoption[] = c(riskyvals[i], 0, safevals[j]);

    choiceP_matrix[i,j] = choice_probability(c(visualization_rho, visualization_mu), tempchoiceoption);
  }
}

# Heatmap of the choiceProbability values
pdf(file=sprintf('choice_probability_surface_rho%g_mu%g.pdf', visualization_rho, visualization_mu));
image(riskyvals, safevals, choiceP_matrix,
      col = hcl.colors(100, palette = "red-green", rev = F), 
      breaks = seq(from = 0, to = 1, length.out = 101),
      main = sprintf('Rho = %g, Mu = %g\n min(p) = %.2f, max(p) = %.2f', visualization_rho, visualization_mu, min(choiceP_matrix), max(choiceP_matrix)),
      xlab = 'Risky values ($)', ylab = 'Safe values ($)')
points(choiceset$riskyoption1[choiceset$ischecktrial == 0], choiceset$safeoption[choiceset$ischecktrial == 0])
dev.off();


#### TO DO ####
# - Run final creation of high-resolution files for different parameter combinations. 


#### APPENDIX ####

#### Optimization code ####

negLLprospect_cgt(c(1.2, 20), choiceset, simulatedchoices)
# It works!

eps = .Machine$double.eps;
lower_bounds = c(eps, 0); # R, M
upper_bounds = c(2,50); 
number_of_parameters = length(lower_bounds);

# Create placeholders for parameters, errors, NLL (and anything else you want)
number_of_iterations = 200; # 100 or more
temp_parameters = array(dim = c(number_of_iterations,number_of_parameters));
temp_hessians = array(dim = c(number_of_iterations,number_of_parameters,number_of_parameters));
temp_NLLs = array(dim = c(number_of_iterations,1));

# tic() # start the timer

for(iter in 1:number_of_iterations){
  # Randomly set initial values within supported values
  # using uniformly-distributed values. Many ways to do this!
  
  initial_values = runif(number_of_parameters, min = lower_bounds, max = upper_bounds)
  
  temp_output = optim(initial_values, negLLprospect_cgt,
                      choiceset = choiceset,
                      choices = simulatedchoices,
                      lower = lower_bounds,
                      upper = upper_bounds,
                      method = "L-BFGS-B",
                      hessian = T)
  
  # Store the output we need access to later
  temp_parameters[iter,] = temp_output$par; # parameter values
  temp_hessians[iter,,] = temp_output$hessian; # SEs
  temp_NLLs[iter,] = temp_output$value; # the NLLs
}

# toc() # stop the timer; how long did it take? Use this to plan!

# How'd we do? Look at the NLLs to gauge quality of fit
unique(temp_NLLs) # they look the same but are not...

# Compare output; select the best one
sim_nll = min(temp_NLLs); # the best NLL for this person
sim_best_ind = which(temp_NLLs == sim_nll)[1]; # the index of that NLL

sim_parameters = temp_parameters[sim_best_ind,] # the parameters
sim_parameter_errors = sqrt(diag(solve(temp_hessians[sim_best_ind,,]))); # the SEs

true_vals
sim_parameters
sim_parameter_errors


#### Visualization of Estimated parameters, likelihoods, & easy/difficult lines ####

# Plot actual choices
plot(choiceset$riskyoption1[simulatedchoices == 0], choiceset$safeoption[simulatedchoices == 0],col = 'red',
     xlim = c(0,30), ylim = c(0,12))
points(choiceset$riskyoption1[simulatedchoices == 1], choiceset$safeoption[simulatedchoices == 1],col = 'green')

# Plot the probability of various choices + lines that define the probability regions
pal = colorRampPalette(c('red','white','green'))
choiceset$pal = pal(100)[as.numeric(cut(choiceP, breaks = 100))]
# choiceset$pal = pal(100)[0:100/100]
plot(choiceset$riskyoption1, choiceset$safeoption, col = 'black', bg = choiceset$pal, 
     xlim = c(0,30), ylim = c(0,12), pch = 21)

xval = 35;
r = true_vals[1];
m = true_vals[2];
pval = c(choiceP_range_easy_lower[2], choiceP_range_difficult, choiceP_range_easy_upper[1])
yval_true = array(dim = c(length(pval),1));

for (i in 1:length(pval)){
  yval_true[i] = ((log(1/pval[i] - 1)/(m/(max(choiceset[,1:3])^r)))+0.5*(xval^r))^(1/r)
  lines(x = c(0,xval), y = c(0,yval_true[i]))
}

# Plot the new choice set (THIS MAY NOT WORK, GIVEN EDITS TO CHOICE SET CREATION ABOVE)
plot(new_choiceset$riskyoption1[new_choiceset$easy0difficult1==0],
     new_choiceset$safeoption[new_choiceset$easy0difficult1==0], col = 'blue',
     xlim = c(0,30), ylim = c(0,12))
points(new_choiceset$riskyoption1[new_choiceset$easy0difficult1==1],new_choiceset$safeoption[new_choiceset$easy0difficult1==1], col = 'red')

r = sim_parameters[1];
m = sim_parameters[2];
yval_sim = array(dim = c(length(pval),1));

for (i in 1:length(pval)){
  yval_sim[i] = ((log(1/pval[i] - 1)/(m/(max(choiceset[,1:3])^r)))+0.5*(xval^r))^(1/r)
  lines(x = c(0,xval), y = c(0,yval_sim[i]), lty = 'dashed')
}
