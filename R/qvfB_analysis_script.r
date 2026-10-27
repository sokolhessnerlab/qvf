

# QVF Study B (Model Predicted Intermediate Difficulty) Data Analysis Script
###############################################################################

# combine the choice P and actual choice to create to likelihood

# choice likelihood should look closer for high WMC
# look farther for low WMC

# this time this is predicting likelihood 

# intermediate was based on the softmax 2561
# mu should be lower which would mean 
# throw out negative mu values - these are people who are subverting the task
# issues with change terms and parameters
# - 
# - we do a change to make everything possitive 2579
# mu will have a baseline value that increases base on previous difficukty
# lower to 0 values means that there are no effects of previous difficulty
# then does this is 

# we then run an optimizer 
# bounds have been set - exp

# hard to specify what is a reasonable change for these paramters (currently using addition)
# we could use multiplication to carry out a percentage change
# we are using an exp transformation 

# change and constrained model 
# LRT tests - does a change in mu fit better than the static

# doing the paralellizations

# check the results (in the old one where it started no results cause of binning etc.)
# t-test
# LRT


# get to the likelihood stage - estimation can come later 
# this should all work now that we have intermediate difficulty

# i might have to change some of the code for easy1prev0 - the numbers might be a little different
# - checking the bounds of what we define as easy and difficulty
# - just a little different now because 

# i can run the previous basic analyses - see if we replicate the previous trial 
# i can also now check differences against intermediate difficutlty


# cge likelihood functions changed the bounds of highest choice set creation

# with inclusion of intermediate, our effects may no longer be linear effects
# - for example curvilinear 
# - think about how intermediate difficulty would change the kinds of effects
# - difficulty in our original studies 

# what was the paper that accounted for equating physical effort across people???

# checkl the estimation code - how we define mu




# maybe I can load the file for all the pariticpants that  
# do a loop within an if 
# repopulate the values based on the matched choices values 
# set values that aren't recoverable and set to NA - don't make any assumptions




# STEP 01: CLEAR THE SPACE #####

rm(list=ls()); # Clear the workspace

# STEP 02: SET WORKING DIRECTORY ##### 

setwd('/Users/shlab/Documents/GitHub/qvf/qvf_StudyB')
#setwd('/Users/sokolhessner/Documents/gitrepos/qvf/qvf_StudyB')

# STEP 03: RUN FROM SHLAB DRIVE #####

## using the config.yml to more flexibly access the shlab drive across multiple devices -----
config = config::get()
setwd(config$path$data$processed)

# STEP 04: LOAD DATA #####

## Load Decision-Making Data -----
fn = dir(pattern = glob2rx('qvfB_processed_decisionmaking_data*.csv'),full.names = T)
number_of_files = length(fn) # ASSUMES YOU WANT THE MOST-RECENT PROCESSED DATA
data_dm = read.csv(fn[number_of_files]) # decision-making data

## Load Working Memory Data -----
fn = dir(pattern = glob2rx('qvfB_processed_complexspan_data*.csv'),full.names = T)
number_of_files = length(fn) # ASSUMES YOU WANT THE MOST-RECENT PROCESSED DATA
complexSpanScores = read.csv(fn[number_of_files]) # working memory data

# ## Load Qualtrics Survey Data -----
# fn = dir(pattern = glob2rx('qvfB_processed_survey_data*.csv'),full.names = T)
# number_of_files = length(fn) # ASSUMES YOU WANT THE MOST-RECENT PROCESSED DATA
# survey_data = read.csv(fn[number_of_files]) # working memory data

number_of_subjects = length(unique(data_dm$subjectnumber))

# STEP 05: EXCLUDE 'BAD' DATA #####

## decision-making data check -----

### EXCLUSION: check trials -----

check_trial_failurerate = array(dim = c(number_of_subjects,1));

for (subj in 1:number_of_subjects){
  tmpdata = data_dm[data_dm$subjectnumber == subj,];
  check_trial_index = which(tmpdata$ischecktrial==1);
  correct_answers = (0.5 * tmpdata$riskyopt1[check_trial_index] +
                       0.5 * tmpdata$riskyopt2[check_trial_index]) > tmpdata$safe[check_trial_index];
  check_trial_failurerate[subj] = length(which(!tmpdata$choice[check_trial_index] == correct_answers))/length(check_trial_index);

  # Plot the choice data
  plot(tmpdata$riskyopt1[tmpdata$choice == 1],tmpdata$safe[tmpdata$choice == 1], col = 'green',
       xlab = 'Risky Gain $', ylab = 'Safe $', main = paste0('All Subjects; Subj ', subj),
       xlim = c(0,30), ylim = c(0,12))
  points(tmpdata$riskyopt1[tmpdata$choice == 0],tmpdata$safe[tmpdata$choice == 0], col = 'red')
}

check_trial_criterion = 0.2; # The maximum percent of check trials that can be missed
# (there were 10 check trials)
# chance is 0.5, perfect is 0.0.

keep_check_trial = check_trial_failurerate <= check_trial_criterion; # 2 did not meet criteria: 16, 45, 76 (4/5/24)

### EXCLUSION: check trials -----
mean_rts = array(dim = c(number_of_subjects,1));

for (subj in 1:number_of_subjects){
  tmpdata = data_dm[data_dm$subjectnumber == subj,];

  mean_rts[subj] = mean(tmpdata$reactiontime, na.rm = T)
}

keep_dm_rt = mean_rts > 0.85; # drop no one

mean_rts[keep_dm_rt]
hist(mean_rts[keep_dm_rt]) # histogram of mean rts
mean(mean_rts[keep_dm_rt]) # mean rt 1.548 seconds (2/25/25)

# STEP 06: BASIC ANALYSES #####

keep_participants = which(keep_check_trial & keep_dm_rt);

# Create clean data frames for data!
clean_data_dm = data_dm[data_dm$subjectnumber %in% keep_participants,]
clean_data_complexspan = complexSpanScores[complexSpanScores$subjectnumber %in% keep_participants,]
#clean_data_survey = survey_data[survey_data$subjectID %in% keep_participants,]

number_of_clean_subjects = length(keep_participants);
number_of_clean_subjects # 71 participants (Feb '25)

# Create a re-scaled version of trial number for use in subsequent analyses
clean_data_dm$trialnumberRS = clean_data_dm$trialnumber/max(clean_data_dm$trialnumber)

# Create a better-behaved version of dprime using a square-root transform
#clean_data_dm$sqrtdprime = sqrt(abs(clean_data_dm$dprime))*((clean_data_dm$dprime>=0)*2-1)
#clean_data_dm$sqrtdprime_zeroed = clean_data_dm$sqrtdprime + min(clean_data_dm$sqrtdprime, na.rm = T);
# Have to do it this way, b/c of (slight) negative dprime values
#clean_data_survey$sqrtdprime = sqrt(abs(clean_data_survey$dprime))*((clean_data_survey$dprime>=0)*2-1)
# hist(clean_data_survey$dprime)
# hist(sqrt(clean_data_survey$dprime))

library(corrplot)

## survey data -----

## decision-making data -----
# Analysis for static trials: Mean p(gamble), optimization
mean_pgamble = array(dim = c(number_of_clean_subjects,1));
static_mean_pgamble = array(dim = c(number_of_clean_subjects, 1));
dynamic_mean_pgamble = array(dim = c(number_of_clean_subjects, 1));
easy_mean_pgamble = array(dim = c(number_of_clean_subjects, 1));
diff_mean_pgamble = array(dim = c(number_of_clean_subjects, 1));
easyACC_mean_pgamble = array(dim = c(number_of_clean_subjects, 1));
easyREJ_mean_pgamble = array(dim = c(number_of_clean_subjects, 1));

# e0i1d2
easy00_mean_pgamble = array(dim = c(number_of_clean_subjects, 1));
inte01_mean_pgamble = array(dim = c(number_of_clean_subjects, 1));
diff02_mean_pgamble = array(dim = c(number_of_clean_subjects, 1));
easy00ACC_mean_pgamble = array(dim = c(number_of_clean_subjects, 1));
easy00REJ_mean_pgamble = array(dim = c(number_of_clean_subjects, 1));
inte01ACC_mean_pgamble = array(dim = c(number_of_clean_subjects, 1));
inte01REJ_mean_pgamble = array(dim = c(number_of_clean_subjects, 1));

for (subj in 1:number_of_clean_subjects){
  subj_id = keep_participants[subj];
  tmpdata = clean_data_dm[clean_data_dm$subjectnumber == subj_id,]; # defines this person's data
  
  mean_pgamble[subj] = mean(tmpdata$choice, na.rm = T);
  static_mean_pgamble[subj] = mean(tmpdata$choice[tmpdata$static0dynamic1 == 0], na.rm=T);
  dynamic_mean_pgamble[subj] = mean(tmpdata$choice[tmpdata$static0dynamic1 == 1], na.rm=T);
  easy_mean_pgamble[subj] = mean(tmpdata$choice[tmpdata$easyP1difficultN1 == 1], na.rm = T);
  diff_mean_pgamble[subj] = mean(tmpdata$choice[tmpdata$easyP1difficultN1 == -1], na.rm = T);
  easyACC_mean_pgamble[subj] = mean(tmpdata$choice[(tmpdata$easyP1difficultN1 == 1) & (tmpdata$choiceP > .98)], na.rm = T);
  easyREJ_mean_pgamble[subj] = mean(tmpdata$choice[(tmpdata$easyP1difficultN1 == 1) & (tmpdata$choiceP < .02)], na.rm = T);

  easy00_mean_pgamble[subj] = mean(tmpdata$choice[tmpdata$e0i1d2 == 0], na.rm = T);
  inte01_mean_pgamble[subj] = mean(tmpdata$choice[tmpdata$e0i1d2 == 1], na.rm = T);
  diff02_mean_pgamble[subj] = mean(tmpdata$choice[tmpdata$e0i1d2 == 2], na.rm = T);
  easy00ACC_mean_pgamble[subj] = mean(tmpdata$choice[(tmpdata$easyP1difficultN1 == 1) & (tmpdata$choiceP > .98)], na.rm = T);
  easy00REJ_mean_pgamble[subj] = mean(tmpdata$choice[(tmpdata$easyP1difficultN1 == 1) & (tmpdata$choiceP < .02)], na.rm = T);
  inte01ACC_mean_pgamble[subj] = mean(tmpdata$choice[(tmpdata$easyP1difficultN1 == 1) & (tmpdata$choiceP > .78) & (tmpdata$choiceP < .92)], na.rm = T);
  inte01REJ_mean_pgamble[subj] = mean(tmpdata$choice[(tmpdata$easyP1difficultN1 == 1) & (tmpdata$choiceP > .08) & (tmpdata$choiceP < .22)], na.rm = T);

  # Plot the choice data
  plot(tmpdata$riskyopt1[tmpdata$choice == 1],tmpdata$safe[tmpdata$choice == 1], col = 'green',
       xlab = 'Risky Gain $', ylab = 'Safe $', main = paste0('Kept Subjects; Subj ', subj_id),
       xlim = c(0,30), ylim = c(0,12))
  points(tmpdata$riskyopt1[tmpdata$choice == 0],tmpdata$safe[tmpdata$choice == 0], col = 'red')
}

# Create simple summary object with p(gamble) descriptives
column_names_rdm = c(
  'mean_pgamble',
  'static_mean_pgamble',
  'dynamic_mean_pgamble',
  'easy_mean_pgamble',
  'diff_mean_pgamble',
  'easyACC_mean_pgamble',
  'easyREJ_mean_pgamble',
  'easy00_mean_pgamble',
  'inte01_mean_pgamble',
  'diff02_mean_pgamble',
  'easyACC_mean_pgamble',
  'easyREJ_mean_pgamble',
  'inte01ACC_mean_pgamble',
  'inte01REJ_mean_pgamble'
);

data_rdm = array(data = NA, dim = c(0, length(column_names_rdm)));
colnames(data_rdm) <-column_names_rdm
data_rdm <- data.frame(mean_pgamble,static_mean_pgamble,dynamic_mean_pgamble,easy_mean_pgamble,diff_mean_pgamble,easyACC_mean_pgamble,easyREJ_mean_pgamble)

#Q:standard error, means, range for gambling behavior for particpants

#Define function
# sem <- function(x) sd(x)/sqrt(length(x));
# SEM function (FANCY)
sem <- function(x){
  if(!is.null(dim(drop(x)))){ # if x is 2-dimensional
    sem_value = array(dim = dim(x)[1]) # make SEM vector assuming ROWS
    for (row_num in 1:dim(x)[1]){
      sem_value[row_num] = sd(x[row_num,], na.rm = T)/sqrt(sum(!is.na(x[row_num,])))
    }
  } else { # if unidimensional
    sem_value = sd(x, na.rm = T)/sqrt(sum(!is.na(x)))
  }
  return(sem_value)
}


# Calculate M's and SEMs
mean(mean_pgamble) # 0.5119297
mean(static_mean_pgamble) # 0.5171569
mean(dynamic_mean_pgamble) # 0.5109657
sem(mean_pgamble) # 0.01744603
sem(static_mean_pgamble) # 0.01998875
sem(dynamic_mean_pgamble) # 0.01989382

mean(easyACC_mean_pgamble) # 0.9509031
mean(easyREJ_mean_pgamble) # 0.05719322
sem(easyACC_mean_pgamble) # 0.01528863
sem(easyREJ_mean_pgamble) # 0.01166399

mean(diff_mean_pgamble) # 0.520023
sem(diff_mean_pgamble) # 0.03502338

min(mean_pgamble) # 0.2176871
max(mean_pgamble) # 0.6530612

mean(easy00ACC_mean_pgamble) # 0.9509031
mean(easy00REJ_mean_pgamble) # 0.05719322
sem(easy00ACC_mean_pgamble) # 0.01528863
sem(easy00REJ_mean_pgamble) # 0.01166399

mean(inte01ACC_mean_pgamble) # 1
mean(inte01REJ_mean_pgamble) # 0.025
sem(inte01ACC_mean_pgamble) # 0.07155975
sem(inte01REJ_mean_pgamble) # 0.07438333

# Q: How did risk-taking compare across the different dynamic trial types?
# e.g. easy accept vs. easy reject vs. difficult
hist(easyACC_mean_pgamble,
     col = rgb(0,1,0,.5), breaks = seq(from = 0, to = 1, by = 0.05), xlim = c(0,1),
     xlab = 'Mean Probability of Risk-Taking', ylab = 'Frequency',
     main = 'Average Risk-Taking Behavior By Trial Type')
hist(easyREJ_mean_pgamble,
     col = rgb(1,0,0,.5), breaks = seq(from = 0, to = 1, by = 0.05), add = TRUE)
hist(diff_mean_pgamble,
     col = rgb(0,0,1,.5), breaks = seq(from = 0, to = 1, by = 0.05), add = TRUE)
# A: As expected red < blue < green (easy reject < difficult < easy accept), blue is more spread out


# Q: Can we collapse across easy accept & reject trials based on relative distance of CHOICES from 0.5?
plot(abs(easyACC_mean_pgamble-.5),abs(easyREJ_mean_pgamble-.5),xlim = c(0,.5), ylim = c(0,.5),
     xlab = 'Easy Accept', ylab = 'Easy Reject',
     main = 'Distance of Observed p(risky) from 0.5 (chance) for EASY trial subtypes',
     pch = 19, col = rgb(0,0,0,.2), cex = 2)
lines(x = c(0,1), y = c(0,1), col = 'blue')

# Statistically test relative difficulty observed in easy ACC vs. easy REJ
t.test(abs(easyACC_mean_pgamble-.5), abs(easyREJ_mean_pgamble-.5), paired = T) # t = 0.74377, df = 28, p-value = 0.4632
wilcox.test(abs(easyACC_mean_pgamble-.5), abs(easyREJ_mean_pgamble-.5), paired = T) # p = 0.1106

## Optimization Function Creation ############################################

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
negLLprospect_cge <- function(parameters,choiceset,choices) {
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


## Optimization ############################################
eps = .Machine$double.eps;
lower_bounds = c(eps, 0); # R, M
upper_bounds = c(2,80);
number_of_parameters = length(lower_bounds);

# Create placeholders for parameters, errors, NLL (and anything else you want)
number_of_iterations = 200; # 100 or more
estimated_parameters = array(dim = c(number_of_clean_subjects,2));
estimated_parameter_errors = array(dim = c(number_of_clean_subjects,2));
NLLs = array(dim = c(number_of_clean_subjects,1));

clean_data_dm$all_choiceP = NA;

cat('Beginning Optimization\n')

for (subj in 1:number_of_clean_subjects){
  subj_id = keep_participants[subj];
  print(subj_id)

  tmpdata = clean_data_dm[(clean_data_dm$subjectnumber == subj_id) &
                            (clean_data_dm$static0dynamic1 == 0) &
                            is.finite(clean_data_dm$choice),]; # defines this person's data

  temp_parameters = array(dim = c(number_of_iterations,number_of_parameters));
  temp_hessians = array(dim = c(number_of_iterations,number_of_parameters,number_of_parameters));
  temp_NLLs = array(dim = c(number_of_iterations,1));

  choiceset = as.data.frame(cbind(tmpdata$riskyopt1, tmpdata$riskyopt2, tmpdata$safe));
  colnames(choiceset) <- c('riskyoption1', 'riskyoption2', 'safeoption');

  # tic() # start the timer

  for(iter in 1:number_of_iterations){
    # Randomly set initial values within supported values
    # using uniformly-distributed values. Many ways to do this!

    initial_values = runif(number_of_parameters, min = lower_bounds, max = upper_bounds)

    temp_output = optim(initial_values, negLLprospect_cge,
                        choiceset = choiceset,
                        choices = tmpdata$choice,
                        lower = lower_bounds,
                        upper = upper_bounds,
                        method = "L-BFGS-B",
                        hessian = T)

    # Store the output we need access to later
    temp_parameters[iter,] = temp_output$par; # parameter values
    temp_hessians[iter,,] = temp_output$hessian; # SEs
    temp_NLLs[iter,] = temp_output$value; # the NLLs
  }

  # Compare output; select the best one
  NLLs[subj] = min(temp_NLLs); # the best NLL for this person
  best_ind = which(temp_NLLs == NLLs[subj])[1]; # the index of that NLL

  estimated_parameters[subj,] = temp_parameters[best_ind,] # the parameters
  estimated_parameter_errors[subj,] = sqrt(diag(solve(temp_hessians[best_ind,,]))); # the SEs

  # Calculating all choice probabilities for this participant, given best-fit parameters
  all_choice_ind = (clean_data_dm$subjectnumber == subj_id) & is.finite(clean_data_dm$choice)
  tmpdata = clean_data_dm[all_choice_ind,]; # defines this person's data

  choiceset = as.data.frame(cbind(tmpdata$riskyopt1, tmpdata$riskyopt2, tmpdata$safe));
  colnames(choiceset) <- c('riskyoption1', 'riskyoption2', 'safeoption');

  clean_data_dm$all_choiceP[all_choice_ind] = choice_probability(temp_parameters[best_ind,],choiceset);
}


## Grid Search ############################################

### Q: Does optimized analysis match grid search analysis?###

n_rho_values = 200; # SET THIS TO THE DESIRED DEGREE OF FINENESS
n_mu_values = 201; # IBID

rho_values = seq(from = 0.3, to = 2.2, length.out = n_rho_values); # the range of fit-able values
mu_values = seq(from = 7, to = 80, length.out = n_mu_values);

best_rhos = array(dim = c(number_of_clean_subjects,1));
best_mus = array(dim = c(number_of_clean_subjects,1));

cat('Beginning Grid Search\n')

for (subj in 1:number_of_clean_subjects){
  subj_id = keep_participants[subj];
  print(subj_id)
  #
  tmpdata = clean_data_dm[(clean_data_dm$subjectnumber == subj_id) &
                            (clean_data_dm$static0dynamic1 == 0) &
                            is.finite(clean_data_dm$choice),]; # defines this person's data

  choiceset = as.data.frame(cbind(tmpdata$riskyopt1, tmpdata$riskyopt2, tmpdata$safe));
  colnames(choiceset) <- c('riskyoption1', 'riskyoption2', 'safeoption');

  grid_nll_values = array(dim = c(n_rho_values, n_mu_values));

  for(r in 1:n_rho_values){
    for(m in 1:n_mu_values){
      grid_nll_values[r,m] = negLLprospect_cge(c(rho_values[r],mu_values[m]), choiceset, tmpdata$choice)
    }
  }

  min_nll = min(grid_nll_values); # identify the single best value
  indexes = which(grid_nll_values == min_nll, arr.ind = T); # Get indices for that single best value

  best_rhos[subj] = rho_values[indexes[1]]; # what are the corresponding rho & mu values?
  best_mus[subj] = mu_values[indexes[2]];
}

# look at all best rho & mu per participant
grid_bestRho = array(dim = c(number_of_clean_subjects,1));
grid_bestMu = array(dim = c(number_of_clean_subjects,1));
for (subj in 1:number_of_clean_subjects){
  subj_id = keep_participants[subj];

  tmpdata = clean_data_dm[clean_data_dm$subjectnumber == subj_id,];

  grid_bestRho[subj] = rho_values[unique(tmpdata$bestRho)];
  grid_bestMu[subj] = mu_values[unique(tmpdata$bestMu)];
}

# First, check fresh grid search best Rho & Mu against experiment-executed grid search Rho & Mu
# (should be trivial and match!)

if (any((grid_bestRho - best_rhos) != 0)){
  print('RHO MISMATCH!')
}else{
  print('RHO Grid search values match (as expected)')
}

if (any((grid_bestMu - best_mus) != 0)){
  print('MU MISMATCH!')
}else{
  print('MU Grid search values match (as expected)')
}


# Grid search does *not* replicate... WHY NOT? 

# Then check estimated parameters vs. grid search parameters
plot(grid_bestRho,estimated_parameters[,1], main = 'RHO',
     xlab = 'Grid Search Estimate', ylab = 'Optimization estimate',
     xlim = c(0, 2), ylim = c(0, 2))
lines(c(0, 2), c(0, 2))

plot(grid_bestMu,estimated_parameters[,2], main = 'MU',
     xlab = 'Grid Search Estimate', ylab = 'Optimization estimate',
     xlim = c(0, 100), ylim = c(0, 100))
lines(c(0, 100), c(0, 100))

hist(grid_bestRho - estimated_parameters[,1], xlim = c(-2,2),
     breaks = seq(from = -2, to = 2, by = 0.02), main = 'Difference in Rho Estimates',
     ylab = 'Participants', xlab = 'Grid estimate - MLE estimate')
hist(grid_bestMu - estimated_parameters[,2], xlim = c(-100,100),
     breaks = seq(from = -100, to = 100, by = 0.5), main = 'Difference in Mu Estimates',
     ylab = 'Participants', xlab = 'Grid estimate - MLE estimate')
# This is supposed to look silly! Should cluster around 0
# ... and it does, as of 4/5/24!

t.test(grid_bestRho, estimated_parameters[,1], paired = T) # no sig. diff (rho)... 4/11/24 (t(84) = 1.7804, p = 0.07862)
t.test(grid_bestMu, estimated_parameters[,2], paired = T) # no sig. diff (mu)... 4/11/24 (t(84) = 1.8545, p = 0.06718)

cor.test(grid_bestRho, estimated_parameters[,1]) # both extremely highly correlated. r(83) = 0.9974685, p = 2.2e-16
cor.test(grid_bestMu, estimated_parameters[,2]) # r(83) = 0.9997898, p = 2.2e-16

# A: YES, grid-search values match optimized values very closely.

# Who are our subjects?
mean_params = colMeans(estimated_parameters);

hist(estimated_parameters[,1], xlim = c(0,2),
     breaks = seq(from = 0, to = 2, by = 0.2), main = '',
     ylab = 'Participants', xlab = 'Rho (risk attitudes)')
lines(x = c(1,1), y = c(0,50), col = 'black', lwd = 2, lty = 'dashed')
lines(x = c(mean_params[1], mean_params[1]), y = c(0,50), col = 'green', lwd = 5, lend = 'butt')
# Seems evenly spread around 1 (risk neutral)

hist(estimated_parameters[,2], xlim = c(0,upper_bounds[2]),
     breaks = seq(from = 0, to = upper_bounds[2], by = 8), main = '',
     ylab = 'Participants', xlab = 'Mu (consistency)')
lines(x = c(mean_params[2], mean_params[2]), y = c(0,50), col = 'orange', lwd = 5, lend = 'butt')
# Seems positively skewed. Around 20-30.




