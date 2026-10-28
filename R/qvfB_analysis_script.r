

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

# setwd('/Users/shlab/Documents/GitHub/qvf/qvf_StudyB')
setwd('/Users/sokolhessner/Documents/gitrepos/qvf/qvf_StudyB')

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

keep_check_trial = check_trial_failurerate <= check_trial_criterion; # 2 did not meet criteria

### EXCLUSION: reaction times -----
mean_rts = array(dim = c(number_of_subjects,1));

for (subj in 1:number_of_subjects){
  tmpdata = data_dm[data_dm$subjectnumber == subj,];

  mean_rts[subj] = mean(tmpdata$reactiontime, na.rm = T)
}

keep_dm_rt = mean_rts > 0.85; # drop no one

mean_rts[keep_dm_rt]
hist(mean_rts[keep_dm_rt]) # histogram of mean rts
mean(mean_rts[keep_dm_rt]) # mean rt 1.51 seconds (10/2026)

# STEP 06: BASIC ANALYSES #####

keep_participants = which(keep_check_trial & keep_dm_rt);

# Create clean data frames for data!
clean_data_dm = data_dm[data_dm$subjectnumber %in% keep_participants,]
clean_data_complexspan = complexSpanScores[complexSpanScores$subjectnumber %in% keep_participants,]
#clean_data_survey = survey_data[survey_data$subjectID %in% keep_participants,]

number_of_clean_subjects = length(keep_participants);
number_of_clean_subjects # 29 participants (10/2026)

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
  easyACC_mean_pgamble[subj] = mean(tmpdata$choice[(tmpdata$easyP1difficultN1 == 1) & (tmpdata$choiceP > .97)], na.rm = T);
  easyREJ_mean_pgamble[subj] = mean(tmpdata$choice[(tmpdata$easyP1difficultN1 == 1) & (tmpdata$choiceP < .03)], na.rm = T);

  easy00_mean_pgamble[subj] = mean(tmpdata$choice[tmpdata$e0i1d2 == 0], na.rm = T);
  inte01_mean_pgamble[subj] = mean(tmpdata$choice[tmpdata$e0i1d2 == 1], na.rm = T);
  diff02_mean_pgamble[subj] = mean(tmpdata$choice[tmpdata$e0i1d2 == 2], na.rm = T);
  easy00ACC_mean_pgamble[subj] = mean(tmpdata$choice[(tmpdata$e0i1d2 == 0) & (tmpdata$choiceP > .97)], na.rm = T);
  easy00REJ_mean_pgamble[subj] = mean(tmpdata$choice[(tmpdata$e0i1d2 == 0) & (tmpdata$choiceP < .03)], na.rm = T);
  inte01ACC_mean_pgamble[subj] = mean(tmpdata$choice[(tmpdata$e0i1d2 == 1) & (tmpdata$choiceP > .77) & (tmpdata$choiceP < .93)], na.rm = T);
  inte01REJ_mean_pgamble[subj] = mean(tmpdata$choice[(tmpdata$e0i1d2 == 1) & (tmpdata$choiceP > .07) & (tmpdata$choiceP < .23)], na.rm = T);

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
     col = rgb(0,1,0,.5), breaks = seq(from = 0, to = 1, by = 0.05), xlim = c(0,1), ylim = c(0,25),
     xlab = 'Mean Probability of Risk-Taking', ylab = 'Frequency',
     main = 'Average Risk-Taking Behavior By Trial Type')
hist(easyREJ_mean_pgamble,
     col = rgb(1,0,0,.5), breaks = seq(from = 0, to = 1, by = 0.05), add = TRUE)
hist(diff_mean_pgamble,
     col = rgb(0,0,1,.5), breaks = seq(from = 0, to = 1, by = 0.05), add = TRUE)
hist(inte01ACC_mean_pgamble,
     col = rgb(0,1,0,.8), breaks = seq(from = 0, to = 1, by = 0.05), add = TRUE)
hist(inte01REJ_mean_pgamble,
     col = rgb(1,0,0,.8), breaks = seq(from = 0, to = 1, by = 0.05), add = TRUE)

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
  div <- 30^rho; # decorrelates rho & mu

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

# NOTE:
#  Grid Search Optimization from Subjects 1-31 used the wrong rho and mu values. 
#  Choice set generation had used 0.35-2.2 (200 values) for rho, and 11-80 (201
#  values) for mu. In the Python script for S1-31, the script used 0.3-2.2 (200
#  values) for rho and 7-80 (201 values) for mu. As a result, the script 
#  identified the incorrect indices for the bespoke choice sets. Degree of 
#  error was greater the lower the estimated rho and mu values were as the 
#  lower values in the ranges were where the difference between rho & mu values
#  was present (upper values are identical). 

n_rho_values = 200; # SET THIS TO THE DESIRED DEGREE OF FINENESS
n_mu_values = 201; # IBID

rho_values = seq(from = 0.35, to = 2.2, length.out = n_rho_values); # the range of fit-able values
mu_values = seq(from = 11, to = 80, length.out = n_mu_values); # the range of fit-able values

best_rhos = array(dim = c(number_of_clean_subjects,1));
best_mus = array(dim = c(number_of_clean_subjects,1));

cat('Beginning Fresh Grid Search\n')

for (subj in 1:number_of_clean_subjects){
  subj_id = keep_participants[subj];
  print(subj_id)
  
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

# Identify the rho and mu values associated with the dynamic choice set that was
# actually used with this participant.
#
# (Not always accurate b/c of error, see above)
selectedchoiceset_bestRho = array(dim = c(number_of_clean_subjects,1));
selectedchoiceset_bestMu = array(dim = c(number_of_clean_subjects,1));

old_rho_values = seq(from = 0.3, to = 2.2, length.out = n_rho_values); # the range of fit-able values
old_mu_values = seq(from = 7, to = 80, length.out = n_mu_values); # the range of fit-able values

for (subj in 1:number_of_clean_subjects){
  subj_id = keep_participants[subj];

  tmpdata = clean_data_dm[clean_data_dm$subjectnumber == subj_id,];
  
  r_ind = unique(tmpdata$bestRho)
  m_ind = unique(tmpdata$bestMu)
  
  selectedchoiceset_bestRho[subj] = rho_values[r_ind];
  selectedchoiceset_bestMu[subj] = mu_values[m_ind];
}

# Plot the grid-search-fit Rho values against the rho values associated 
# with the choiceset actually used with this participant (this will be off
# for participants 1-31 b/c of above error). Should be accurate thereafter!
plot(best_rhos[keep_participants <= 31], selectedchoiceset_bestRho[keep_participants <= 31],
     col = 'purple', pch = 4, xlim = c(0,2.3), ylim = c(0, 2.3),
     xlab = 'Grid Search Estimate', ylab = 'Choiceset Parameter Value', main = 'Rho Grid Search & Choice Set') # these will not be on the line
lines(x = c(0,3), y = c(0,3))
points(best_rhos[keep_participants > 31], selectedchoiceset_bestRho[keep_participants > 31]) # these should be on the line

plot(best_mus[keep_participants <= 31], selectedchoiceset_bestMu[keep_participants <= 31],
     col = 'purple', pch = 4, xlim = c(0, 85), ylim = c(0, 85),
     xlab = 'Grid Search Estimate', ylab = 'Choiceset Parameter Value', main = 'Mu Grid Search & Choice Set') # these will not be on the line
lines(x = c(0,100), y = c(0,100))
points(best_rhos[keep_participants > 31], selectedchoiceset_bestRho[keep_participants > 31]) # these should be on the line


# Q: How off were the rho values for the first 31 people?
hist(best_rhos[keep_participants <= 31] - selectedchoiceset_bestRho[keep_participants <= 31])
mean(best_rhos[keep_participants <= 31] - selectedchoiceset_bestRho[keep_participants <= 31])
median(best_rhos[keep_participants <= 31] - selectedchoiceset_bestRho[keep_participants <= 31])
# A: An average of 0.022 (median of 0.028; MAX possible should be ~0.05, MIN possible should be ~0)
# Max error is possible at low rho values, min error is possible at high rho values.

hist(best_mus[keep_participants <= 31] - selectedchoiceset_bestMu[keep_participants <= 31])
mean(best_mus[keep_participants <= 31] - selectedchoiceset_bestMu[keep_participants <= 31])
median(best_mus[keep_participants <= 31] - selectedchoiceset_bestMu[keep_participants <= 31])
# A: An average of -1.30 (median of -1.73; MAX possible should be ~4, MIN possible should be ~0)
# Max error is possible at low rho values, min error is possible at high rho values.

# The errors are systematic but SMALL IN MAGNITUDE. 

# Then check estimated parameters vs. grid search parameters
plot(best_rhos,estimated_parameters[,1], main = 'RHO',
     xlab = 'Grid Search Estimate', ylab = 'Optimization estimate',
     xlim = c(0, 2), ylim = c(0, 2))
lines(c(0, 2), c(0, 2))

plot(best_mus,estimated_parameters[,2], main = 'MU',
     xlab = 'Grid Search Estimate', ylab = 'Optimization estimate',
     xlim = c(0, 100), ylim = c(0, 100))
lines(c(0, 100), c(0, 100))

hist(best_rhos - estimated_parameters[,1], xlim = c(-2,2),
     breaks = seq(from = -2, to = 2, by = 0.02), main = 'Difference in Rho Estimates',
     ylab = 'Participants', xlab = 'Grid estimate - MLE estimate')
hist(best_mus - estimated_parameters[,2], xlim = c(-100,100),
     breaks = seq(from = -100, to = 100, by = 0.5), main = 'Difference in Mu Estimates',
     ylab = 'Participants', xlab = 'Grid estimate - MLE estimate')
# This is supposed to look silly! Should cluster around 0
# ... and it does, as of 4/5/24!

t.test(best_rhos, estimated_parameters[,1], paired = T) # no sig. diff (rho)... 4/11/24 (t(84) = 1.7804, p = 0.07862)
t.test(best_mus, estimated_parameters[,2], paired = T) # no sig. diff (mu)... 4/11/24 (t(84) = 1.8545, p = 0.06718)

cor.test(best_rhos, estimated_parameters[,1]) # both extremely highly correlated. r(83) = 0.9974685, p = 2.2e-16
cor.test(best_mus, estimated_parameters[,2]) # r(83) = 0.9997898, p = 2.2e-16

# A: YES, grid-search values match optimized values very closely.

# There was an error in identifying the correct bespoke file for the first 
# 31 participants. The consequence of this is that somewhat incorrect choice 
# sets were selected (they were slightly off). Indices in 0.3-2.2 and 7-80 were
# used to pick choicesets in 0.35-2.2 and 11-80 space. The ACTUAL parameter values
# associated with a given choice set were thus systematically slightly too high.
# The offset was generally very small (see above) in parameter space. 

# Visualize mismatch between choiceP (choice set assumed choice likelihood) and
# allchoiceP (estimated choice likelihood given behavior). 
plot(clean_data_dm$choiceP[clean_data_dm$subjectnumber <= 31],
     clean_data_dm$all_choiceP[clean_data_dm$subjectnumber <= 31], 
     pch = 16, col = rgb(0,0,0,.2), xlim = c(0,1), ylim = c(0,1),
     xaxs = "i", yaxs = 'i',
     xlab = 'Assumed choice likelihoods', ylab = 'Actual fit choice likelihoods')
# abline(v = c(0.02, 0.98, 0.08, 0.22, 0.78, 0.92, 0.45, 0.55))
# abline(h = c(0.02, 0.98, 0.08, 0.22, 0.78, 0.92, 0.45, 0.55))
polygon(x = c(0.45, 0.55, 0.55, 0.45, 0.45), # difficult box
        y = c(0.45, 0.45, 0.55, 0.55, 0.45), border = 'red', lwd = 2)
polygon(x = c(0.08, 0.22, 0.22, 0.08, 0.08), # lower intermediate box
        y = c(0.08, 0.08, 0.22, 0.22, 0.08), border = 'purple', lwd = 2)
polygon(x = c(0, 0.02, 0.02, 0, 0), # lower easy box
        y = c(0, 0, 0.02, 0.02, 0), border = 'green', lwd = 2)
polygon(x = c(0.78, 0.92, 0.92, 0.78, 0.78), # upper intermediate box
        y = c(0.78, 0.78, 0.92, 0.92, 0.78), border = 'purple', lwd = 2)
polygon(x = c(0.98, 1, 1, 0.98, 0.98), # upper easy box
        y = c(0.98, 0.98, 1, 1, 0.98), border = 'green', lwd = 2)

# CONSEQUENCES OF THE GRID SEARCH ERROR FOR ANALYSIS
# - Stimulus set difficulty is slightly off.
# - continuous difficulty metrics are trustworthy
# - categorical approaches are weakened, as actual difficulty varies outside
#   of bounds slightly more than expected.
#
# Reasons not to be concerned:
# - We can still quantify difficulty on a per trial, per-person basis. 
# - Easy/Intermediate/Difficult trial categories still rank appropriately (just
#   more variable within-category than planned). 


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


# STEP 07: START SOME REGRESSIONS #####

### Create Continuous difficulty metric ###
#clean_data_dm$diff_cont = abs(abs(clean_data_dm$choiceP - 0.5)*2-1); # JUST for the easy/difficult dynamic trials
clean_data_dm$all_diff_cont = abs(abs(clean_data_dm$all_choiceP - 0.5)*2-1); # for ALL trials

clean_data_dm$prev_all_diff_cont = c(NA,clean_data_dm$all_diff_cont[1:(nrow(clean_data_dm)-1)]) # for ALL trials
clean_data_dm$prev_all_diff_cont[clean_data_dm$trialnumber == 1] = NA;

clean_data_dm$prev2_all_diff_cont = c(NA,clean_data_dm$prev_all_diff_cont[1:(length(clean_data_dm$prev_all_diff_cont)-1)]) # for ALL trials
clean_data_dm$prev2_all_diff_cont[clean_data_dm$trialnumber == 1] = NA;

clean_data_dm$prev3_all_diff_cont = c(NA,clean_data_dm$prev2_all_diff_cont[1:(length(clean_data_dm$prev2_all_diff_cont)-1)]) # for ALL trials
clean_data_dm$prev3_all_diff_cont[clean_data_dm$trialnumber == 1] = NA;

clean_data_dm$prev4_all_diff_cont = c(NA,clean_data_dm$prev3_all_diff_cont[1:(length(clean_data_dm$prev3_all_diff_cont)-1)]) # for ALL trials
clean_data_dm$prev4_all_diff_cont[clean_data_dm$trialnumber == 1] = NA;

clean_data_dm$prev5_all_diff_cont = c(NA,clean_data_dm$prev4_all_diff_cont[1:(length(clean_data_dm$prev4_all_diff_cont)-1)]) # for ALL trials
clean_data_dm$prev5_all_diff_cont[clean_data_dm$trialnumber == 1] = NA;

clean_data_dm$prev6_all_diff_cont = c(NA,clean_data_dm$prev5_all_diff_cont[1:(length(clean_data_dm$prev5_all_diff_cont)-1)]) # for ALL trials
clean_data_dm$prev6_all_diff_cont[clean_data_dm$trialnumber == 1] = NA;

clean_data_dm$prev7_all_diff_cont = c(NA,clean_data_dm$prev6_all_diff_cont[1:(length(clean_data_dm$prev6_all_diff_cont)-1)]) # for ALL trials
clean_data_dm$prev7_all_diff_cont[clean_data_dm$trialnumber == 1] = NA;

clean_data_dm$prev8_all_diff_cont = c(NA,clean_data_dm$prev7_all_diff_cont[1:(length(clean_data_dm$prev7_all_diff_cont)-1)]) # for ALL trials
clean_data_dm$prev8_all_diff_cont[clean_data_dm$trialnumber == 1] = NA;


# Visualize relationship between choice probability and difficulty
plot(clean_data_dm$all_choiceP, clean_data_dm$all_diff_cont, 
     pch = 16, col = rgb(0,0,0,.1), xaxs = 'i', yaxs = 'i',
     xlab = 'Choice Likelihood (0 = safe, 1 = risky)',
     ylab = 'Choice Difficulty (0 = easy, 1 = difficult)')
abline(v = c(0.02, 0.08, 0.22, 0.45, 0.55, 0.78, 0.92, 0.98))


# What are the Difficulty values that correspond to the different
# choice likelihood cutoffs we've used? 
difficulty_bounds = abs(abs(c(0.02, 0.08, 0.22, 0.45, 0.55, 0.78, 0.92, 0.98) - 0.5)*2-1)
# Easy = 0-0.04
# Int = 0.16-0.44
# Diff = 0.90-1

plot(clean_data_dm$all_diff_cont, clean_data_dm$prev_all_diff_cont,
     pch = 16, col = rgb(0,0,0,.2), xaxs = 'i', yaxs = 'i',
     xlab = 'Current Difficulty',
     ylab = 'Previous Difficulty')
abline(v = c(0.04, 0.16, 0.44, 0.9), col = 'darkorange', lwd = 3)
abline(h = c(0.04, 0.16, 0.44, 0.9), col = 'darkorange', lwd = 3)



## Basic RT analysis ############################################

# Q:DO RT differ across conditions? Reaction times for easy risky, easy safe, and hard (hard > easy (either))
#mean easy RT
mean_rt_static = array(dim = c(number_of_clean_subjects, 1));
mean_rt_dynamic = array(dim = c(number_of_clean_subjects, 1));
mean_rt_easy = array(dim = c(number_of_clean_subjects, 1));
mean_rt_diff = array(dim = c(number_of_clean_subjects, 1));
median_rt_easy = array(dim = c(number_of_clean_subjects, 1));
median_rt_diff = array(dim = c(number_of_clean_subjects, 1));
var_rt_easy = array(dim = c(number_of_clean_subjects, 1));
var_rt_diff = array(dim = c(number_of_clean_subjects, 1));
mean_rt_easyACC = array(dim = c(number_of_clean_subjects, 1));
mean_rt_easyREJ = array(dim = c(number_of_clean_subjects, 1));

mean_rt_easy00 = array(dim = c(number_of_clean_subjects, 1));
mean_rt_inte01 = array(dim = c(number_of_clean_subjects, 1));
mean_rt_diff02 = array(dim = c(number_of_clean_subjects, 1));
mean_rt_easy00ACC = array(dim = c(number_of_clean_subjects, 1));
mean_rt_easy00REJ = array(dim = c(number_of_clean_subjects, 1));
mean_rt_inte01ACC = array(dim = c(number_of_clean_subjects, 1));
mean_rt_inte01REJ = array(dim = c(number_of_clean_subjects, 1));


for (subj in 1:number_of_clean_subjects){
  subj_id = keep_participants[subj];
  tmpdata = data_dm[data_dm$subjectnumber == subj_id,]; # needs to be clean data dm not data dm????
  
  mean_rt_static[subj] = mean(tmpdata$reactiontime[(tmpdata$static0dynamic1 == 0)],na.rm = T);
  mean_rt_dynamic[subj] = mean(tmpdata$reactiontime[(tmpdata$static0dynamic1 == 1)], na.rm = T);
  
  # Identify just EASY dynamic data
  tmpdataEasyDyn = tmpdata[tmpdata$easyP1difficultN1 == 1,];
  
  # RTs within easy dynamic data
  mean_rt_easy[subj] = mean(tmpdataEasyDyn$reactiontime, na.rm = T)
  median_rt_easy[subj] = median(tmpdataEasyDyn$reactiontime, na.rm = T)
  mean_rt_easyACC[subj] = mean(tmpdataEasyDyn$reactiontime[(tmpdataEasyDyn$all_choiceP > .97)], na.rm = T);
  mean_rt_easyREJ[subj] = mean(tmpdataEasyDyn$reactiontime[(tmpdataEasyDyn$all_choiceP < .03)], na.rm = T);
  var_rt_easy[subj] = var(tmpdataEasyDyn$reactiontime, na.rm = T);
  
  # Identify just DIFFICULT dynamic data
  tmpdataDiffDyn = tmpdata[tmpdata$easyP1difficultN1 == -1,];
  
  mean_rt_diff[subj] = mean(tmpdataDiffDyn$reactiontime, na.rm = T)
  median_rt_diff[subj] = median(tmpdataDiffDyn$reactiontime, na.rm = T)
  var_rt_diff[subj] = var(tmpdataDiffDyn$reactiontime, na.rm = T);
  
}


# RTs between easy & difficult
plot(mean_rt_diff, mean_rt_easy,
     main = 'Average RT By Trial Type', xlab = "Mean RT Difficult", ylab = "Mean RT Easy",
     xlim = c(0.75, 2.75), ylim = c(0.75, 2.75))
lines(c(0, 2.75), c(0, 2.75))
points(mean(mean_rt_diff), mean(mean_rt_easy), pch = 16, col = rgb(1, 0, 1, .75), cex = 2.5)

plot(median_rt_diff, median_rt_easy,
     main = 'Median RT By Trial Type', xlab = "Median RT Difficult", ylab = "Median RT Easy",
     xlim = c(0.75, 2.75), ylim = c(0.75, 2.75))
lines(c(0, 2.75), c(0, 2.75))
points(mean(median_rt_diff), mean(median_rt_easy), pch = 16, col = rgb(1, 0, 1, .75), cex = 2.5)

# test easy RTs vs. diff. RTs
rt_mean_test = t.test(mean_rt_easy,mean_rt_diff, paired = T) # VERY significant t(70)= -7.7118, p = 6.197e-11
mean_diff = mean(mean_rt_diff, na.rm = T) # 1.686421
sd_diff = sd(mean_rt_diff, na.rm = T) # 0.3432346
mean_easy = mean(mean_rt_easy, na.rm = T) # 1.447765
sd_easy = sd(mean_rt_easy, na.rm = T) # 0.223144
rt_mean_test




# WORKING MEMORY BASIC ANALYSIS #####

cat(sprintf('Out of a total of %i participants, we have O-Span scores for %i, Sym-Span scores for %i, and composite span scores for %i.\n',
            number_of_subjects,
            sum(is.finite(complexSpanScores$ospanScore)),
            sum(is.finite(complexSpanScores$symspanScore)),
            sum(is.finite(complexSpanScores$compositeSpanScore))))
# Out of a total of 31 participants, we have O-Span scores for 19, Sym-Span scores for 24, and composite span scores for 27 10/27/26
cat(sprintf('%i participants have both scores, %i participants are missing only one score, and %i participants are missing both scores.\n',
            sum(is.finite(complexSpanScores$ospanScore) & is.finite(complexSpanScores$symspanScore)),
            sum(xor(is.finite(complexSpanScores$ospanScore),is.finite(complexSpanScores$symspanScore))),
            number_of_subjects-sum(is.finite(complexSpanScores$compositeSpanScore))))
# 16 participants have both scores, 11 participants are missing only one score, and 4 participants are missing both scores.10/27/26
# Mean, Median, and Variance of ospan, symspan, and compositespan
mean_ospan = mean(complexSpanScores$ospanScore, na.rm = T)
mean_symspan = mean(complexSpanScores$symspanScore, na.rm = T)
mean_compositespan = mean(complexSpanScores$compositeSpanScore, na.rm = T)

sd_ospan = sd(complexSpanScores$ospanScore, na.rm = T)
sd_symspan = sd(complexSpanScores$symspanScore, na.rm = T)
sd_compositespan = sd(complexSpanScores$compositeSpanScore, na.rm = T)

median_ospan = median(complexSpanScores$ospanScore, na.rm = T)
median_symspan = median(complexSpanScores$symspanScore, na.rm = T)
median_compositespan = median(complexSpanScores$compositeSpanScore, na.rm = T)

variance_ospan = var(complexSpanScores$ospanScore, na.rm = T)
variance_symspan = var(complexSpanScores$symspanScore, na.rm = T)
variance_compositespan = var(complexSpanScores$compositeSpanScore, na.rm = T)

cat("Mean ospan:", mean_ospan, "\n") # 0.5536842 
cat("Mean symspan:", mean_symspan, "\n") # 0.5595238
cat("Mean compositeSpan:", mean_compositespan, "\n") # 0.5334392 

cat("Median ospan:", median_ospan, "\n")
cat("Median symspan:", median_symspan, "\n")
cat("Median compositeSpan:", median_compositespan, "\n")

cat("Variance ospan:", variance_ospan, "\n")
cat("Variance symspan:", variance_symspan, "\n")
cat("Variance compositeSpan:", variance_compositespan, "\n")

max_compositespan = max(complexSpanScores$compositeSpanScore, na.rm = T)
cat("Max composite span score:", max_compositespan, "\n")

# Include in the processing - Correlation between OSpan and SymSpan
ospanScores = complexSpanScores$ospanScore 
symspanScores = complexSpanScores$symspanScore
compositeSpanScores = complexSpanScores$compositeSpanScore
# clean_data_dm$wmc_cont = compositeSpanScores[keep_participants]
# SEE: clean_data_dm$complexspan or clean_data_dm$complexspan_demeaned

cor.test(ospanScores, symspanScores) # t = 0.1485, df = 14, p-value = 0.8841
var.test(ospanScores, symspanScores) # F = 0.93014, num df = 18, denom df = 23, p-value = 0.8869
t.test(ospanScores, symspanScores, paired = T) # t = -0.036029, df = 15, p-value = 0.9717

# SUMMARY: O-Span & Sym-Span scores are not correlated, and have a similar variance (is this because there are such different samples?) 
# different from one another. They are NOT redundant.

plot(ospanScores, symspanScores,
     pch = 19, col = rgb(.5, .5, .5, .5),
     xlim = c(0, 1), ylim = c(0, 1), cex = 2.5,
     xlab = 'OSpan Scores', ylab = 'SymSpan Scores',
     main = 'Complex Span Scores')
lines(x = c(-1, 2), y = c(-1, 2)) # so line extends to edge

capacity_HighP1_lowN1 = (compositeSpanScores > median_compositespan)*2-1;
# NOTE: This is all participants who completed the WMC task
# Includes people we drop from e.g. decision-making analysis

sum(capacity_HighP1_lowN1 == 1, na.rm = T) # 13
sum(capacity_HighP1_lowN1 == -1, na.rm = T) # 14

# Plot the distribution w/ the median value
hist(compositeSpanScores, breaks = 10, xlab = 'Composite Span Score', main = 'Distribution of Spans');
abline(v = median_compositespan, col = 'red', lwd = 5)



# Limiting analysis to those people we kept for DECISION-MAKING ANALYSIS...
sum(clean_data_complexspan$compositeSpanScore > median(clean_data_complexspan$compositeSpanScore, na.rm = T), na.rm = T)
sum(clean_data_complexspan$compositeSpanScore <= median(clean_data_complexspan$compositeSpanScore, na.rm = T), na.rm = T)
# 12 with > median capacity
# 13 with <= median capacity

# Create *CLEAN DATA* capacity median split
clean_data_complexspan$capacity_HighP1_lowN1 = (clean_data_complexspan$compositeSpanScore > 
                                                  median(clean_data_complexspan$compositeSpanScore, na.rm = T))*2-1;

clean_data_dm$capacity_HighP1_lowN1 = NA;

for(s in 1:number_of_clean_subjects){
  subj_id = keep_participants[s];
  clean_data_dm$capacity_HighP1_lowN1[clean_data_dm$subjectnumber == subj_id] = 
    clean_data_complexspan$capacity_HighP1_lowN1[clean_data_complexspan$subjectnumber == subj_id];
}

clean_data_dm$complexspan_demeaned = clean_data_dm$complexspan - mean(clean_data_complexspan$compositeSpanScore, na.rm = T);

plot(sort(compositeSpanScores))
abline(h = median_compositespan, col = 'red', lwd = 2)


# RT on trials regressions #####

library(lme4)
library(lmerTest)

clean_data_dm$sqrtRT = sqrt(clean_data_dm$reactiontime);

# WHY WAS THIS DONE IN EDI???

# clean_data_dm$easy = as.double(clean_data_dm$easyP1difficultN1 == 1)
# clean_data_dm$difficult = as.double(clean_data_dm$easyP1difficultN1 == -1)

# # input shifted version of desired content
# clean_data_dm$easyP1difficultN1_prev = c(0,clean_data_dm$easyP1difficultN1[1:(length(clean_data_dm$easyP1difficultN1)-1)])
# # fix the few problematic trials (#1)
# clean_data_dm$easyP1difficultN1_prev[clean_data_dm$trialnumber == 1] = 0;
# # input shifted version of desired content
# clean_data_dm$sqrtRT_prev = c(NA,clean_data_dm$sqrtRT[1:(length(clean_data_dm$sqrtRT)-1)])
# # fix the few problematic trials (#1)
# clean_data_dm$sqrtRT_prev[clean_data_dm$trialnumber == 1] = NA;

# DEMEANED DTs: Current and Previous Decision Time # THIS WAS DONE IN CGE
mean_sqrtRT = mean(clean_data_dm$sqrtRT, na.rm = T)
clean_data_dm$sqrtRT_demeaned = clean_data_dm$sqrtRT - mean_sqrtRT # currDT

clean_data_dm$sqrtRT_demeaned_prev = c(NA,clean_data_dm$sqrtRT_demeaned[1:(length(clean_data_dm$sqrtRT_demeaned)-1)]) # prevDT
clean_data_dm$sqrtRT_demeaned_prev[clean_data_dm$trialnumber == 1] = NA;

## Model 0: Current RT based on current easy difficult
m0_diffcat = lm(sqrtRT ~ 1 + easyP1difficultN1, data = clean_data_dm); # LM
summary(m0_diffcat) # difficulty predicts RT! # faster on easy

m0_diffcat_rfx = lmer(sqrtRT ~ 1 + easyP1difficultN1 + (1 | subjectnumber), data = clean_data_dm); # LMER
summary(m0_diffcat_rfx) # Same with RFX!

m0_diffcat_dynonly_rfx = lmer(sqrtRT ~ 1 + easyP1difficultN1 + (1 | subjectnumber),
                              data = clean_data_dm[clean_data_dm$static0dynamic1 == 1,]);
summary(m0_diffcat_dynonly_rfx) # Same with only dynamic trials

# Model 0 takeaway: in all cases, people are slower in difficult trials

# use continuous diff metric instead of easy/difficult 

# NOT USING diff_cont because it uses choiceP
# m0_diffcont = lm(sqrtRT ~ 1 + diff_cont , data = clean_data_dm);
# summary(m0_diffcont) # matches categorical
# 
# m0_diffcont_rfx = lmer(sqrtRT ~ 1 + diff_cont + (1 | subjectnumber), data = clean_data_dm);
# summary(m0_diffcont_rfx) # matches categorical
# 
# m0_diffcont_dynonly_rfx = lmer(sqrtRT ~ 1 + diff_cont + (1 | subjectnumber),
#                                data = clean_data_dm[clean_data_dm$static0dynamic1 == 1,]);
# summary(m0_diffcont_dynonly_rfx) # matches categorical

m0_alldiffcont = lm(sqrtRT ~ 1 + all_diff_cont, data = clean_data_dm);
summary(m0_alldiffcont) # matches categorical

m0_alldiffcont_rfx = lmer(sqrtRT ~ 1 + all_diff_cont + (1 | subjectnumber), data = clean_data_dm);
summary(m0_alldiffcont_rfx) # matches categorical

# Which model should we use?
# It's between m0_diffcat_rfx and m0_alldiffcont_rfx
AIC(m0_diffcat_rfx) # -5943.512
AIC(m0_alldiffcont_rfx) # -6057.361 <- BETTER (more negative)

anova(m0_diffcat_rfx,m0_alldiffcont_rfx) # CONFIRMS that all_diff_cont outperforms easyp1difficultn1

xval_plot = seq(from = 0, to = 1, by = .1);
coef_vals = fixef(m0_alldiffcont_rfx)

plot(x = xval_plot, y = (coef_vals["(Intercept)"] + xval_plot*coef_vals["all_diff_cont"])^2,
     type = 'l', lwd = 5, col = 'purple',
     main = 'Effect of current difficulty', xlab = 'Difficulty (0 = easy, 1 = difficult)', ylab = 'Reaction Time (seconds)')

# BIG TAKEAWAY:
# Across categorical and two kinds of continuous difficulty, difficult trials are slower.
#
# m0_alldiffcont_rfx is best (see AICs)

## Model 1: PREVIOUS DIFFICULTY: Create Shifted versions of difficulty for use in regressions

# input shifted version of desired content
clean_data_dm$easyP1difficultN1_prev = c(0,clean_data_dm$easyP1difficultN1[1:(length(clean_data_dm$easyP1difficultN1)-1)])
# fix the few problematic trials (#1)
clean_data_dm$easyP1difficultN1_prev[clean_data_dm$trialnumber == 1] = 0;
# input shifted version of desired content
clean_data_dm$sqrtRT_prev = c(NA,clean_data_dm$sqrtRT[1:(length(clean_data_dm$sqrtRT)-1)]) ### Previous Decision Time was here...
# fix the few problematic trials (#1)
clean_data_dm$sqrtRT_prev[clean_data_dm$trialnumber == 1] = NA;

# Does previous difficulty influence subsequent RT?
# LMs
m1_diffcat_prev = lm(sqrtRT ~ 1 + easyP1difficultN1 + easyP1difficultN1_prev, data = clean_data_dm);
summary(m1_diffcat_prev) # slight effect, OPPOSITE to main pattern (p = 0.0466 as of 2/25/24)
#  (if prev. trial easy, slower on current trial .... OR
#   if prev. trial difficult, faster on current trial)

m1_diffcat_prev_intxn = lm(sqrtRT ~ 1 + easyP1difficultN1 * easyP1difficultN1_prev, data = clean_data_dm);
summary(m1_diffcat_prev_intxn) # no interaction, same main effect as m1_diffcat_prev

# LMERs, i.e. RFX Versions
m1_diffcat_prev_rfx = lmer(sqrtRT ~ 1 + easyP1difficultN1 + easyP1difficultN1_prev +
                             (1 | subjectnumber), data = clean_data_dm);
summary(m1_diffcat_prev_rfx) # previous difficulty has strong effect (p = 0.02)
# Same direction as in m1_diffcat_prev

m1_diffcat_prev_intxn_rfx = lmer(sqrtRT ~ 1 + easyP1difficultN1 * easyP1difficultN1_prev + (1 | subjectnumber), data = clean_data_dm);
summary(m1_diffcat_prev_intxn_rfx) # no interaction, same main effect.

# TAKEAWAY: Previous categorical difficulty has OPPOSITE effect on current RTs as current difficulty.
#    THIS IS DIFFERENT FROM CGT! CGT HAD NO MAIN EFFECT OF PREV. CATEGORICAL DIFFICULTY.

m1_prev_alldiffCont_intxn_rfx = lmer(sqrtRT ~ 1 +
                                       all_diff_cont * prev_all_diff_cont +
                                       (1 | subjectnumber), data = clean_data_dm);
summary(m1_prev_alldiffCont_intxn_rfx) # Previous CONTINUOUS difficulty is VERY significant (p = 1.2e-7), no interaction w/ current diff
# Sign is negative: the more difficult the prev. trial was, the faster people were on the current trial
# ... facilitory? "giving up"?

xval_plot = seq(from = 0, to = 1, length.out = 10)

predict_data_m1 = clean_data_dm[0,];
predict_data_m1[1:20,] = NA;
predict_data_m1$all_diff_cont[1:10] = xval_plot
predict_data_m1$all_diff_cont[11:20] = xval_plot
predict_data_m1$prev_all_diff_cont[1:10] = 0;
predict_data_m1$prev_all_diff_cont[11:20] = 1;

predict_output_m1 = predict(m1_prev_alldiffCont_intxn_rfx, newdata = predict_data_m1, type = 'response', re.form = NA)^2

# Plot it!
plot(x = xval_plot, y = predict_output_m1[1:10],
     type = 'l', lwd = 5, col = 'blue',
     main = 'Effect of current & previous difficulty', xlab = 'Current difficulty (0 = easy, 1 = difficult)', ylab = 'Reaction Time (seconds)',
     ylim = c(1.3, 1.8))
lines(x = xval_plot, y = predict_output_m1[11:20],
      lwd = 5, col = 'red')

# BLUE = previous trial easy
# RED = previous trial difficult

# Plot Previous Difficult Choice only
par(mfrow = c(1,1))
par(mar = c(5, 6, 4, 4))
plot(x = xval_plot, y = predict_output_m1[1:10], cex.axis = 1.3, las = 1,
     type = 'l', lwd = 5, col = 'blue',ylim = c(1.3, 1.8),
     xlab = expression(bold("Current Difficulty")), ylab = expression(bold("Reaction Time (seconds)")), cex.lab = 1.5,)
lines(x = xval_plot, y = predict_output_m1[11:20],
      lwd = 5, col = 'red')
legend("topleft", legend = c(expression(bold("Previous Difficulty")), "Easy Choice", "Difficult Choice"), cex = 1.3, y.intersp = 1.3,
       col = c(NA, 'blue', 'red'), lwd = c(NA, 2, 2),
       bty = "o", inset = c(0.025, 0.025))

# TAKEAWAY: I AM NOT SEEING PREVIOUS DIFFICULTY EFFECTS... ???



# SKIPPING TO WORKING MEMORY CAPACITY #####

# Examine best-fitting threshold values
best_testing_model = as.formula('sqrtRT ~ 1 + all_diff_cont * capacity_HighP1_lowN1_temp * trialnumberRS +
                  prev_all_diff_cont * capacity_HighP1_lowN1_temp * trialnumberRS +
                  (1 | subjectnumber)')
best_model = as.formula('sqrtRT ~ 1 + all_diff_cont * capacity_HighP1_lowN1_best * trialnumberRS +
                  prev_all_diff_cont * capacity_HighP1_lowN1_best * trialnumberRS +
                  (1 | subjectnumber)')

unique_compositeSpanScores_clean = sort(unique(clean_data_complexspan$compositeSpanScore)) # note: the use of sort() removes NAs in composite span scores!
possible_threshold_values = diff(unique_compositeSpanScores_clean)/2+unique_compositeSpanScores_clean[1:(length(unique_compositeSpanScores_clean)-1)]

all_aic_values = array(data = NA, dim = length(possible_threshold_values));

clean_data_dm$capacity_HighP1_lowN1_temp = NA;

for(ind in 1:length(possible_threshold_values)){
  break_val = possible_threshold_values[ind];
  clean_data_dm$capacity_HighP1_lowN1_temp[clean_data_dm$complexspan > break_val] = 1;
  clean_data_dm$capacity_HighP1_lowN1_temp[clean_data_dm$complexspan < break_val] = -1;
  
  cat(sprintf('This many people are < break_val: %g\n',sum(clean_data_complexspan$compositeSpanScore<break_val, na.rm = T)))
  
  if((sum(clean_data_complexspan$compositeSpanScore<break_val, na.rm = T) <= 3) | (sum(clean_data_complexspan$compositeSpanScore>break_val, na.rm = T) <= 3)){
    next # don't use any categorizations that create a 'group' with 3 or fewer people
  }
  
  m3_tmp = lmer(best_testing_model, data = clean_data_dm, REML = F);
  all_aic_values[ind] = AIC(m3_tmp)
}

best_aic = min(all_aic_values, na.rm = T);
best_threshold = possible_threshold_values[which(all_aic_values == best_aic)];

cat(sprintf('The best fitting model uses a CompositeSpan threshold value of %g.\n', best_threshold)) # 0.142858
cat(sprintf('The best AIC obtained with the CompositeSpan was %0.2f\n', best_aic)) # -5717.21

plot(sort(clean_data_complexspan$compositeSpanScore), xlab = 'Participants', ylab = 'Span')
abline(h = best_threshold, col = 'blue', lwd = 4)
abline(h = median(clean_data_complexspan$compositeSpanScore, na.rm = T), col = 'red', lwd = 4, lty = 'dotted') # median value
legend('topleft', legend = c('Span','Best threshold','Median span'), col = c('black','blue', 'red'), lty = 1)

plot(x = possible_threshold_values, y = all_aic_values, type = 'l', col = 'green', xlab = 'Thresholds', ylab = 'AIC values (lower better)', lwd = 3, xlim = c(0.1, 0.9) )
abline(v = median_compositespan, col = 'red', lwd = 4, lty = 'dotted')
abline(v = best_threshold, col = 'blue', lwd = 4)
legend('bottomleft', legend = c('AIC values', 'Median span', 'Best threshold'), col = c('green','red','blue'), lty = 1)
# The best-fitting threshold (just slightly above the median value) is 0.6571439

break_val = best_threshold; # as of 3/19/25, this was 0.6785724

# capacity_HighP1_lowN1_Best = (clean_data_complexspan$compositeSpanScore > break_val)*2 - 1;

# Make the variable in the DM dataframe
clean_data_dm$capacity_HighP1_lowN1_best[clean_data_dm$complexspan > break_val] = 1;
clean_data_dm$capacity_HighP1_lowN1_best[clean_data_dm$complexspan < break_val] = -1;

# Make the variable in the Span dataframe
clean_data_complexspan$capacity_HighP1_lowN1_best[clean_data_complexspan$compositeSpanScore > break_val] = 1;
clean_data_complexspan$capacity_HighP1_lowN1_best[clean_data_complexspan$compositeSpanScore < break_val] = -1;

# Visual Inspection of AIC curves suggest that moderate cutoff values (between composite
# span values of ~0.35 and ~0.55) all do equivalently and reasonably well. This 
# is broadly consistent with cutoff values observed in prior studies, if slightly
# lower (the current [10/28/2026] best-fitting value with N = 31 is 0.68)
# CGE: 0.6785724
# CGT: ??
# EDI: 0.6571439
#
# The three cutoff values producing equivalent-ish fits are:
# 0.3692857 0.4571429 0.5571429

# Use a more moderate cutoff value
moderate_cutoff_value = 0.4571429
clean_data_dm$capacity_HighP1_lowN1_moderate[clean_data_dm$complexspan > moderate_cutoff_value] = 1
clean_data_dm$capacity_HighP1_lowN1_moderate[clean_data_dm$complexspan < moderate_cutoff_value] = -1



# Continuous difficulty (including previous) and categorical capacity
m3_prev_diffCont_capacityCatMedian_intxn_rfx = lmer(sqrtRT ~ 1 +
                                                all_diff_cont * prev_all_diff_cont * capacity_HighP1_lowN1 +
                                                (1 | subjectnumber), data = clean_data_dm);
summary(m3_prev_diffCont_capacityCatMedian_intxn_rfx)

m3_prev_diffCont_capacityCatBest_intxn_rfx = lmer(sqrtRT ~ 1 +
                                                    all_diff_cont * capacity_HighP1_lowN1_best +
                                                    prev_all_diff_cont * capacity_HighP1_lowN1_best +
                                                    (1 | subjectnumber), data = clean_data_dm);
summary(m3_prev_diffCont_capacityCatBest_intxn_rfx)

m3_prev_diffCont_capacityCatModerate_intxn_rfx = lmer(sqrtRT ~ 1 +
                                                        all_diff_cont * capacity_HighP1_lowN1_moderate + 
                                                        prev_all_diff_cont * capacity_HighP1_lowN1_moderate +
                                                        (1 | subjectnumber), data = clean_data_dm);
summary(m3_prev_diffCont_capacityCatModerate_intxn_rfx)


# Summary 10/28/2026
#
# - Broadly, these regressions capture the same patterns we've seen before:
#   - current difficulty has a strong positive effect on decision time with
#   people being slower the harder the trial is. 
#   - Previous difficulty has a negative effect on decision time, with people 
#   being faster after more difficult trials. 
#
# - Things that seem different at the moment (with N = 31) include 1) that previous
# difficulty effect is not (always) significant and 2) that WMC is not reliably
# emerging as an important modifier of current and previous difficulty effects. 
# Possibly compounding this is the fact that the WMC data is lower quality than
# in previous versions (EDI, CGE, CGT) - more people are missing one or both
# of the WMC measures than before (in CGE, 6/85 people were missing both and 
# 23/85 were missing 1; in EDI, 6/73 were missing both & 33/73 were missing one; 
# here, 4/31 are missing both and 11/31 missing one). 
#
# - Two other major differences here are 1) the inclusion of ~1/3 intermediate trials
# which we have not used before, and 2) significantly more trials (294) of the 
# decision-making task (CGT/CGE/EDI: 180) which take place before the WMC 
# measures. 


# Things to consider checking/analyzing:
# 
# 1. Nonlinear effects of time (simple averages of RT by blocks of 10 trials? or 4?)
# 2. Nonlinear effects of difficulty (current and/or previous)
#
# To carry these out, There are two main approaches: 
# 1. model-free averages based on binning (across time, across curr & prev. diff.)
# 2. nonlinear transforms that are then used in e.g. regressions. 
#
# Option #1 is likely the easiest to do (just requires care in binning).
#   - bin RTs as a function of trial #
#   - do more categorical curr/prev difficulty analyses
#




### Throwing In The Towel ############################################

### Explaining lower RTs (faster choices) after difficult trials:
# Are low-capacity participants "throwing in the towel" after difficult choices, or
# is their choice process *facilitated*, and thus better?

# APPROACH:
#   1. Use all_choice_P values (calculated from estimates of rho & mu from static trials) to...
#   2. Calculate the LIKELIHOOD of data on a per-trial basis in dynamic trials, and then...
#   3. Calculate the MEAN LIKELIHOOD of dynamic trials that follow a difficult trial
#      and compare it to the MEAN LIKELIHOOD of dynamic trials that follow an easy trial
#   4. Calculate the difference in MEAN LIKELIHOOD (easy - difficult), and...
#   5. Relate that difference to capacity group and/or continuous capacity.

# 1. is already done.

# 2. Calculate the likelihood of the choices.

clean_data_dm$all_choice_likelihood = clean_data_dm$choice * clean_data_dm$all_choiceP + (1-clean_data_dm$choice) * (1-clean_data_dm$all_choiceP);

# 3. Calculate the mean likelihood on dynamic trials after diff. vs. easy trials.

mean_choice_lik_prevEasy = array(data = NA, dim = c(number_of_clean_subjects,1));
mean_choice_lik_prevDiff = array(data = NA, dim = c(number_of_clean_subjects,1));

for (s in 1:number_of_clean_subjects){
  subj_id = keep_participants[s]
  tmpdata = clean_data_dm[clean_data_dm$subjectnumber == subj_id,];
  
  mean_choice_lik_prevEasy[s] = mean(tmpdata$all_choice_likelihood[tmpdata$easyP1difficultN1_prev == 1], na.rm = T); # na.rm b/c of missed trials
  mean_choice_lik_prevDiff[s] = mean(tmpdata$all_choice_likelihood[tmpdata$easyP1difficultN1_prev == -1], na.rm = T);
  
  # Use this code to run this test ONLY on current difficult trials
  # mean_choice_lik_prevEasy[s] = mean(tmpdata$all_choice_likelihood[(tmpdata$easyP1difficultN1_prev == 1) & (tmpdata$easyP1difficultN1 == -1)], na.rm = T); # na.rm b/c of missed trials
  # mean_choice_lik_prevDiff[s] = mean(tmpdata$all_choice_likelihood[(tmpdata$easyP1difficultN1_prev == -1) & (tmpdata$easyP1difficultN1 == -1)], na.rm = T);
  
  # Use this code to run this test ONLY on current easy trials
  # mean_choice_lik_prevEasy[s] = mean(tmpdata$all_choice_likelihood[(tmpdata$easyP1difficultN1_prev == 1) & (tmpdata$easyP1difficultN1 == 1)], na.rm = T); # na.rm b/c of missed trials
  # mean_choice_lik_prevDiff[s] = mean(tmpdata$all_choice_likelihood[(tmpdata$easyP1difficultN1_prev == -1) & (tmpdata$easyP1difficultN1 == 1)], na.rm = T);
}

t.test(mean_choice_lik_prevEasy, mean_choice_lik_prevDiff, paired = T) # t = 0.22866, df = 28, p-value = 0.8208
wilcox.test(mean_choice_lik_prevEasy, mean_choice_lik_prevDiff, paired = T) # V = 227, p-value = 0.848
# 

cor.test(mean_choice_lik_prevEasy, mean_choice_lik_prevDiff) # t = 3.7522, df = 27, p-value = 0.0008494, r = 0.5854272 
cor.test(mean_choice_lik_prevEasy, mean_choice_lik_prevDiff, method = 'spearman') # S = 2084, p-value = 0.00807
# Correlated w/ each other, unsurprisingly.

plot(mean_choice_lik_prevEasy, mean_choice_lik_prevDiff)
lines(x = c(0, 1), y = c(0, 1))

# 4. Calculate the difference in mean choice likelihood as a function of prev. difficulty

mean_choice_lik_relative = mean_choice_lik_prevEasy - mean_choice_lik_prevDiff;
# Positive numbers = Likelihood after easy is HIGHER than after difficult.
# Negative numbers = likelihood after easy is LOWER than after difficult.
#
# We predict more positive numbers for low-capacity folks.

# 5. Relate that difference to capacity group and/or continuous capacity.

plot(mean_choice_lik_relative, compositeSpanScores[keep_participants])
cor.test(mean_choice_lik_relative, compositeSpanScores[keep_participants]) # t = 1.4348, df = 23, p-value = 0.1648, r = 0.2866311 
cor.test(mean_choice_lik_relative, compositeSpanScores[keep_participants], method = 'spearman')
# Direction of the (non-sig.) correlation is negative, as expected
# Low capacity folks have a greater gap between their choice likelihood
# after easy vs. difficult (and easy is >> difficult).

# t.test(mean_choice_lik_relative[capacity_HighP1_lowN1_Best == 1], mean_choice_lik_relative[capacity_HighP1_lowN1_Best == -1])
# wilcox.test(mean_choice_lik_relative[capacity_HighP1_lowN1_Best == 1], mean_choice_lik_relative[capacity_HighP1_lowN1_Best == -1])
# n.s. p = 0.63 on 2/25/24
# High cap. mean difference = -0.017
# Low cap. mean difference = -0.011 (more positive)


# Try a regression-based approach to this whole question?
likmodel_catDiff_catCap = lmer(all_choice_likelihood ~ 1 + easyP1difficultN1 * easyP1difficultN1_prev * capacity_HighP1_lowN1_best +
                                 (1 | subjectnumber), data = clean_data_dm)
summary(likmodel_catDiff_catCap)
# no signs of anything going on in previous difficulty

likmodel_contDiff_catCap = lmer(all_choice_likelihood ~ 1 + all_diff_cont * prev_all_diff_cont * capacity_HighP1_lowN1_best +
                                  (1 | subjectnumber), data = clean_data_dm)
summary(likmodel_contDiff_catCap)
# no signs of anything going on in previous difficulty


# Another Approach - look for lower choice consistency after difficult trials
negLLprospect_cge_muPrevDiff <- function(parameters,choiceset,choices) {
  # A negative log likelihood function for a prospect-theory estimation.
  # Assumes parameters are [rho, mu, muchange]
  # Assumes choiceset has columns riskyoption1, riskyoption2, safeoption, and all_diff_cont
  # Assumes choices are binary/logical, with 1 = risky, 0 = safe.
  #
  # Peter Sokol-Hessner
  # July 2021
  
  rho = parameters[1]
  mu = parameters[2]
  muchange = parameters[3]
  
  # Correct parameter bounds
  if(rho <= 0){
    rho = .Machine$double.eps;
  }
  
  applied_mu = exp(mu + choiceset$prev_all_diff_cont*muchange)
  
  # calculate utility of the two options
  utility_risky_option = 0.5 * choiceset$riskyoption1^rho +
    0.5 * choiceset$riskyoption2^rho;
  utility_safe_option = choiceset$safeoption^rho;
  
  # normalize values using this term
  div <- 30^rho; # decorrelates rho & mu
  
  # calculate the probability of selecting the risky option
  choiceP = 1/(1+exp(-applied_mu/div*(utility_risky_option - utility_safe_option)));
  
  likelihood = choices * choiceP + (1 - choices) * (1-choiceP);
  likelihood[likelihood == 0] = .Machine$double.eps;
  # likelihood[likelihood == 0] = 0.000000000000001; # 1e-15, i.e. 14 zeros followed by a 1; USE FOR consistency with psychopy script
  
  nll <- -sum(log(likelihood));
  return(nll)
}

#### Optimization ############################################

# Set up the parallelization
n.cores <- parallel::detectCores() - 1; # Use 1 less than the full number of cores.

# Bounds and parameters
eps = .Machine$double.eps;
lower_bounds = c(eps, -30, -10); # R, M, muchange (mu & muchange are theoretically unbounded)
upper_bounds = c(2,5,10);
number_of_parameters = length(lower_bounds);

lb_constr = c(eps, 0); # R, M, muchange (mu & muchange are theoretically unbounded)
ub_constr = c(2, 80);
number_of_parameters_constr = length(lb_constr);

# Create placeholders for parameters, errors, NLL (and anything else you want)
number_of_iterations = 200; # 100 or more
estimated_parameters = array(dim = c(number_of_clean_subjects,number_of_parameters));
estimated_parameter_errors = array(dim = c(number_of_clean_subjects,number_of_parameters));
NLLs = array(dim = c(number_of_clean_subjects,1));

estimated_parameters_constr = array(dim = c(number_of_clean_subjects,number_of_parameters_constr));
estimated_parameter_errors_constr = array(dim = c(number_of_clean_subjects,number_of_parameters_constr));
NLLs_constr = array(dim = c(number_of_clean_subjects,1));

cat('Beginning Optimization\n')

my.cluster <- parallel::makeCluster(
  n.cores,
  type = "FORK"
)
doParallel::registerDoParallel(cl = my.cluster)


##### Do the Estimations ###############################
library(tictoc)
library(doParallel)
library(doRNG)
library(numDeriv)
tic()
for (subj in 1:number_of_clean_subjects){
  # for (subj in 1){
  subj_id = keep_participants[subj];
  cat(sprintf('%i of %i\n',subj_id, number_of_clean_subjects))
  
  tmpdata = clean_data_dm[(clean_data_dm$subjectnumber == subj_id) &
                            is.finite(clean_data_dm$choice) & 
                            is.finite(clean_data_dm$prev_all_diff_cont),]; # defines this person's data
  
  choiceset = as.data.frame(cbind(tmpdata$riskyopt1, tmpdata$riskyopt2, tmpdata$safe, tmpdata$prev_all_diff_cont));
  colnames(choiceset) <- c('riskyoption1', 'riskyoption2', 'safeoption', 'prev_all_diff_cont');
  
  # Mu Change Model
  temp_parameters = array(dim = c(number_of_iterations,number_of_parameters));
  temp_hessians = array(dim = c(number_of_iterations,number_of_parameters,number_of_parameters));
  temp_NLLs = array(dim = c(number_of_iterations,1));
  
  tmpoutput <- foreach(iter=1:number_of_iterations, .combine=rbind) %dorng% {
    # you need to randomize the initial so you can end up at different points
    initial_values = runif(number_of_parameters, min = lower_bounds, max = upper_bounds)
    
    # The estimation itself
    output <- optim(initial_values, negLLprospect_cge_muPrevDiff,
                    choiceset = choiceset,
                    choices = tmpdata$choice,
                    lower = lower_bounds,
                    upper = upper_bounds,
                    method = "L-BFGS-B",
                    hessian = T)
    
    c(output$par,output$value); # the things (parameter values & NLL) to save/combine across parallel estimations
  }
  
  tmpoutput = as.array(tmpoutput[,]) # Remove all the extra RNG info
  dimnames(tmpoutput) <- NULL
  
  # Store the output we need access to later
  best_ind = which.min(tmpoutput[,number_of_parameters+1]); # the index of that NLL
  
  # Compare output; select the best one
  estimated_parameters[subj,] = tmpoutput[best_ind,1:number_of_parameters] # the parameters
  NLLs[subj] = tmpoutput[best_ind,number_of_parameters+1]; # the best NLL for this person
  
  tmp_hess = hessian(negLLprospect_cge_muPrevDiff,estimated_parameters[subj,], 
                     choiceset = choiceset,
                     choices = tmpdata$choice)
  
  estimated_parameter_errors[subj,] = sqrt(diag(solve(tmp_hess))); # the SEs
  
  
  # Standard (Constrained, no mu change) Model
  
  temp_parameters = array(dim = c(number_of_iterations,number_of_parameters_constr));
  temp_hessians = array(dim = c(number_of_iterations,number_of_parameters,number_of_parameters_constr));
  temp_NLLs = array(dim = c(number_of_iterations,1));
  
  tmpoutput <- foreach(iter=1:number_of_iterations, .combine=rbind) %dorng% {
    # you need to randomize the initial so you can end up at different points
    initial_values = runif(number_of_parameters_constr, min = lb_constr, max = ub_constr)
    
    # The estimation itself
    output <- optim(initial_values, negLLprospect_cge,
                    choiceset = choiceset,
                    choices = tmpdata$choice,
                    lower = lb_constr,
                    upper = ub_constr,
                    method = "L-BFGS-B",
                    hessian = T)
    
    c(output$par,output$value); # the things (parameter values & NLL) to save/combine across parallel estimations
  }
  
  tmpoutput = as.array(tmpoutput[,]) # Remove all the extra RNG info
  dimnames(tmpoutput) <- NULL
  
  # Store the output we need access to later
  best_ind = which.min(tmpoutput[,number_of_parameters_constr+1]); # the index of that NLL
  
  # Compare output; select the best one
  estimated_parameters_constr[subj,] = tmpoutput[best_ind,1:number_of_parameters_constr] # the parameters
  NLLs_constr[subj] = tmpoutput[best_ind,number_of_parameters_constr+1]; # the best NLL for this person
  
  tmp_hess = hessian(negLLprospect_cge,estimated_parameters_constr[subj,], 
                     choiceset = choiceset,
                     choices = tmpdata$choice)
  
  estimated_parameter_errors_constr[subj,] = sqrt(diag(solve(tmp_hess))); # the SEs
}
stopCluster(my.cluster)
toc()

##### Process the Results ###############################
ttest_result = t.test(estimated_parameters[,3])
cat(sprintf('Results of one-sample t-test: t = %.2f (df = %i), p = %.4f\n',
            ttest_result$statistic, ttest_result$parameter, ttest_result$p.value))
# Value is *slightly* positive, M = 0.17, SD = 0.54; t(84) = 2.9, p = 0.005
hist(estimated_parameters[,3], xlab = 'Mu Change parameter value')

lrt = 2*(NLLs_constr - NLLs)
df = 1
lrtp = 1 - pchisq(lrt, df)

cat(sprintf('Out of %i mu-change parameter values:\n - %i are not different from 0\n - %i are different & negative\n - %i are different & positive.\n', 
            length(lrtp), sum(lrtp >= .05), 
            sum(((lrtp < .05) & (estimated_parameters[,3] < 0))), 
            sum(((lrtp < .05) & (estimated_parameters[,3] > 0)))))

cat(sprintf('Likelihood Ratio Test: Test statistic = %.1f, p = %.5f.\n', lrt, lrtp))

cor.test(estimated_parameters[,3], clean_data_complexspan$compositeSpanScore)
cor.test(estimated_parameters[,3], clean_data_complexspan$compositeSpanScore, method = 'spearman')
plot(estimated_parameters[,3], clean_data_complexspan$compositeSpanScore, 
     xlab = 'Mu Change parameter', ylab = 'WMC')
# No significant correlation (either spearman's rho or pearson's r)

# Had predicted...
# - negative muchange parameter value
# - muchange parameter value correlated with WMC (or diff. by group),
#   such that low WMC people had a negative muchange parameter value
#   indicating reduced choice consistency after difficult trials. 
#
# Find instead:
# - weak average positive value (significant at a one-sample t-test, but
#   not significant on a per-person basis at any meaningful level)
# - No relationship between WMC and muchange.


# TAKEAWAY:
# There are no strong signs in peoples' choices that after difficult trials, choice likelihood
# is relatively lower for low capacity folks than for high capacity folks. That *would* have
# been consistent with 'giving up'. The signs are in the expected direction, but no stats
# reach significance.
#
# Note that no stats are instead consistent with *facilitation* after difficult trials (which
# might predict more internally-consistent choices after difficult trials).

# This analysis likely suffers from too much breaking-into-parts (and any others going down this
# path would suffer more, e.g. is the likelihood lower, on difficult trials only, after difficult
# trials than after easy trials?)

# This analysis is ALSO fundamentally limited by our design - easy and difficult trials are
# DESIGNED with a particular choice likelihood. That's the literal definition of what makes an
# easy trial easy etc. In particular, difficult trials, where we might expect to see the most
# movement, are designed with likelihoods *right in the middle*, so biases to act in a
# particular way on those trials (e.g. always gamble; always safe) might be impossible to see
# as *all actions are equally (un)likely*. If anything, this counterintuitively suggests that
# EASY trials would be the place to observe unusual behavior, but we may have again hurt
# ourselves as easy trials are SO easy as to be obvious, and not requiring much cognitive
# work at all to decide about. In short, the way to have observed this might have been with
# trials that were in-between easy and difficult.






