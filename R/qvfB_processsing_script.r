


# QVF Study B (Model Predicted Intermediate Difficulty) Data Processing Script
###############################################################################

# STEP 1: SET YOUR WORKING DIRECTORY
setwd('/Users/shlab/Documents/GitHub/qvf/qvf_StudyB')


# Connect to the SHLab drive using the config.yml script to begin processing participant files
setwd(config$path$data$raw)



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

