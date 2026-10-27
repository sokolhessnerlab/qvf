


# QVF Study B (Model Predicted Intermediate Difficulty) Data Processing Script
###############################################################################

rm(list=ls()); # Clear the workspace

# STEP 1: SET YOUR WORKING DIRECTORY #####

## setting the working directory on kebab (and some others) -----
setwd('/Users/shlab/Documents/GitHub/qvf/qvf_StudyB')
setwd('/Users/sokolhessner/Documents/gitrepos/qvf/qvf_StudyB')


# STEP 2: RUN FROM SHLAB DRIVE #####

## using the config.yml to more flexibly access the shlab drive across multiple devices -----
config = config::get()
setwd(config$path$data$raw)

## list all data files -----
rdmfn = dir(pattern = glob2rx('qvfB*BGP*.csv'),full.names = T, recursive = T)
ospfn = dir(pattern = glob2rx('qvfB*OSpan*bothReal*.csv'), full.names = T, recursive = T)
sspfn = dir(pattern = glob2rx('qvfB*SymSpan*bothReal*.csv'), full.names = T, recursive = T)
#qualfn = dir(pattern = glob2rx('Qualtrics*.csv'), full.names = T, recursive = T)
#psqfn = dir(pattern = glob2rx('EDI_Post_Study_Questionnaire_Quant_HandEnteredData_*.csv'), full.names = T, recursive = T)

## identify the number of participants from the file listing -----
subjectIDs = c()
for (rfn in 1:length(rdmfn)){
  subjectIDs[rfn] = as.numeric(substr(rdmfn[rfn],7,9)) # qvfB000_ -> 7th & 9th character???
}

number_of_subjects = length(subjectIDs)

subjects_to_replace_choiceP = c(3, 4, 5, 6, 7, 8) # had a data storage error resulting in duplicate choiceP entries in the data file

# STEP 3: SAVE OUT DATA #####

## qualtrics survey processing -----

### demographics -----

### intolerance of uncertainty scale (IUS-12; Carleton et al., 2007) -----

### subjective numeracy scale (SNS; Fagerline et al., 2007) -----

## bespoke gambling paradigm (BGP; Monteza et al., in process) ----

cat('Processing decision-making and working memory data...')

### create a vector to hold decision-making (BGP) and working memory data (OSpan & SymSpan) -----
column_names_dm = c(
  'trialnumber',
  'subjectnumber',
  'riskyopt1',
  'riskyopt2',
  'safe',
  'choice',
  'reactiontime',
  'outcome',
  'ischecktrial',
  'static0dynamic1', # type_e0i1d2reject0accept1dynamicblocknum
  'e0i1d2', 
  'easyP1difficultN1', # intermediate = 0 here, like static trials
  'choiceP',
  'bestRho',
  'bestMu',
  'ospan',
  'symspan',
  'complexspan'
)

data_dm = array(data = NA, dim = c(0, length(column_names_dm)));
colnames(data_dm) <- column_names_dm

### set up variables to hold working memory data -----
number_of_ospan_trials_per_person = 25
number_of_sspan_trials_per_person = 14


ospanExclude = c()
sspanExclude = c()
complexSpanExclude = as.data.frame(matrix(data=0, 
                                          nrow = number_of_subjects, 
                                          ncol=3, 
                                          dimnames=list(c(NULL), c("subjectnumber", "ospanExclude", "symspanExclude"))));
complexSpanExclude$subjectnumber = 1:number_of_subjects

complexSpanScores = as.data.frame(matrix(data=NA, 
                                         nrow = number_of_subjects, 
                                         ncol=4, 
                                         dimnames=list(c(NULL), c("subjectnumber", "ospanScore", "symspanScore", "compositeSpanScore"))));
complexSpanScores$subjectnumber= 1:number_of_subjects

### subject data file loop -----
for(s in 1:number_of_subjects){
  
  sub_id = subjectIDs[s]
  
  # BGP Data -----

  # load decision-making data
  tmpdata = read.csv(rdmfn[s])
  
  ### store some basic information about the size of the decision-making task -----
  dm_index_static = is.finite(tmpdata$checkTrial)
  dm_index_dynamic = is.finite(tmpdata$choiceDifficulty)
  dm_index_realchoices = dm_index_static | dm_index_dynamic
  
  num_static_trials = sum(dm_index_static) # should always be 50
  num_dynamic_trials = sum(dm_index_dynamic) # is 240 for subj. 1, 244 for everyone else
  number_of_dm_trials_per_person = num_static_trials + num_dynamic_trials # 290 for subject 1, 294 for everyone else
  
  # decision-making data array
  dm_data_to_add = array(data = NA, dim = c(number_of_dm_trials_per_person,length(column_names_dm)))

  tmp_trialnum = 1:number_of_dm_trials_per_person

  dm_data_to_add[,1] = tmp_trialnum # trial number
  dm_data_to_add[,2] = s # subject number

  tmp_riskyopt1 = c(tmpdata$gainValue[dm_index_static],
                    tmpdata$gainValue[dm_index_dynamic])
  tmp_riskyopt2 = c(tmpdata$lossValue[dm_index_static],
                    tmpdata$lossValue[dm_index_dynamic])
  tmp_safe = c(tmpdata$safeValue[dm_index_static],
               tmpdata$safeValue[dm_index_dynamic])

  dm_data_to_add[,3:5] = cbind(tmp_riskyopt1,tmp_riskyopt2,tmp_safe) # dollar amounts

  dm_data_to_add[,6] = c(tmpdata$choiceMade[dm_index_static],
                         tmpdata$choiceMade[dm_index_dynamic]) # choices

  dm_data_to_add[,7] = c(tmpdata$choiceEnd[dm_index_static]-tmpdata$choiceStart[dm_index_static],
                         tmpdata$choiceEnd[dm_index_dynamic]-tmpdata$choiceStart[dm_index_dynamic]) # RTs

  dm_data_to_add[,8] = c(tmpdata$outcomeValue[dm_index_static],
                         tmpdata$outcomeValue[dm_index_dynamic]) # outcomes

  dm_data_to_add[,9] = c(tmpdata$checkTrial[dm_index_static],
                         array(data = 0, dim = c(1,num_dynamic_trials))) # is check trial

  dm_data_to_add[,10] = c(array(data = 0, dim = c(1,num_static_trials)),
                         array(data = 1, dim = c(1,num_dynamic_trials))) # static 0, dynamic 1

  dm_data_to_add[,11] = c(tmpdata$choiceDifficulty[dm_index_realchoices])  # e0i1d2
  
  dm_data_to_add[,12] = 1*(tmpdata$choiceDifficulty[dm_index_realchoices] == 0) - 
    1*(tmpdata$choiceDifficulty[dm_index_realchoices] == 2)# easy +1, difficult -1
  dm_data_to_add[is.na(dm_data_to_add[,12]),12] = 0 # give all the static trials a default value of 0

  dm_data_to_add[,13] = c(array(data = NA, dim = c(1,num_static_trials)),
                          tmpdata$choiceProbability[dm_index_dynamic]) # choice probability on easy/diff dynamic trials

  dm_data_to_add[,14] = tmpdata$bestRho[is.finite(tmpdata$bestRho)]
  dm_data_to_add[,15] = tmpdata$bestMu[is.finite(tmpdata$bestMu)]
  
  # Loop to fix `choiceP` variable storage error
  if(sub_id %in% subjects_to_replace_choiceP){ # if you are one of the subjects with this issue
    bestRhoInd = unique(dm_data_to_add[,14])
    bestMuInd = unique(dm_data_to_add[,15])
    
    choiceset_fn = sprintf('qvf_bespoke_choiceset_rhoInd%03i_muInd%03i.csv',bestRhoInd, bestMuInd)
    choiceset_relpath = sprintf('../../stimuli/bespoke_choicesets/%s',choiceset_fn)
    
    orig_stimuli = read.csv(choiceset_relpath)
    
    for(t in 1:number_of_dm_trials_per_person){ # go through each trial one at a time
      if(dm_data_to_add[t,10] == 1) { # if this is a dynamic trial... 
        rowInd = which((dm_data_to_add[t,3] == orig_stimuli$riskyoption1) & #... identify which line in the orig_stimuli data matches the dollar values...
                         (dm_data_to_add[t,4] == orig_stimuli$riskyoption2) & 
                         (dm_data_to_add[t,5] == orig_stimuli$safeoption))
        dm_data_to_add[t,13] = orig_stimuli$choiceP[rowInd] # ... and pull out that choiceP value and store it here. 
      }
    }
  }
  
  ### OSPAN DATA ###
  
  osp_ind = which(grepl(sprintf('qvfB%03i',sub_id),ospfn))
  if (length(osp_ind) == 1){
    
    ospantmpdata = read.csv(ospfn[osp_ind]);
    ospantmpdata$subid = as.integer(substr(ospfn[osp_ind],6,8));
    
    
    if (any(ospantmpdata$percentCorrectMath[is.finite(ospantmpdata$percentCorrectMath)]<85)){
      ospanExclude = c(ospanExclude,ospantmpdata$subid[1]);
      complexSpanExclude$ospanExclude[s] = 1;
    } else {
      correctIndospan = which(ospantmpdata$correctCount == ospantmpdata$setSize)
      complexSpanScores$ospanScore[s] = sum(ospantmpdata$correctCount[correctIndospan])/number_of_ospan_trials_per_person;
    }
  } else {
    ospanExclude = c(ospanExclude,sub_id);
    complexSpanExclude$ospanExclude[s] = 1;
  }
  
  ### SYMSPAN DATA ###

  ssp_ind = which(grepl(sprintf('qvfB%03i',sub_id),sspfn))
  
  if (length(ssp_ind) == 1){
    
    sspantmpdata = read.csv(sspfn[ssp_ind]);
    sspantmpdata$subid = as.integer(substr(sspfn[ssp_ind],6,8));
    
    if (any(sspantmpdata$percentCorrectSym[is.finite(sspantmpdata$percentCorrectSym)]<85)){
      sspanExclude = c(sspanExclude,sspantmpdata$subid[1]);
      complexSpanExclude$symspanExclude[s] = 1;
    } else {
      correctIndsymspan = which(sspantmpdata$squareCorrectCount == sspantmpdata$setSize)
      complexSpanScores$symspanScore[s] = sum(sspantmpdata$squareCorrectCount[correctIndsymspan])/number_of_sspan_trials_per_person;
    }
  } else {
    sspanExclude = c(sspanExclude,sub_id);
    complexSpanExclude$symspanExclude[s] = 1;
  }
  
  
  ### COMPOSITE SPAN ###
  if ((complexSpanExclude$ospanExclude[s] == 0) & (complexSpanExclude$symspanExclude[s] == 0)){ # if both scores available
    complexSpanScores$compositeSpanScore[s] = mean(c((complexSpanScores$ospanScore[s]),(complexSpanScores$symspanScore[s]))); # average the two scores
  } else if ((complexSpanExclude$ospanExclude[s] == 1) & (complexSpanExclude$symspanExclude[s] == 0)){ # if only SymSpan
    complexSpanScores$compositeSpanScore[s] = complexSpanScores$symspanScore[s];
  } else if ((complexSpanExclude$ospanExclude[s] == 0) & (complexSpanExclude$symspanExclude[s] == 1)){ # if only OSpan
    complexSpanScores$compositeSpanScore[s] = complexSpanScores$ospanScore[s];
  }; # ... else, leave it NA

  # Span scores
  dm_data_to_add[,16] = complexSpanScores$ospanScore[s];
  dm_data_to_add[,17] = complexSpanScores$symspanScore[s];
  dm_data_to_add[,18] = complexSpanScores$compositeSpanScore[s];

#   # Survey data + Heartbeat Detection Data
#   survey_data_to_add = drop(survey_data[s,2:ncol(survey_data)]);
#   for (n in 18:length(column_names_dm)){
#     dm_data_to_add[,n] = as.numeric(survey_data_to_add[n-17])
#   }

  # Add this person's DM data to the total DM data.
  data_dm = rbind(data_dm,dm_data_to_add);

}

data_dm = as.data.frame(data_dm) # make it a data frame so it plays nice
colnames(data_dm) = column_names_dm

cat('Done.\n')

# save out CSVs with the clean, compiled data!
cat('Saving out data... ')
setwd(config$path$data$processed);

write.csv(data_dm, file=sprintf('qvfB_processed_decisionmaking_data_%s.csv',format(Sys.Date(), format="%Y%m%d")),
          row.names = F);
write.csv(complexSpanScores, file=sprintf('qvfB_processed_complexspan_data_%s.csv',format(Sys.Date(), format="%Y%m%d")),
          row.names = F);
# write.csv(survey_data, file=sprintf('qvfB_processed_survey_data_%s.csv',format(Sys.Date(), format="%Y%m%d")),
#           row.names = F);

cat('Done.\n\nAll data has been processed.\n\n')
# all done!








