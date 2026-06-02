
# -------------------------------------------------------------------------------------------------------
# QUITTING VS. FACILIATION (QVF) STUDY: BESPOKE GAMBLING PARADIGM (BGP) - DISCRETE RANGES FULL CONTINUUM
# -------------------------------------------------------------------------------------------------------

"""

Bespoke Gambling Paradigm (BGP) - Discrete Ranges Full Continuum

Author: J. Von R. Monteza (2026.05.22)

Overview:
    
    This BGP design addresses the limitations of previous BGP iterations in CGT (Anna Rini), CGE (Von Monteza), and EDI (Sophie Forcier). 
    In previous iterations of the BGP, choice difficulty was binary: easy vs. difficult. This allowed us to clearly compare the effects of 
    easy and difficult choices on cognitive effort within and across trials. HOWEVER, this design limited our ability to examine whether or
    not previous choice difficulty was associated with (mal)adaptive choice behavior on the current trial. Easy choices, regardless of context, were
    designed to be "completely" predictable. The probability of choosing one option over the other is extremely high (pGamble >= .85). The "best" option is 
    always obvious. Difficult choices, regardless of context, were designed to be "completely" unpredictable. The probability of choosing one option over 
    the other is extremely low (pGamble ≈ .50).The "best" option is never obvious. In other words, although we could examine the effects of 
    previous trial difficulty on cognitive effort on the current trial, we could not examine the effects of previous trial difficulty on choice behavior.

    To address this limitation of previous BGP iterations, we need choices that are predictable but not obvious. Something in between easy and difficult. 
    In this BGP iteration (Discrete Ranges Full Continuum), we have included intermediate difficult choices (choices that are moderately difficult). 
    Intermediate difficult choices are partially predictable. The probability of choosing one option over the other is modest (.65 < pGamble > .75). 
    The "best" option is slightly obvious. In other words, choice behavior is more flexible, compared to easy and difficulty choices, and 
    can thus be affected by previous choice difficulty. If previous choice difficulty is associated with disengagement: Quitting (due to cognitive fatigue, 
    budgeting, etc.), then participants will both be faster on the current trial and make "bad" (the unpredicted "better" choice) and possibly more
    variable choices. If previous difficulty is associated with adaptation: Faciliation (cognitive "gearing up"), then participants will both be faster on
    the current trial and make "good" (the predicted "better" choice) and possible more consistent choices. 
    
    Note* It is unclear how fast decision times reflect cognitive effort. On one hand, it may reflect less cognitive effort, and on the other, it
    may reflect more cognitive effort. The types of choices must also be considered: "bad"/inaccurate or "good"/accurate. In "bad"/inaccurate choices,
    faster decision times most likely reflect less cognitive effort. In "good"/accurate choices, faster decision times may reflect less cognitive effort due to
    learning over time (getting better at the task), or it may reflect an intense but short duration of cognitive effort to determine the "better"/accurate choice.

"""

##### TIME TO SET THINGS UP! #########################################################

# ------------------------------------------------------------------------------------
# PACKAGE & MODULE SETUP #
# ------------------------------------------------------------------------------------

# directory and filing
import os
import sys
import time

# psychopy 
from psychopy import visual, core, event, monitors

# task related
import random
import pandas as pd
import numpy as np 
import math

# ------------------------------------------------------------------------------------
# PARTICIPANT/SESSION SETUP #
# ------------------------------------------------------------------------------------

# set the participant ID
subID = '999'

# name of the study
studyName = 'qvf'

# name of the task
taskName = 'BGP'

# set if it is a real or test run
isReal = 0 # 1 = yes, a real run vs. 0 = no, a test run

if isReal == 0: 
    practiceSet = 2
    staticSet = 6 # doesn't take less than 6? itiStatic splitting breaks?
    dynamicSet = 6
elif isReal == 1:
    practiceSet = 5
    staticSet = 50 # 40 real trials & 10 check trials
    dynamicSet = 240 # 60 easy trials & 60 difficult trials

# ------------------------------------------------------------------------------------
# DIRECTORY SETUP #
# ------------------------------------------------------------------------------------

# create the various directories
BGP_taskFolder = os.path.dirname(os.path.abspath(__file__)) # BGP Task Folder - to run the BGP (current directory)
print (BGP_taskFolder)
qvfTasks_folder = os.path.dirname(BGP_taskFolder) # QVF Tasks Folder
print (qvfTasks_folder)
data_folder = os.path.join(qvfTasks_folder, "data") # QVF Data Folder
print (data_folder)
BGP_choiceBehavior_dataFolder = os.path.join(data_folder, "BGP_choiceBehaviorData") # BGP Choice Behavior Data Folder
print (BGP_choiceBehavior_dataFolder)
BGP_eyeTracking_dataFolder = os.path.join(data_folder, "BGP_eyeTrackingData") # BGP Eye-Tracking Data Folder
print (BGP_eyeTracking_dataFolder)

# set the current directory to the BGP folder to run the BGP
os.chdir(BGP_taskFolder)

# ------------------------------------------------------------------------------------
# GENERAL OBJECT FORMATTING SETUP #
# ------------------------------------------------------------------------------------

# luminance equated colors
c1 = [0.5216,0.5216,0.5216] # gray color for background, choice option text, and risky option line
c2 = [-0.0667,0.6392,1] # blue color for choice option circles, OR text, and V-Left and N-Left text

#col1
#col2 clr, colr

# monitor screen
#screenSize = [1280, 1024] # How do I make it so that it gets the right screen size no matter what device?

# font stle
a = 'Arial'

# font size
instructionsH = .05
choiceTextH = .05
choiceValuesH = .1
fixCrossH = .05
noRespH = .08
ocH = .1 # oc = Outcome

# shape size
choiceSize = [.5,.5]
riskSize = [.5,.01]
hideSize = [.6,.3] # may not need this

# locations for choice option shapes and text
center = [0, 0]
leftLoc = [-.35,0]
rightLoc = [.35,0]
vLoc = [-.35,-.35]
nLoc = [.35,-.35]

# choice trial timing
choiceWin = 4 # decision window

isi = 1 # interstimulus interval window

oc = 1 # outcome window

itiStatic = [] # intertrial interval window
itiDynamic = []
itiStatic = [3, 3.5] * (staticSet // 2) # jittered between 3 and 3.5 seconds for all trials
itiDynamic = [3, 3.5] * (dynamicSet // 2) # jittered between 3 and 3.5 seconds for all trials
# randomize iti time for both the trials in the static and dynamic choice sets
def shuffle(array):
    currentIndex = len(array)
    while currentIndex != 0:
        randomIndex = random.randint(0, currentIndex - 1)
        currentIndex -= 1
        array[currentIndex], array[randomIndex] = array[randomIndex], array[currentIndex]
shuffle(itiStatic)
shuffle(itiDynamic)

# ------------------------------------------------------------------------------------
# COUNT TRIALS #
# ------------------------------------------------------------------------------------

isCount = 0 # 1 = yes, count trials vs. 0 = no, don't count trials

if isCount == 0:
    countLoc = [0, 7]
    countColor = c1
elif isCount == 1:
    countLoc = [0, -.35]
    countColor = c2

# ------------------------------------------------------------------------------------
# WINDOW SETUP #
# ------------------------------------------------------------------------------------

win = visual.Window(
    size = [1280, 1024], 
    units = 'height', 
    monitor ='testMonitor', 
    fullscr = True, 
    color = c1
) # Don't quite now yet the exact translation from height to pix in terms of size and location


# ------------------------------------------------------------------------------------
# EYE-TRACKING DATA COLLECTION SETUP #
# ------------------------------------------------------------------------------------

# set to collect eye-tracking data
doET = 0 # 1 = yes, do eye-tracking vs. 0 = no, only do behavioral

if doET:
    
    #
    ##
    ### IMPORTING EYE-TRACKING MODULES ###
    import pylink
    from PIL import Image  # for preparing the Host backdrop image
    from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy # eye-tracking py module located in ediTasks
    import subprocess # for converting edf to asc 
    
    #
    ##
    ### SETTING UP EYE-TRACKER ###
    
    # Step 1: Connect to the EyeLink Host PC # The Host IP address, by default, is "100.1.1.1".
    et = pylink.EyeLink("100.1.1.1")
    # Step 2: Open an EDF data file on the Host PC
    edf_fname = f"{studyName}{subID}"
    et.openDataFile(edf_fname + '.edf')
    # Step 3: Configure the tracker
    # Put the tracker in offline mode before we change tracking parameters
    et.setOfflineMode()
    # Step 4: Setting parameters
    # File and Link data control
    file_sample_flags = 'GAZE,GAZERES,AREA,BUTTON,STATUS,INPUT'
    et.sendCommand("file_sample_data = %s" % file_sample_flags)
    # Optional tracking parameters
    # Sample rate, 250, 500, 1000, or 2000, check your tracker specification
    et.sendCommand("sample_rate 1000")
    # Choose a calibration type, H3, HV3, HV5, HV13 (HV = horizontal/vertical),
    et.sendCommand("calibration_type = HV9")

if doET:
    #
    ##
    ### CALIBRATION AND VALIDATION SETUP ###
    # get the native screen resolution used by PsychoPy
    scn_width, scn_height = win.size

    # Pass the display pixel coordinates (left, top, right, bottom) to the tracker
    # see the EyeLink Installation Guide, "Customizing Screen Settings"
    et_coords = "screen_pixel_coords = 0 0 %d %d" % (scn_width - 1, scn_height - 1)
    et.sendCommand(et_coords)

    # Write a DISPLAY_COORDS message to the EDF file
    # Data Viewer needs this piece of info for proper visualization, see Data
    # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
    dv_coords = "DISPLAY_COORDS  0 0 %d %d" % (scn_width - 1, scn_height - 1)
    et.sendMessage(dv_coords)

if doET:
    
    # Configure a graphics environment (genv) for tracker calibration
    genv = EyeLinkCoreGraphicsPsychoPy(et, win)
    print(genv)  # print out the version number of the CoreGraphics library
    # Set up the calibration target # genv.setTargetType to "circle", "picture", "movie", or "spiral", e.g.,
    genv.setTargetType('circle')
    # Use a picture as the calibration target
    #genv.setTargetType('picture') ### The picture doesn't show the second time around
    #genv.setPictureTarget(os.path.join('images', 'fixTarget.bmp'))
    # Beeps to play during calibration, validation and drift correction # parameters: target, good, error
    # Each parameter could be ''--default sound, 'off'--no sound, or a wav file
    genv.setCalibrationSounds('', '', '')
    # Request Pylink to use the PsychoPy window we opened above for calibration
    pylink.openGraphicsEx(genv)
    et.doTrackerSetup()

# ------------------------------------------------------------------------------------
# STIMULI SETUP #
# ------------------------------------------------------------------------------------

# ~ Instructions ~

# general instructions
rdmStartTxt = visual.TextStim(
    win,
    text = 'As discussed in the instructions, you will choose between a gamble and a guaranteed alternative choice option.\n\nPress "V" to select the option on the left OR "N" to select the option on the right.\n\n Press "Enter/Return" to move on the next screen.',
    font = a,
    height = instructionsH,
    pos = center,
    color = c2
)

# practice instructions
pracStartTxt = visual.TextStim(
    win,
    text = 'There will now be 5 practice trials.\n\nThe structure in the practice round is identical to what you will encounter in the real rounds. The goal of the practice round is to practice timing of decision-making within the four (4) second response window.\n\nWhen you are ready to begin the practice, press "V" or "N" to begin.',
    font = a,
    height = instructionsH,
    pos = center,
    color = c2
)

# static choice set instructions
statStartTxt = visual.TextStim(
    win,
    text = 'Practice complete.\n\nKeep in mind that responding quickly in this task will not speed up the task. Please take enough time to view and consider each choice option before you make a choice within the four (4) second response window.\n\nWhen you are ready to start ROUND 1 of the task, press "V" or "N".',
    font = a,
    height = instructionsH,
    pos = center,
    color = c2
)

# fitting process instructions
fittingStartTxt = visual.TextStim(
    win,
    text = 'ROUND 1 of the gambling task is complete!\n\nSetting up for the last round of the gambling task.\n\nPlease wait...',
    font = a,
    height = instructionsH,
    pos = center,
    color = c2
)

# dynamic choice set instructions
dynaStartTxt = visual.TextStim(
    win,
    text = 'Keep in mind that responding quickly in this task will not speed up the task. Please take enough time to view and consider each choice option before you make a choice within the four (4) second response window.\n\nWhen you are ready to start ROUND 2 of the task, press "V" or "N".',
    font = a,
    height = instructionsH,
    pos = center,
    color = c2
)

# task closing instructions
endStartTxt = visual.TextStim(
    win,
    text = 'You have sucessfully completed the first task in this experiment!\n\nPlease take a brief 1 minute break. \n\nYou are welcome to take a longer break, but keep in mind this study should take no longer than 1 hour to complete. \n\nWhen you are ready to move on, press "enter to continue.\n',
    font = a,
    height = instructionsH,
    pos = center,
    color = c2
)

# call experimenter instructions
callStartTxt = visual.TextStim(
    win,
    text = 'You have sucessfully completed the first task in this experiment!\n\nPlease press the white call button.\n',
    font = a,
    height = instructionsH,
    pos = center,
    color = c2
)

# ~ Choice Trial Components ~

# decision window stimuli
vOpt = visual.Circle(
    win,
    size = choiceSize, 
    fillColor = c2, 
    lineColor = c2,
    pos = leftLoc,
    edges = 128
)

nOpt = visual.Circle(
    win,
    size = choiceSize, 
    fillColor = c2, 
    lineColor = c2,
    pos = rightLoc,
    edges = 128
)

riskSplit = visual.Rect(
    win, 
    size = riskSize,
    fillColor = c1, 
    lineColor = c1 
)

gainTxt = visual.TextStim(
    win, 
    font = a,
    height = choiceValuesH,
    color = c1
)

lossTxt = visual.TextStim(
    win, 
    font = a,
    height = choiceValuesH,
    color = c1
)

safeTxt = visual.TextStim(
    win, 
    font = a,
    height = choiceValuesH,
    color = c1
)

orTxt = visual.TextStim(
    win, 
    text = "OR",
    font = a,
    height = choiceTextH,
    pos = center,
    color = c2
)

vTxt = visual.TextStim(
    win, 
    text = "V - Left",
    font = a,
    height = choiceTextH,
    pos = vLoc,
    color = c2);

nTxt = visual.TextStim(
    win, 
    text = "N - Left",
    font = a,
    height = choiceTextH,
    pos = nLoc,
    color = c2);

countTrialTxt = visual.TextStim(
    win, 
    text = "",
    font = a,
    height = choiceTextH,
    pos = countLoc,
    color = countColor)

# isi and iti fixation stimuli
fixTxt = visual.TextStim(
    win, 
    text = "+",
    font = a,
    height = choiceTextH,
    pos = center,
    color = c2
)

# choice outcome stimuli
noRespTxt = visual.TextStim(
    win, 
    text = "You did not respond in time",
    font = a,
    height = ocH,
    pos = center,
    color = c2
)

ocRiskyHide = visual.Rect(
    win,
    size = hideSize, 
    fillColor = c1, 
    lineColor = c1,
)

# ------------------------------------------------------------------------------------
# QVF DATA FRAME SETUP #
# ------------------------------------------------------------------------------------

ediData = []
ediData.append(
    [
        "trialNumber", # [0] # should incrementally increase by 1
        "checkTrial", # [1] # should be "0" for no, not a check trial or "1" for yes, a check trial
        "gainValue", # [2]
        "lossValue", # [3] # should always be $0 (except if a check trial)
        "safeValue", # [4]
        "choiceProbability", # [5]
        "type_e0i1d2", # [6] # easy0difficult1
        "choiceMade", # [7] # should be "0" if safe option was chosen or "1" if the risky option was chosen
        "choiceKey", # [8]
        "outcomeValue", # [9] # if a choice was made, the value should match the value location and key response of choiceMade
        "location", # [10]
        "riskSplitLocation", # [11]
        "gainLocation", # [12]
        "lossLocation", # [13]
        "safeLocation", # [14]
        "hideGainLocation", # [15]
        "hideLossLocation", # [16]
        "instructionStart", # [17] # first point should be ground 0 for when the task starts # second point should be ground 0 for eye-tracking # last should be for closing instructions
        "instructionEnd", # [18]
        "choiceStart", # [19] # should be the same time as when the choice values and texts are shown - their Start (choices are shown)
        "choiceEnd", # [20] # should be the same time as when the choice values and texts are disappear - their End (choice is made)
        "isiStart", # [21] # should be just after or exactly at the moment of choiceEND
        "isiEnd", # [22] # should be 1 sec
        "outcomeStart", # [23] # should be just after or exactly at the moment of isiEND
        "outcomeEnd", # [24] # should be 1 sec
        "itiStart", # [25] # should be just after or exactly at the moment of outcomeEND
        "itiEnd", # [26] # should be either 3 or 3.5 sec
        "bestRho", # [27]
        "bestMu" # [28]
    ]
)

# ------------------------------------------------------------------------------------
# STIMULI SETUP #
# ------------------------------------------------------------------------------------
#
##
### CREATING TRIAL DESIGN FUNCTIONS ### (this way I don't have to repeat code since practice, static, and dynamic trial structure is the same)


#
##
### CREATING GENERAL FUNCTIONS ###

# end the task
def endTask():
    endIT = event.getKeys(keyList = ['escape']) # kb. vs. event.
    
    if 'escape' in endIT:
        win.close()
        core.quit()  
        sys.exit()

# prospect model
def choice_probability(parameters, riskyGv, riskyLv, certv):
    # Pull out parameters
    rho = parameters[0];
    mu = parameters[1];
    
    nTrials = len(riskyGv);
    
    # Calculate utility of the two options
    utility_riskygain_value = [math.pow(value, rho) for value in riskyGv];
    utility_riskyloss_value = [math.pow(value, rho) for value in riskyLv];
    utility_risky_option = [.5 * utility_riskygain_value[t] + .5 * utility_riskyloss_value[t] for t in range(nTrials)];
    utility_safe_option = [math.pow(value, rho) for value in certv]
    
    # Normalize values with div
    div = max(riskyGv)**rho;
    
    # Softmax
    p = [1/(1 + math.exp(-mu/div*(utility_risky_option[t] - utility_safe_option[t]))) for t in range(nTrials)];
    return p

# negative log likelihood (nll) of the participant's static choice set responses for the grid search
def pt_nll(parameters, riskyGv, riskyLv, certv, choices):
    choiceP = choice_probability(parameters, riskyGv, riskyLv, certv);
    
    nTrials = len(choiceP);
    
    likelihood = [choices[t]*choiceP[t] + (1-choices[t])*(1-choiceP[t]) for t in range(nTrials)];
    zeroindex = [likelihood[t] == 0 for t in range(nTrials)];
    for ind in range(nTrials):
        if zeroindex[ind]:
            likelihood[ind] = 0.000000000000001;
    
    loglikelihood = [math.log(likelihood[t]) for t in range(nTrials)];
    
    nll = -sum(loglikelihood);
    return nll

# rounding the choice option values in the choice stimuli file to be shown during the decision window
def choice_value_rounding():
    global gainRounded, lossRounded, safeRounded
    
    gainRounded = '%.2f' % round(gain, 2) # removed the $ here and moved it to the bottom - it was being saved into the data
    lossRounded = '%.0f' % round(loss, 0)
    safeRounded = '%.2f' % round(safe, 2)
    
    gainTxt.setText('$' + gainRounded)
    lossTxt.setText('$' + lossRounded)
    safeTxt.setText('$' + safeRounded)

# randomize choice option locations
def choice_location_randomizing():
    global loc, riskSplitLoc, gainTxtLoc, lossTxtLoc, safeTxtLoc, hideGainLoc, hideLossLoc
    
    loc = random.choice([1,2]) # initial randomization of the decision window stimuli

    if loc == 1:
        riskSplitLoc = [-.35,0] # risky option is on the left = v
        gainTxtLoc = [-.35,.1]
        lossTxtLoc = [-.35,-.1]
        safeTxtLoc = [.35,0]
        hideGainLoc = [-.35, .15]
        hideLossLoc = [-.35, -.15]
    else:
        riskSplitLoc = [.35,0] # risky option is on the right = n
        gainTxtLoc = [.35,.1]
        lossTxtLoc = [.35,-.1]
        safeTxtLoc = [-.35,0]
        hideGainLoc = [.35, .15]
        hideLossLoc = [.35, -.15]
    
    riskSplit.setPos(riskSplitLoc) # set the position of the decision window stimuli # do these get properly used in the decision_window_starting() function??? - seems to be
    gainTxt.setPos(gainTxtLoc)
    lossTxt.setPos(lossTxtLoc)
    safeTxt.setPos(safeTxtLoc)

# counting trials if set to count trials
def trial_counting():
    if isCount == 1:
        countTrialTxt.setText(trial)
        countTrialTxt.draw()

# drawing decision window stimuli and retrieving trial start time
def decision_window_starting():
    global choiceStart
    
    vOpt.draw() # draw choice options
    nOpt.draw()
    riskSplit.draw()
    gainTxt.draw()
    lossTxt.draw()
    safeTxt.draw()
    orTxt.draw()
    vTxt.draw()
    nTxt.draw()
    trial_counting()
    
    win.flip() # show choice options
      
    choiceStart = timer.getTime() # get time 

# recording the choice made and time
def decision_making():
    global response, choiceMade, choiceKey, outcomeValue, choiceEnd

    endTask() # end task if wanted/needed
    
    response = event.waitKeys(maxWait = choiceWin, keyList = ['v', 'n'], timeStamped = timer)

    if response is None:
        choiceMade = math.nan
        choiceKey = math.nan
        outcomeValue = math.nan
        choiceEnd = math.nan
    elif response[0][0] == 'v' or response[0][0] == 'n':
        if (loc == 1 and response[0][0] == 'v') or (loc == 2 and response[0][0] == 'n'):
            choiceMade = 1 # chose the risky option
            choiceKey = response[0][0]
            outcomeValue = random.choice([gainRounded, lossRounded]) # randomly chooses the gain or loss
        elif (loc == 1 and response[0][0] == 'n') or (loc == 2 and response[0][0] == 'v'):
            choiceMade = 0 # chose the safe option
            choiceKey = response[0][0]
            outcomeValue = safeRounded
        choiceEnd = response[0][1]

# setting up isi
def isi_waiting():
    global isiStart, isiEnd
    
    fixTxt.draw()
    win.flip()
    
    endTask() # end task if wanted/needed
    
    isiStart = timer.getTime()
    core.wait(isi)
    isiEnd = timer.getTime()

# showing outcome of the choice made
def outcome_showing():
    global outcomeStart, outcomeEnd
    
    if response is None:
        noRespTxt.draw()
        win.flip()
        endTask() # end task if wanted/needed
        outcomeStart = timer.getTime()
        core.wait(oc)
        outcomeEnd = timer.getTime()
    elif (loc == 1 and response[0][0] == 'v'):
        if outcomeValue == gainRounded: # risky option on the left was chosen and won
            ocRiskyHide.setPos(hideLossLoc)
            vOpt.draw()
            gainTxt.draw()
            ocRiskyHide.draw()
            win.flip()
            endTask() # end task if wanted/needed
            outcomeStart = timer.getTime()
            core.wait(oc)
            outcomeEnd = timer.getTime()
        elif outcomeValue == lossRounded: # risky option on the left was chosen and lost 
            ocRiskyHide.setPos(hideGainLoc)
            vOpt.draw()
            lossTxt.draw()
            ocRiskyHide.draw()
            win.flip()
            endTask() # end task if wanted/needed
            outcomeStart = timer.getTime()
            core.wait(oc)
            outcomeEnd = timer.getTime()
    elif (loc == 2 and response[0][0] == 'n'):
        if outcomeValue == gainRounded: # risky option on the right was chosen and won
            ocRiskyHide.setPos(hideLossLoc)
            nOpt.draw()
            gainTxt.draw()
            ocRiskyHide.draw()
            win.flip()
            endTask() # end task if wanted/needed
            outcomeStart = timer.getTime()
            core.wait(oc)
            outcomeEnd = timer.getTime()
        elif outcomeValue == lossRounded: # risky option on the right was chosen and lost
            ocRiskyHide.setPos(hideGainLoc)
            nOpt.draw()
            lossTxt.draw()
            ocRiskyHide.draw()
            win.flip()
            endTask() # end task if wanted/needed
            outcomeStart = timer.getTime()
            core.wait(oc)
            outcomeEnd = timer.getTime()
    elif (loc == 1 and response[0][0] == 'n') and outcomeValue == safeRounded: # safe option on the right was chosen
        nOpt.draw()
        safeTxt.draw()
        win.flip()
        endTask() # end task if wanted/needed
        outcomeStart = timer.getTime()
        core.wait(oc)
        outcomeEnd = timer.getTime()
    elif (loc == 2 and response[0][0] == 'v') and outcomeValue == safeRounded: # safe option on the left was chosen
        vOpt.draw()
        safeTxt.draw()
        win.flip()
        endTask() # end task if wanted/needed
        outcomeStart = timer.getTime()
        core.wait(oc)
        outcomeEnd = timer.getTime()
        
# setting up iti
def iti_waiting(): # each of the choice set's iti's are done differently - doesn't clearly work as well to do itiEnd
    fixTxt.draw()
    win.flip()
    
    endTask() # end task if wanted/needed


#
##
### CREATING DATA APPENDING FUNCTIONS ###

# append a new row of empty data ("") into the data structure that matches the length of the columns 
def empty_data_appending():
    ediData.append([""] * len(ediData[0]))

# index the last row of the data structure for future appending (e.g., ediData[data_appending_index][<whatever column>])
def data_appending_index():
    return len(ediData) - 1


##### TIME TO START THE TASK! ########################################################

# ------------------------------------------------------------------------------------
# TIMER SETUP #
# ------------------------------------------------------------------------------------

timer = core.Clock()

# ------------------------------------------------------------------------------------
# GENERAL INSTRUCTIONS START #
# ------------------------------------------------------------------------------------

rdmStartTxt.draw()
win.flip()
genInstructionsStart = timer.getTime()
response = event.waitKeys(keyList = ['return'], timeStamped = timer)
genInstructionsEnd = response[0][1]
endTask()
empty_data_appending()
ediData[data_appending_index()][17:19] = [genInstructionsStart, genInstructionsEnd]

# ------------------------------------------------------------------------------------
# PRACTICE CHOICE SET #
# ------------------------------------------------------------------------------------

# practice choice set instructions
pracStartTxt.draw()
win.flip()
if doET:
    et.sendMessage('before practice instruction start')
pracInstructionsStart = timer.getTime()
if doET:
    et.sendMessage('after practice instruction start')
if doET: # if doing eye-tracking, then start recording
    # put tracker in idle/offline mode before recording
    et.setOfflineMode()
    # start recording events
    et.startRecording(1, 0, 0, 0)
    # allocate some time for the tracker to cache some samples
    et.sendMessage('pre 100 pause')
    pylink.pumpDelay(100)
    # send message that recording has started
    et.sendMessage('ediRDM Pupillometry Recording Started - Practice Instructions Shown')
endTask()
response = event.waitKeys(keyList = ['v', 'n'], timeStamped = timer)
pracInstructionsEnd = response[0][1]
empty_data_appending()
ediData[data_appending_index()][17:19] = [pracInstructionsStart, pracInstructionsEnd]

# load task stimuli file 
practiceDF = pd.read_excel("BGP_practiceTrials.xlsx") # sequentially presented

# practice choice set task
for p in range(practiceSet):

    # Need to call in the practice file for the gain, loss, and safe text
    gain = practiceDF.riskyGain[p]
    loss = practiceDF.riskyLoss[p]
    safe = practiceDF.alternative[p]

    # Adjusting Trial Start - Python starts at 0: This makes trials start at 1
    trial = p + 1 

    # Round Choice Option Monetary Values to be Shown
    choice_value_rounding()

    # Randomize Choice Option Locations
    choice_location_randomizing()

    # Start of Trial
    decision_window_starting()

    # Choice Made
    decision_making()
    
    # ISI
    isi_waiting() # technically in the practice file, but all isi = 1, so this was easier
    
    # Choice Outcome
    outcome_showing()
    
    # ITI
    iti = practiceDF.iti[p] # iti set in file to each choice option combination
    
    iti_waiting()
    
    itiStart = timer.getTime()
    core.wait(iti)
    itiEnd = timer.getTime()
    
    # Saving Data
    empty_data_appending()
    ediData[data_appending_index()][0] = trial 
    ediData[data_appending_index()][2:5] = [gainRounded, lossRounded, safeRounded] 
    ediData[data_appending_index()][7:17] = [choiceMade, choiceKey, outcomeValue, 
                                          loc, riskSplitLoc, gainTxtLoc, lossTxtLoc, safeTxtLoc, hideGainLoc, hideLossLoc]
    ediData[data_appending_index()][19:27] = [choiceStart, choiceEnd, 
                                           isiStart, isiEnd, 
                                           outcomeStart, outcomeEnd,
                                           itiStart, itiEnd]

# ------------------------------------------------------------------------------------
# STATIC CHOICE SET #
# ------------------------------------------------------------------------------------

# static choice set instructions
statStartTxt.draw()
win.flip()
statInstructionsStart = timer.getTime()
endTask()
response = event.waitKeys(keyList = ['v', 'n'], timeStamped = timer)
statInstructionsEnd = response[0][1]
empty_data_appending()
ediData[data_appending_index()][17:19] = [statInstructionsStart, statInstructionsEnd]

# load task stimuli file 
staticDF = pd.read_csv("BGP_staticTrials.csv") # create before I officially joined the lab - I never noticed before - I wonder why the practice is an excel while the static is a csv

# randomize trials 
staticRandTrial = staticDF.sample(frac = 1).reset_index(drop = True) # use pandas to take all the rows and randomize them

# preparation for grid search
riskygain_values = [] # for gain (riskyoption1)
riskyloss_values = [] # for loss (riskyoption2)
certain_values = [] # for safe (safeoption)
choices = [] # for choiceMade

# static choice set task
for s in range(staticSet):

    # Need to call in the practice file for the gain, loss, and safe text 
    gain = staticRandTrial.riskyoption1[s]
    loss = staticRandTrial.riskyoption2[s]
    safe = staticRandTrial.safeoption[s]

    # Adjusting Trial Start - Python starts at 0: This makes trials start at 1
    trial = s + 1 

    # Check Trial
    checkTrial = staticRandTrial.ischecktrial[s]

    # Round Choice Option Monetary Values to be Shown
    choice_value_rounding()

    # Randomize Choice Option Locations
    choice_location_randomizing()

    # Start of Trial
    decision_window_starting()

    # End Task if Wanted/Needed
    endTask()

    # Choice Made
    decision_making()
    
    # ISI
    isi_waiting()
    
    # Choice Outcome
    outcome_showing()
    
    # ITI
    iti_waiting()
    
    itiStart = timer.getTime()
    core.wait(itiStatic[s])
    itiEnd = timer.getTime()
    
    # Saving Data
    empty_data_appending()
    ediData[data_appending_index()][0:2] = [trial, checkTrial] 
    ediData[data_appending_index()][2:5] = [gainRounded, lossRounded, safeRounded] 
    ediData[data_appending_index()][7:17] = [choiceMade, choiceKey, outcomeValue, 
                                          loc, riskSplitLoc, gainTxtLoc, lossTxtLoc, safeTxtLoc, hideGainLoc, hideLossLoc]
    ediData[data_appending_index()][19:27] = [choiceStart, choiceEnd, 
                                           isiStart, isiEnd, 
                                           outcomeStart, outcomeEnd,
                                           itiStart, itiEnd]
    
    # Grid Search Data
    riskygain_values.append(gain)
    riskyloss_values.append(loss)
    certain_values.append(safe)
    choices.append(choiceMade)
    

# ------------------------------------------------------------------------------------
# GRID SEARCH #
# ------------------------------------------------------------------------------------

# Prepare choice set values to remove any nans
finiteGainVals = []
finiteLossVals = []
finiteSafeVals = []
finiteChoices = []

# just save trial things where participant responded
for t in range(len(choices)):
    if math.isfinite(choices[t]):
        finiteGainVals.append(riskygain_values[t])
        finiteLossVals.append(riskyloss_values[t])
        finiteSafeVals.append(certain_values[t])
        finiteChoices.append(choices[t])
        
# Prepare rho & mu values
n_rho_values = 200;
n_mu_values = 201;

rmin = 0.3
rmax = 2.2
rstep = (rmax - rmin)/(n_rho_values-1)

mmin = 7
mmax = 80
mstep = (mmax - mmin)/(n_mu_values-1)

rho_values = [];
mu_values = [];

for r in range(n_rho_values):
    rho_values += [rmin + r*rstep];

for m in range(n_mu_values):
    mu_values += [mmin + m*mstep];      

# Execute the grid search
best_nll_value = 1e10; # a preposterously bad first NLL

for r in range(n_rho_values):
    for m in range(n_mu_values):
        nll_new = pt_nll([rho_values[r], mu_values[m]], finiteGainVals, finiteLossVals, finiteSafeVals, finiteChoices);
        if nll_new < best_nll_value:
            best_nll_value = nll_new;
            bestR = r + 1; # "+1" corrects for diff. in python vs. R indexing
            bestM = m + 1; # "+1" corrects for diff. in python vs. R indexing

print('The best R index is', bestR, 'while the best M index is', bestM, ', with an NLL of', best_nll_value);

# getting dynamic choice set files
fname = []

#fname.append("../edi/ediTasks/day1_rdm_wmc/ediRDM/ediRDMdynamic/bespoke_choiceset_rhoInd%03i_muInd%03i.csv" % (bestR, bestM))
#bespokeFilename = os.path.join(ediRDMdir, "ediRDMdynamic", "bespoke_choiceset_rhoInd%03i_muInd%03i.csv" % (bestR, bestM))
bespokeFilename = os.path.join(BGP_taskFolder, "BGP_dynamicTrials_trinaryDifficulty", "bespoke_choiceset_rhoInd%03i_muInd%03i.csv" % (bestR, bestM)) # this calls the Dynamic Choice Set files with intermediate difficulty
fname.append(bespokeFilename)
dynamicChoiceSetFilename = fname[0] # dyanmic choice set file to be used for participant

# saving out parameter data - "bestRho" & "bestMu"
empty_data_appending()
ediData[data_appending_index()][27:29] = [bestR, bestM]

# prepping for dynamic choice set instructions
fittingStartTxt.draw()
win.flip()
fitInstructionsStart = timer.getTime()
endTask()
core.wait(choiceWin) # added this waiting time - original would end grid search once it would run through everything
fitInstructionsEnd = timer.getTime()
empty_data_appending()
ediData[data_appending_index()][17:19] = [fitInstructionsStart, fitInstructionsEnd]

# ------------------------------------------------------------------------------------
# DYNAMIC CHOICE SET #
# ------------------------------------------------------------------------------------

# dynamic choice set instructions
dynaStartTxt.draw()
win.flip()
dynaInstructionsStart = timer.getTime()
endTask()
response = event.waitKeys(keyList = ['v', 'n'], timeStamped = timer)
dynaInstructionsEnd = response[0][1]
empty_data_appending()
ediData[data_appending_index()][17:19] = [dynaInstructionsStart, dynaInstructionsEnd]

# load task stimuli file 
dynamicDF = pd.read_csv(dynamicChoiceSetFilename) # create before I officially joined the lab - I never noticed before - I wonder why the practice is an excel while the static is a csv

# evenly split task stimuli file (let's make it a function?)
def equal_split_by_difficulty(fileDF, trialCount, choiceType = "type_e0i1d2", optionType = "reject0accept"):
    
    # equally split the choice types: easy (0), intermediate (1), difficult (2)
    easy = fileDF[fileDF[choiceType] == 0]
    intermediate = fileDF[fileDF[choiceType] == 1]
    difficult = fileDF[fileDF[choiceType] == 2]
    
    equalSplit = trialCount // 3 # dividing by 3
    
    equalEasy = equalSplit
    equalIntermediate = equalSplit
    equalDifficult = equalSplit
    
    # equally split the choice type BY option type: reject (0: easy and intermediate), accept (1:easy and intermediate), 2s (difficult)
    easyReject = easy[easy[optionType] == 0]
    easyAccept = easy[easy[optionType] == 1]
    
    easyTrials = pd.concat([
        easyReject.sample(n = equalSplit // 2),
        easyAccept.sample(n = equalSplit // 2)
        ])
    
    intermediateReject = intermediate[intermediate[optionType] == 0]
    intermediateAccept = intermediate[intermediate[optionType] == 1]
    
    intermediateTrials = pd.concat([
        intermediateReject.sample(n = equalSplit // 2),
        intermediateAccept.sample(n = equalSplit // 2)
        ])
    
    #difficultUnknown = difficult[difficult[optionType] == 2]
    difficultTrials = difficult.sample(n = equalSplit)
    
    # put it all into a new data frame
    return pd.concat([
        easyTrials,
        intermediateTrials,
        difficultTrials
        ]).sample(frac = 1).reset_index(drop = True)

# randomize trials 
#dynamicRandTrial = dynamicDF.sample(frac = 1).reset_index(drop = True) # use pandas to take all the rows and randomize them

dynamicRandTrial = equal_split_by_difficulty(
    dynamicDF,
    dynamicSet,
    choiceType = "type_e0i1d2",
    optionType = "reject0accept1"
    )

# break logic (a break in halfway through the trials)
if dynamicSet % 2 == 0:
    breakTime = dynamicSet // 2
else:
    breakTime = random.choice([
        dynamicSet // 2,
        dynamicSet // 2+1
    ])
    
breakTxt = visual.TextStim(
    win,
    text = 'Do not move your head from the mount.\n\n You may take a break for one minute.\n\n Press "V" or "N" to move forward once a minute has passed.',
    font = a,
    height = instructionsH,
    pos = center,
    color = c2
)

# dynamic choice set task
for d in range(dynamicSet):

    # Give the participant a break halfway through
    if d == breakTime:
        breakTimer = core.Clock()
        onBreak = True
        while onBreak and breakTimer.getTime() < 60:
            breakTxt.draw()
            win.flip()
            response = event.waitKeys(keyList = ['v', 'n'], timeStamped = timer)
            pracInstructionsEnd = response[0][1]
            empty_data_appending()
            ediData[data_appending_index()][17:19] = [pracInstructionsStart, pracInstructionsEnd]
            
    # Need to call in the practice file for the gain, loss, and safe text 
    gain = dynamicRandTrial.riskyoption1[d]
    loss = dynamicRandTrial.riskyoption2[d]
    safe = dynamicRandTrial.safeoption[d]

    # Adjusting Trial Start - Python starts at 0: This makes trials start at 1
    trial = d + 1 

    # Dynamic Choice Set Specific Data
    choiceP = dynamicRandTrial.choiceP[d]
    difficulty = dynamicRandTrial.type_e0i1d2[d]

    # Round Choice Option Monetary Values to be Shown
    choice_value_rounding()

    # Randomize Choice Option Locations
    choice_location_randomizing()

    # Start of Trial
    decision_window_starting()

    # End Task if Wanted/Needed
    endTask()

    # Choice Made
    decision_making()
    
    # ISI
    isi_waiting()
    
    # Choice Outcome
    outcome_showing()
    
    # ITI
    iti_waiting()
    
    itiStart = timer.getTime()
    core.wait(itiDynamic[d])
    itiEnd = timer.getTime()
    
    # Saving Data
    empty_data_appending()
    ediData[data_appending_index()][0] = trial 
    ediData[data_appending_index()][2:5] = [gainRounded, lossRounded, safeRounded] 
    ediData[data_appending_index()][5:7] = [choiceP, difficulty]
    ediData[data_appending_index()][7:17] = [choiceMade, choiceKey, outcomeValue, 
                                          loc, riskSplitLoc, gainTxtLoc, lossTxtLoc, safeTxtLoc, hideGainLoc, hideLossLoc]
    ediData[data_appending_index()][19:27] = [choiceStart, choiceEnd, 
                                           isiStart, isiEnd, 
                                           outcomeStart, outcomeEnd,
                                           itiStart, itiEnd]

#
##
### CLOSING INSTRUCTIONS ###

# break instructions
endStartTxt.draw()
win.flip()
endInstructionsStart = timer.getTime()
endTask()
response = event.waitKeys(keyList = ['return'], timeStamped = timer)
endInstructionsEnd = response[0][1]
empty_data_appending()
ediData[data_appending_index()][17:19] = [endInstructionsStart, endInstructionsEnd]

# call experimenter instructions
callStartTxt.draw()
win.flip()
callInstructionsStart = timer.getTime()
endTask()
response = event.waitKeys(keyList = ['space'], timeStamped = timer)
callInstructionsEnd = response[0][1]
empty_data_appending()
ediData[data_appending_index()][17:19] = [callInstructionsStart, callInstructionsEnd]

if doET:
    et.sendMessage('BGP Recording Stopped')
    et.sendMessage('post 100 pause')
    pylink.pumpDelay(100)
    et.stopRecording()
    et.closeDataFile()
    dateTime = time.strftime("_%Y%m%d-%H%M%S", time.localtime())
    eyeTracking_dataDir = os.path.join(BGP_eyeTracking_dataFolder + edf_fname + 
        '_' + studyName + 'eyeTracking' + dateTime + '.edf')
    et.receiveDataFile(edf_fname + '.edf', eyeTracking_dataDir)
    et.close()

if doET:
    subprocess.run(["edf2asc.exe", eyeTracking_dataDir])

#os.path.join('/Users/Display/Desktop/Github/edi/ediTasks/day1_rdm_wmc/ediData/'


win.close()



# saving out data
os.chdir(qvfTasks_folder)

bgpDF = pd.DataFrame(ediData)
dateTime = time.strftime("%Y%m%d-%H%M%S", time.localtime())
BGP_choiceBehavior_fileName = os.path.join(BGP_choiceBehavior_dataFolder, f"{studyName}{subID}_{taskName}_choiceBehavior_{dateTime}.csv")
bgpDF.to_csv(BGP_choiceBehavior_fileName, header = False, index = False)

# make it so that it creates a data folder in the desktop too
#os.makedirs(qvfBGPdata_choiceBehavior_folder, exist_ok=True)
#os.makedirs(qvfBGPdata_eyeTracking_folder, exist_ok=True)



#qvfData_folder = os.path.join(qvfBGP_folder, "qvfData") # QVF Data Folder
#qvfBGPdata_choiceBehavior_folder = (qvfData_folder, "qvfBGP_choiceBehaviorData") # BGP Choice Behavior Data Folder
#qvfBGPdata_eyeTracking_folder = (qvfData_folder, "qvfBGP_eyeTrackingData") # BGP Eye-Tracking Data Folder


        




























































