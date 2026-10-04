
# ------------------------------------------------------------------------------------
# QUITTING VS. FACILIATION (QVF) STUDY B: BESPOKE GAMBLING PARADIGM (BGP) - DISCRETE RANGES FULL CONTINUUM (MODEL PREDICTED TRINARY DIFFICULTY)
# ------------------------------------------------------------------------------------

"""

Bespoke Gambling Paradigm (BGP) - Discrete Ranges Full Continuum (Model Predicted Trinary Difficulty)

Author: J. Von R. Monteza (2026.09.22)

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

######################################################################################
##### TIME TO SET THINGS UP! #########################################################
######################################################################################

# ------------------------------------------------------------------------------------
# PACKAGE & MODULE SETUP #
# ------------------------------------------------------------------------------------

# directory and filing
import os
import sys
import time

# psychopy 
import psychopy
from psychopy import visual, core, event, monitors
print(psychopy.__version__)

# task related
import random
import pandas as pd
import numpy as np 
import math

# ------------------------------------------------------------------------------------
# PARTICIPANT SESSION SETUP #
# ------------------------------------------------------------------------------------

# name of the study
studyName = 'qvfB'

# set the participant ID
subID = 'XXX'

# name of the task
taskName = 'BGP'

# set if it is a real or test run
isReal = 0 # 1 = yes, a real run vs. 0 = no, a test run

# set to collect eye-tracking data
doET = 0 # 1 = yes, do eye-tracking vs. 0 = no, only do behavioral

countTrial = 1 # 1 = yes, count trials vs. 0 = no, don't count trials

# ------------------------------------------------------------------------------------
# DIRECTORY SETUP #
# ------------------------------------------------------------------------------------

# create the various directories
BGP_taskFolder = os.path.abspath(os.path.dirname(os.path.abspath(__file__))) # BGP Task Folder - to run the BGP (current directory)
print (BGP_taskFolder)
QVF_studyFolder = os.path.abspath(os.path.dirname(BGP_taskFolder)) # QVF Tasks Folder
print (QVF_studyFolder)
QVF_dataFolder = os.path.abspath(os.path.join(QVF_studyFolder, "data")) # QVF Data Folder
print (QVF_dataFolder)
BGP_choiceBehavior_dataFolder = os.path.join(QVF_dataFolder, "bgp_choiceBehaviorData") # BGP Choice Behavior Data Folder
print (BGP_choiceBehavior_dataFolder)
BGP_eyeTracking_dataFolder = os.path.join(QVF_dataFolder, "bgp_eyeTrackingData") # BGP Eye-Tracking Data Folder
print (BGP_eyeTracking_dataFolder)

# BGP_taskFolder, qvfTasks_folder, data_folder, BGP_choiceBehavior_dataFolder, BGP_eyeTracking_dataFolder

# set the current directory to the BGP folder to run the BGP
os.chdir(BGP_taskFolder)

# ------------------------------------------------------------------------------------
# GENERAL OBJECT FORMATTING SETUP #
# ------------------------------------------------------------------------------------

# luminance equated colors
colr01 = [0.5216,0.5216,0.5216] # gray color for background, choice option text, and risky option line
colr02 = [-0.0667,0.6392,1] # blue color for choice option circles, OR text, and V-Left and N-Left text

# monitor screen
#screenSize = [1280, 1024] # How do I make it so that it gets the right screen size no matter what device?

# font style
font01 = 'Arial'

# font size
instrTxtHgt = .05
trialTxtHgt = .05
optTxtHgt = .10
fixTxtHgt = .05
noRespTxtHgt = .10

# text wrap
txtWrap = 1.3

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
countLoc = [0, -.35]

# trial time
decisionTime = 4 # decision window
isiTime = 1 # interstimulus interval window
ocTime = 1 # outcome window
itiStatic = [] # intertrial interval window (created below)
itiDynamic = []
breakTime = 60

# ------------------------------------------------------------------------------------
# WINDOW SETUP #
# ------------------------------------------------------------------------------------

win = visual.Window(
    size = [1280, 1024], 
    units = 'height', 
    monitor ='testMonitor', 
    fullscr = True, 
    color = colr01
) # Don't quite now yet the exact translation from height to pix in terms of size and location

# ------------------------------------------------------------------------------------
# EYE-TRACKING DATA COLLECTION SETUP #
# ------------------------------------------------------------------------------------

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
    #edf_fname = '%s%s_%s' % (studyName, subID, taskName)
    #edf_fname = '%s%s' % (studyName, subID)
    edf_fname = f"{studyName}{subID}" # only takes a few characters if I remember correctly!
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
    
if doET == 0:
    etInstruction = ''
elif doET == 1:
    etInstruction = '~ Keep your head still on the eye-tracking head mount ~\n\n'

# ------------------------------------------------------------------------------------
# STIMULI SETUP #
# ------------------------------------------------------------------------------------

# ~ Instructions ~

# general instructions
bgpStartTxt = visual.TextStim(
    win,
    text = ('STARTING THE DECISION-MAKING TASK\n\n'
            f'{etInstruction}'
            'You are now starting the decision-making task\n\n'
            'As discussed in the instructions, you will choose between a gamble and a guaranteed alternative choice option. '
            'Keep your left-index finger on the "V" key AND your right-index finger on the "N" key. '
            'Press the "V" key to select the option on the left OR the "N" key to select the option on the right.\n\n' 
            'Press "Enter/Return" to move on to the next screen'),
    font = font01,
    height = instrTxtHgt,
    wrapWidth = txtWrap,
    pos = center,
    color = colr02
)

# practice instructions
pracStartTxt = visual.TextStim(
    win,
    text = '',
    font = font01,
    height = instrTxtHgt,
    wrapWidth = txtWrap,
    pos = center,
    color = colr02
)

# static choice set instructions
statStartTxt = visual.TextStim(
    win,
    text = '',
    font = font01,
    height = instrTxtHgt,
    wrapWidth = txtWrap,
    pos = center,
    color = colr02
)

# fitting process instructions
fittingStartTxt = visual.TextStim(
    win,
    text = 'ROUND 1 COMPLETE!\n\n'
           'Setting up for the last round of\n the decision-making task\n\n'
           'Please wait...',
    font = font01,
    height = instrTxtHgt,
    wrapWidth = txtWrap,
    pos = center,
    color = colr02
)

# dynamic choice set instructions
dynaStartTxt = visual.TextStim(
    win,
    text = '',
    font = font01,
    height = instrTxtHgt,
    wrapWidth = txtWrap,
    pos = center,
    color = colr02
)

## task closing instructions
#endStartTxt = visual.TextStim(
#    win,
#    text = 'You have sucessfully completed the first task in this experiment!\n\nPlease take a brief 1 minute break. \n\nYou are welcome to take a longer break, but keep in mind this study should take no longer than 1 hour to complete. \n\nWhen you are ready to move on, press "enter to continue.\n',
#    font = a,
#    height = instructionsH,
#    wrapWidth = txtWrap,
#    pos = center,
#    color = c2
#)

# task don + call experimenter instructions
taskEndTxt = visual.TextStim(
    win,
    text = 'DECISION-MAKING TASK COMPLETE!\n\n'
           'Congratulations! You have sucessfully completed the\n first task in this study!\n\n'
           'Please press the white doorbell button to\n call the researcher back in...',
    font = font01,
    height = instrTxtHgt,
    wrapWidth = txtWrap,
    pos = center,
    color = colr02
)

# ~ Choice Trial Components ~

# decision window stimuli
vOpt = visual.Circle(
    win,
    size = choiceSize, 
    fillColor = colr02, 
    lineColor = colr02,
    pos = leftLoc,
    edges = 128
)

nOpt = visual.Circle(
    win,
    size = choiceSize, 
    fillColor = colr02, 
    lineColor = colr02,
    pos = rightLoc,
    edges = 128
)

riskSplit = visual.Rect(
    win, 
    size = riskSize,
    fillColor = colr01, 
    lineColor = colr01 
)

gainTxt = visual.TextStim(
    win, 
    font = font01,
    height = optTxtHgt,
    color = colr01
)

lossTxt = visual.TextStim(
    win, 
    font = font01,
    height = optTxtHgt,
    color = colr01
)

safeTxt = visual.TextStim(
    win, 
    font = font01,
    height = optTxtHgt,
    color = colr01
)

orTxt = visual.TextStim(
    win, 
    text = 'OR',
    font = font01,
    height = trialTxtHgt,
    pos = center,
    color = colr02
)

vTxt = visual.TextStim(
    win, 
    text = 'V - Left',
    font = font01,
    height = trialTxtHgt,
    pos = vLoc,
    color = colr02
)

nTxt = visual.TextStim(
    win, 
    text = 'N - Left',
    font = font01,
    height = trialTxtHgt,
    pos = nLoc,
    color = colr02);

countTrialTxt = visual.TextStim(
    win, 
    text = '',
    font = font01,
    height = trialTxtHgt,
    pos = countLoc,
    color = colr02)

# isi and iti fixation stimuli
fixTxt = visual.TextStim(
    win, 
    text = '+',
    font = font01,
    height = fixTxtHgt,
    pos = center,
    color = colr02
)

# choice outcome stimuli
noRespTxt = visual.TextStim(
    win, 
    text = 'You did not\nrespond in time',
    font = font01,
    height = noRespTxtHgt,
    pos = center,
    color = colr02
)

ocRiskyHide = visual.Rect(
    win,
    size = hideSize, 
    fillColor = colr01, 
    lineColor = colr01,
)

# ------------------------------------------------------------------------------------
# FUNCTION SETUP #
# ------------------------------------------------------------------------------------

#
##
### CREATING DATA SAVING FUNCTIONS ###

# save choice behavior data
def save_choiceBehaviorData():
    dateTime = time.strftime("%Y%m%d-%H%M%S")
    BGP_choiceBehaviorData_fileName = os.path.join(BGP_choiceBehavior_dataFolder, f"{studyName}{subID}_{taskName}_choiceBehaviorData_EndTask_{dateTime}.csv")
    bgpDF.to_csv(BGP_choiceBehaviorData_fileName, header = True, index = False)

# save eye-tracking data
def save_eyeTrackingData():
    if doET:
        et.sendMessage('BGP Recording Stopped')
        et.sendMessage('post 100 pause')
        pylink.pumpDelay(100)
        et.stopRecording()
        et.closeDataFile()
        session_identifier = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        BGP_eyeTrackingData_fileName = os.path.join(BGP_eyeTracking_dataFolder, f"{studyName}{subID}_{taskName}_eyeTrackingData_{session_identifier}.edf")
        et.receiveDataFile(edf_fname + '.edf', BGP_eyeTrackingData_fileName)
        et.close()
        subprocess.run(["edf2asc.exe", BGP_eyeTrackingData_fileName])

#
##
### CREATING ESCAPE FUNCTIONS ###

# end the task
def endTask(): 
    
    print("Escape detected - ending task")

    # save choice behavior data
    print("Saving choice behavior data...")
    save_choiceBehaviorData()
    print("Choice behavior data saved")

    # save eye-tracking data
    print("Saving eye-tracking data...")
    save_eyeTrackingData()
    print("Eye-tracking data saved")

    # close task
    print("Closing task...")
    win.close()
    core.quit()
    sys.exit()

# allow task escape 
def wait_OR_escape(timeDur):
    startTime = timer.getTime()

    while timer.getTime() - startTime < timeDur:
        if 'escape' in event.getKeys(keyList=['escape']):
            endTask()
        core.wait(0.01)

#
##
### CREATING PROSPECT MODEL FUNCTIONS ###

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

#
##
### CREATING TRIAL FUNCTIONS ###

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
    if countTrial == 1:
        countTrialTxt.setText(trial)
        countTrialTxt.draw()

# drawing decision window stimuli and retrieving trial times and choices
def decision_Window():
    global choiceStart, choiceEnd
    global response, choiceKey, choiceMade, outcomeValue
    
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
    
    response = event.waitKeys(maxWait = decisionTime, keyList = ['v', 'n', 'escape'], timeStamped = timer)
    
    if response is not None and response[0][0] == 'escape':
        endTask()
    
    if response is None:
        choiceMade = math.nan
        choiceKey = math.nan
        outcomeValue = math.nan
        choiceEnd = math.nan
    elif response[0][0] == 'v' or response[0][0] == 'n':
        if (loc == 1 and response[0][0] == 'v') or (loc == 2 and response[0][0] == 'n'):
            choiceKey = response[0][0]
            choiceMade = 1 # chose the risky option
            outcomeValue = random.choice([gainRounded, lossRounded]) # randomly chooses the gain or loss
        elif (loc == 1 and response[0][0] == 'n') or (loc == 2 and response[0][0] == 'v'):
            choiceKey = response[0][0]
            choiceMade = 0 # chose the safe option
            outcomeValue = safeRounded
        choiceEnd = response[0][1]

# setting up isi
def isi_Window(isiTime):
    global isiStart, isiEnd
    
    fixTxt.draw()
    win.flip()
    isiStart = timer.getTime()
    
    wait_OR_escape(isiTime)
        
    isiEnd = timer.getTime()

# showing outcome of the choice made
def outcome_Window():
    global outcomeStart, outcomeEnd
    
    if response is None:
        noRespTxt.draw()
        win.flip()
        outcomeStart = timer.getTime()
        wait_OR_escape(ocTime)
        outcomeEnd = timer.getTime()
    elif (loc == 1 and response[0][0] == 'v'):
        if outcomeValue == gainRounded: # risky option on the left was chosen and won
            ocRiskyHide.setPos(hideLossLoc)
            vOpt.draw()
            gainTxt.draw()
            ocRiskyHide.draw()
            win.flip()
            outcomeStart = timer.getTime()
            wait_OR_escape(ocTime)
            outcomeEnd = timer.getTime()
        elif outcomeValue == lossRounded: # risky option on the left was chosen and lost 
            ocRiskyHide.setPos(hideGainLoc)
            vOpt.draw()
            lossTxt.draw()
            ocRiskyHide.draw()
            win.flip()
            outcomeStart = timer.getTime()
            wait_OR_escape(ocTime)
            outcomeEnd = timer.getTime()
    elif (loc == 2 and response[0][0] == 'n'):
        if outcomeValue == gainRounded: # risky option on the right was chosen and won
            ocRiskyHide.setPos(hideLossLoc)
            nOpt.draw()
            gainTxt.draw()
            ocRiskyHide.draw()
            win.flip()
            outcomeStart = timer.getTime()
            wait_OR_escape(ocTime)
            outcomeEnd = timer.getTime()
        elif outcomeValue == lossRounded: # risky option on the right was chosen and lost
            ocRiskyHide.setPos(hideGainLoc)
            nOpt.draw()
            lossTxt.draw()
            ocRiskyHide.draw()
            win.flip()
            outcomeStart = timer.getTime()
            wait_OR_escape(ocTime)
            outcomeEnd = timer.getTime()
    elif (loc == 1 and response[0][0] == 'n') and outcomeValue == safeRounded: # safe option on the right was chosen
        nOpt.draw()
        safeTxt.draw()
        win.flip()
        outcomeStart = timer.getTime()
        wait_OR_escape(ocTime)
        outcomeEnd = timer.getTime()
    elif (loc == 2 and response[0][0] == 'v') and outcomeValue == safeRounded: # safe option on the left was chosen
        vOpt.draw()
        safeTxt.draw()
        win.flip()
        outcomeStart = timer.getTime()
        wait_OR_escape(ocTime)
        outcomeEnd = timer.getTime()

# randomize iti time for both the trials in the static and dynamic choice sets
def shuffle(array):
    currentIndex = len(array)
    while currentIndex != 0:
        randomIndex = random.randint(0, currentIndex - 1)
        currentIndex -= 1
        array[currentIndex], array[randomIndex] = array[randomIndex], array[currentIndex]

# setting up iti
def iti_Window(itiTime): # each of the choice set's iti's are done differently - doesn't clearly work as well to do itiEnd
    global itiStart, itiEnd
    
    fixTxt.draw()
    win.flip()
    itiStart = timer.getTime()

    wait_OR_escape(itiTime)

    itiEnd = timer.getTime()

#
##
### CREATING DATA APPENDING FUNCTIONS ###

# append a new row of empty data ("") into the data structure that matches the length of the columns 
def empty_data_appending():
    bgpData.append([""] * len(bgpData[0]))

# index the last row of the data structure for future appending (e.g., ediData[data_appending_index][<whatever column>])
def data_appending_index():
    return len(bgpData) - 1

######################################################################################
##### TIME TO START THE TASK! ########################################################
######################################################################################

# ------------------------------------------------------------------------------------
# BGP DATA FRAME SETUP #
# ------------------------------------------------------------------------------------

bgpData = []
bgpData.append(
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

bgpDF = pd.DataFrame(
    columns = [
    "bestRho", "bestMu", "bestNLL", # prospect theory model fitting values
    "trialNumber", "gainValue", "lossValue", "safeValue", "choiceKey", "choiceMade", "outcomeValue", # trial values
    "choiceProbability", "choiceDifficulty", "choicePredicted", "checkTrial", "blockNumber", # stimuli values
    "choiceLocation", "riskSplitLocation", "gainLocation", "lossLocation", "safeLocation", "hideGainLocation", "hideLossLocation", # trial location values
    "instrStart", "instrEnd", "instrTimeDur", # time values
    "choiceStart", "choiceEnd", "choiceTimeDur", 
    "isiStart", "isiEnd", "isiTimeDur", "isiTimeSet",
    "outcomeStart", "outcomeEnd", "ocTimeDur", "outcomeTimeSet", 
    "itiStart", "itiEnd", "itiTimeDur", "itiTimeSet"
    ]
)

# ------------------------------------------------------------------------------------
# TIMER SETUP #
# ------------------------------------------------------------------------------------

timer = core.Clock()

# ------------------------------------------------------------------------------------
# COUNT TRIALS #
# ------------------------------------------------------------------------------------

#if countTrial == 0:
#    countLoc = [0, 7]
#    countColor = colr01
#elif countTrial == 1:
#    countLoc = [0, -.35]
#    countColor = colr02

# ------------------------------------------------------------------------------------
# GENERAL INSTRUCTIONS START #
# ------------------------------------------------------------------------------------

bgpStartTxt.draw()
win.flip()
bgpInstrStart = timer.getTime()
response = event.waitKeys(keyList = ['return', 'escape'], timeStamped = timer)
if response[0][0] == 'escape':
    endTask()
bgpInstrEnd = response[0][1]
empty_data_appending()
bgpData[data_appending_index()][17:19] = [bgpInstrStart, bgpInstrEnd]

# Saving Time Values
bgpInstrDur = bgpInstrEnd - bgpInstrStart
bgpDF.loc[len(bgpDF), ["instrStart", "instrEnd", "instrTimeDur"]] = [
                        bgpInstrStart, bgpInstrEnd, bgpInstrDur]

# ------------------------------------------------------------------------------------
# PRACTICE CHOICE SET #
# ------------------------------------------------------------------------------------

# load task stimuli file 
practiceDF = pd.read_excel("BGP_practiceTrials.xlsx") # sequentially presented

# set amount of trials 
if isReal == 0: 
    practiceSet = 2
elif isReal == 1:
    practiceSet = len(practiceDF)

# practice choice set instructions
pracStartTxt.text = ('STARTING THE PRACTICE ROUND\n\n'
                     f'{etInstruction}'
                     'We are now moving on to the PRACTICE ROUND\n'
                     f'There will be {practiceSet} practice trials\n\n'
                     'The structure of the practice round is identical to what you will encounter in the real rounds. ' 
                     'The goal of the practice round is to practice the timing of your decision-making within the four (4) second decision window.\n\n'
                     'Press "V" or "N" to begin the PRACTICE ROUND')
pracStartTxt.draw()
win.flip()
if doET:
    et.sendMessage('before practice instruction start')
pracInstrStart = timer.getTime()
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
    et.sendMessage('BGP Pupillometry Recording Started - Practice Instructions Shown')
response = event.waitKeys(keyList = ['v', 'n', 'escape'], timeStamped = timer)
if response[0][0] == 'escape':
    endTask()
pracInstrEnd = response[0][1]
empty_data_appending()
bgpData[data_appending_index()][17:19] = [pracInstrStart, pracInstrEnd]

# Saving Time Values
pracInstrDur = pracInstrEnd - pracInstrStart
bgpDF.loc[len(bgpDF), ["instrStart", "instrEnd", "instrTimeDur"]] = [
                        pracInstrStart, pracInstrEnd, pracInstrDur]

# practice choice set task
for p in range(practiceSet):

    # Trial (Python starts at 0: This makes trials start at 1)
    trial = p + 1
    
    # Extract Choice Option Values
    gain = practiceDF.riskyGain[p]
    loss = practiceDF.riskyLoss[p]
    safe = practiceDF.alternative[p]

    # Round Choice Option Monetary Values
    choice_value_rounding()

    # Randomize Choice Option Locations
    choice_location_randomizing()

    # Decision
    decision_Window()
    
    # ISI
    isiPractice = practiceDF.isi[p] # isi set in file to each choice option combination
    isiTime = isiPractice
    isi_Window(isiTime)
    
    # Outcome
    outcome_Window()
    
    # ITI
    itiPractice = practiceDF.iti[p] # iti set in file to each choice option combination
    itiTime = itiPractice # not really needed, but visually guides logic
    iti_Window(itiTime)
    
    # Calculate Trial Time Data
    choiceDur = choiceEnd - choiceStart
    isiDur = isiEnd - isiStart
    outcomeDur = outcomeEnd - outcomeStart
    itiDur = itiEnd - itiStart
    
    # Saving Data
    empty_data_appending()
    bgpData[data_appending_index()][0] = trial 
    bgpData[data_appending_index()][2:5] = [gainRounded, lossRounded, safeRounded] 
    bgpData[data_appending_index()][7:17] = [choiceMade, choiceKey, outcomeValue, 
                                          loc, riskSplitLoc, gainTxtLoc, lossTxtLoc, safeTxtLoc, hideGainLoc, hideLossLoc]
    bgpData[data_appending_index()][19:27] = [choiceStart, choiceEnd, 
                                           isiStart, isiEnd, 
                                           outcomeStart, outcomeEnd,
                                           itiStart, itiEnd]
    
    # Saving Data
    bgpDF.loc[len(bgpDF), ["trialNumber", "gainValue", "lossValue", "safeValue", "choiceKey", "choiceMade", "outcomeValue",
                           "choiceLocation", "riskSplitLocation", "gainLocation", "lossLocation", "safeLocation", "hideGainLocation", "hideLossLocation",
                           "choiceStart", "choiceEnd", "choiceTimeDur",
                           "isiStart", "isiEnd", "isiTimeDur", "isiTimeSet",
                           "outcomeStart", "outcomeEnd", "ocTimeDur", "outcomeTimeSet", 
                           "itiStart", "itiEnd", "itiTimeDur", "itiTimeSet"]] = [
                            trial, gainRounded, lossRounded, safeRounded, choiceKey, choiceMade, outcomeValue,
                            loc, riskSplitLoc, gainTxtLoc, lossTxtLoc, safeTxtLoc, hideGainLoc, hideLossLoc,
                            choiceStart, choiceEnd, choiceDur,
                            isiStart, isiEnd, isiDur, isiTime,
                            outcomeStart, outcomeEnd, outcomeDur, ocTime,
                            itiStart, itiEnd, itiDur, itiTime]


# ------------------------------------------------------------------------------------
# STATIC CHOICE SET #
# ------------------------------------------------------------------------------------

# load task stimuli file 
staticDF = pd.read_csv("BGP_staticTrials.csv") # create before I officially joined the lab - I never noticed before - I wonder why the practice is an excel while the static is a csv

# set amount of trials 
if isReal == 0: 
    staticSet = 2
elif isReal == 1:
    staticSet = len(staticDF) # 40 real trials & 10 check trials

# static choice set instructions
statStartTxt.text = ('PRACTICE ROUND COMPLETE!\n STARTING ROUND 1\n\n'
                     f'{etInstruction}'
                     'We are now moving on to the first real round: ROUND 1\n'
                     f'There will be {staticSet} trials in the first real round\n\n'
                     'Keep in mind that responding quickly in this task will NOT speed up the task. '
                     'Please take enough time to view and consider each choice option before you make a choice within the four (4) second decision window.\n\n'
                     'Press "V" or "N" to begin ROUND 1')
statStartTxt.draw()
win.flip()
statInstrStart = timer.getTime()
response = event.waitKeys(keyList = ['v', 'n', 'escape'], timeStamped = timer)
if response[0][0] == 'escape':
    endTask()
statInstrEnd = response[0][1]
empty_data_appending()
bgpData[data_appending_index()][17:19] = [statInstrStart, statInstrEnd]

# Saving Time Values
statInstrDur = statInstrEnd - statInstrStart
bgpDF.loc[len(bgpDF), ["instrStart", "instrEnd", "instrTimeDur"]] = [
                        statInstrStart, statInstrEnd, statInstrDur]

# randomize trials 
staticRandTrial = staticDF.sample(frac = 1).reset_index(drop = True) # use pandas to take all the rows and randomize them

# creating and shuffling ITIs 
itiStatic = [3, 3.5] * (staticSet // 2) # jittered between 3 and 3.5 seconds for all trials
shuffle(itiStatic)

# preparation for grid search
riskygain_values = [] # for gain (riskyoption1)
riskyloss_values = [] # for loss (riskyoption2)
certain_values = [] # for safe (safeoption)
choices = [] # for choiceMade

# static choice set task
for s in range(staticSet):

    # Trial (Python starts at 0: This makes trials start at 1)
    trial = s + 1 
    
    # Extract Choice Option Values
    gain = staticRandTrial.riskyoption1[s]
    loss = staticRandTrial.riskyoption2[s]
    safe = staticRandTrial.safeoption[s]

    # Round Choice Option Monetary Values
    choice_value_rounding()

    # Randomize Choice Option Locations
    choice_location_randomizing()

    # Decision
    decision_Window()
    
    # ISI
    isi_Window(isiTime)
    
    # Outcome
    outcome_Window()
    
    # ITI
    itiTime = itiStatic[s]
    iti_Window(itiTime)
    
    # Calculate Trial Time Data
    choiceDur = choiceEnd - choiceStart
    isiDur = isiEnd - isiStart
    outcomeDur = outcomeEnd - outcomeStart
    itiDur = itiEnd - itiStart
    
    # Stimuli File Data
    checkTrial = staticRandTrial.ischecktrial[s]
    
    # Saving Data
    empty_data_appending()
    bgpData[data_appending_index()][0:2] = [trial, checkTrial] 
    bgpData[data_appending_index()][2:5] = [gainRounded, lossRounded, safeRounded] 
    bgpData[data_appending_index()][7:17] = [choiceMade, choiceKey, outcomeValue, 
                                          loc, riskSplitLoc, gainTxtLoc, lossTxtLoc, safeTxtLoc, hideGainLoc, hideLossLoc]
    bgpData[data_appending_index()][19:27] = [choiceStart, choiceEnd, 
                                           isiStart, isiEnd, 
                                           outcomeStart, outcomeEnd,
                                           itiStart, itiEnd]
    
    # Saving Data
    bgpDF.loc[len(bgpDF), ["trialNumber", "gainValue", "lossValue", "safeValue", "choiceKey", "choiceMade", "outcomeValue",
                           "checkTrial",
                           "choiceLocation", "riskSplitLocation", "gainLocation", "lossLocation", "safeLocation", "hideGainLocation", "hideLossLocation",
                           "choiceStart", "choiceEnd", "choiceTimeDur",
                           "isiStart", "isiEnd", "isiTimeDur", "isiTimeSet",
                           "outcomeStart", "outcomeEnd", "ocTimeDur", "outcomeTimeSet", 
                           "itiStart", "itiEnd", "itiTimeDur", "itiTimeSet"]] = [
                            trial, gainRounded, lossRounded, safeRounded, choiceKey, choiceMade, outcomeValue,
                            checkTrial,
                            loc, riskSplitLoc, gainTxtLoc, lossTxtLoc, safeTxtLoc, hideGainLoc, hideLossLoc,
                            choiceStart, choiceEnd, choiceDur,
                            isiStart, isiEnd, isiDur, isiTime,
                            outcomeStart, outcomeEnd, outcomeDur, ocTime,
                            itiStart, itiEnd, itiDur, itiTime]

    # Grid Search Data
    riskygain_values.append(gain)
    riskyloss_values.append(loss)
    certain_values.append(safe)
    choices.append(choiceMade)
    

# ------------------------------------------------------------------------------------
# GRID SEARCH - APPROXIMIZATION OPTIMIZATION PROCEDURE #
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
bespokeFilename = os.path.join(BGP_taskFolder, "BGP_dynamicTrials_modelPredicted_trinaryDifficulty", "qvf_bespoke_choiceset_rhoInd%03i_muInd%03i.csv" % (bestR, bestM)) # this calls the Dynamic Choice Set files with intermediate difficulty
fname.append(bespokeFilename)
dynamicChoiceSetFilename = fname[0] # dyanmic choice set file to be used for participant

# saving out parameter data - "bestRho" & "bestMu"
empty_data_appending()
bgpData[data_appending_index()][27:29] = [bestR, bestM]

## Saving Prospect Theory Model Fitting Values
#bgpDF.loc[len(bgpDF), ["bestRho", "bestMu", "bestNLL"]] = [
#                                bestR, bestM, best_nll_value]

# prepping for dynamic choice set instructions
fittingStartTxt.draw()
win.flip()
fitInstrStart = timer.getTime()
wait_OR_escape(decisionTime) # added this waiting time - original would end grid search once it would run through everything
if 'escape' in event.getKeys(keyList=['escape']):
            endTask()
fitInstrEnd = timer.getTime()
empty_data_appending()
bgpData[data_appending_index()][17:19] = [fitInstrStart, fitInstrEnd]

# Saving Prospect Theory Model Fitting Values and Time Values
fitInstrDur = fitInstrEnd - fitInstrStart
bgpDF.loc[len(bgpDF), ["bestRho", "bestMu", "bestNLL",
                       "instrStart", "instrEnd", "instrTimeDur"]] = [
                        bestR, bestM, best_nll_value,
                        fitInstrStart, fitInstrEnd, fitInstrDur]

# ------------------------------------------------------------------------------------
# DYNAMIC CHOICE SET #
# ------------------------------------------------------------------------------------

# load task stimuli file 
dynamicDF = pd.read_csv(dynamicChoiceSetFilename) # create before I officially joined the lab - I never noticed before - I wonder why the practice is an excel while the static is a csv

# set amount of trials 
if isReal == 0:
    dynamicSet = 4
elif isReal == 1:
    dynamicSet = len(dynamicDF)

# dynamic choice set instructions
dynaStartTxt.text = ('ROUND 1 COMPLETE!\n STARTING ROUND 2\n\n'
                     f'{etInstruction}'
                     'We are now moving on to the second real round: ROUND 2\n'
                     f'There will be {dynamicSet} trials in the second real round\n'
                     'You will have a break halfway through the trials\n\n'
                     'Keep in mind that responding quickly in this task will NOT speed up the task. '
                     'Please take enough time to view and consider each choice option before you make a choice within the four (4) second decision window.\n\n'
                     'Press "V" or "N" to begin ROUND 2')
dynaStartTxt.draw()
win.flip()
dynaInstrStart = timer.getTime()
response = event.waitKeys(keyList = ['v', 'n', 'escape'], timeStamped = timer)
if response[0][0] == 'escape':
    endTask()
dynaInstrEnd = response[0][1]
empty_data_appending()
bgpData[data_appending_index()][17:19] = [dynaInstrStart, dynaInstrEnd]

# Saving Prospect Theory Model Fitting Values and Time Values
dynaInstrDur = dynaInstrEnd - dynaInstrStart
bgpDF.loc[len(bgpDF), ["instrStart", "instrEnd", "instrTimeDur"]] = [
                        dynaInstrStart, dynaInstrEnd, dynaInstrDur]

## separate the dynamic choice set into blocks
#dynamicBlock01 = dynamicDF[dynamicDF["dynamicblocknum"] == 1].sample(frac=1).reset_index(drop=True)
#dynamicBlock02 = dynamicDF[dynamicDF["dynamicblocknum"] == 2].sample(frac=1).reset_index(drop=True)

# number of trials per dynamic block
trialsPerBlock = dynamicSet // 2

# shuffle within each block, then take the desired number of trials
dynamicBlock01 = (dynamicDF[dynamicDF["dynamicblocknum"] == 1].sample(frac=1).reset_index(drop=True).iloc[:trialsPerBlock])
dynamicBlock02 = (dynamicDF[dynamicDF["dynamicblocknum"] == 2].sample(frac=1).reset_index(drop=True).iloc[:trialsPerBlock])

# creating and shuffling ITIs 
itiDynamic = [3, 3.5] * (dynamicSet // 2) # jittered between 3 and 3.5 seconds for all trials
shuffle(itiDynamic)

## Number of actual dynamic trials
#nDynamicTrials = len(dynamicBlock01) + len(dynamicBlock02)
#
## Create one ITI for every actual dynamic trial
#itiDynamic = [3, 3.5] * (nDynamicTrials // 2)
#
## If there is an odd number of trials, add one more ITI
#if len(itiDynamic) < nDynamicTrials:
#    itiDynamic.append(3)
#
#shuffle(itiDynamic)
#
#print("Dynamic block 1:", len(dynamicBlock01))
#print("Dynamic block 2:", len(dynamicBlock02))
#print("Total dynamic trials:", nDynamicTrials)
#print("ITI list length:", len(itiDynamic))

breakStartTxt = visual.TextStim(
    win,
    text = ('FIRST HALF OF ROUND 2 COMPLETE!\n'
            'STARTING YOUR BREAK\n\n'
            f'{etInstruction}'
            'You have completed the first half of the second real round\n\n'
            'You may now take a break for one minute. Further instructions will be provided after the break.'),
    font = font01,
    height = instrTxtHgt,
    wrapWidth = txtWrap,
    pos = center,
    color = colr02
)

breakEndTxt = visual.TextStim(
    win,
    text = ('STARTING THE SECOND HALF OF ROUND 2\n\n'
            f'{etInstruction}'
            'We are now moving on to the second half of ROUND 2\n\n'
            'Keep in mind that responding quickly in this task will NOT speed up the task. '
            'Please take enough time to view and consider each choice option before you make a choice within the four (4) second decision window.\n\n'
            'Press "V" or "N" to continue ROUND 2'),
    font = font01,
    height = instrTxtHgt,
    wrapWidth = txtWrap,
    pos = center,
    color = colr02
)

# dynamic choice set task: block 1
for d in range(len(dynamicBlock01)):
    
    # Trial (Python starts at 0: This makes trials start at 1)
    trial = d + 1 
    
    # Extract Choice Option Values
    gain = dynamicBlock01.riskyoption1[d]
    loss = dynamicBlock01.riskyoption2[d]
    safe = dynamicBlock01.safeoption[d]
    
    # Round Choice Option Monetary Values
    choice_value_rounding()

    # Randomize Choice Option Locations
    choice_location_randomizing()

    # Decision
    decision_Window()
    
    # ISI
    isi_Window(isiTime)
    
    # Outcome
    outcome_Window()
    
    # ITI
    itiTime = itiDynamic[d]
    iti_Window(itiTime)
    
    # Calculate Trial Time Data
    choiceDur = choiceEnd - choiceStart
    isiDur = isiEnd - isiStart
    outcomeDur = outcomeEnd - outcomeStart
    itiDur = itiEnd - itiStart
    
    # Stimuli File Data
    choiceP = dynamicBlock01.choiceP[d]
    difficulty = dynamicBlock01.type_e0i1d2[d]
    predicted = dynamicBlock01.reject0accept1[d]
    block = dynamicBlock01.dynamicblocknum[d]
    
    # Saving Data
    empty_data_appending()
    bgpData[data_appending_index()][0] = trial 
    bgpData[data_appending_index()][2:5] = [gainRounded, lossRounded, safeRounded] 
    bgpData[data_appending_index()][5:7] = [choiceP, difficulty]
    bgpData[data_appending_index()][7:17] = [choiceMade, choiceKey, outcomeValue, 
                                          loc, riskSplitLoc, gainTxtLoc, lossTxtLoc, safeTxtLoc, hideGainLoc, hideLossLoc]
    bgpData[data_appending_index()][19:27] = [choiceStart, choiceEnd, 
                                           isiStart, isiEnd, 
                                           outcomeStart, outcomeEnd,
                                           itiStart, itiEnd]
    
    # Saving Data
    bgpDF.loc[len(bgpDF), ["trialNumber", "gainValue", "lossValue", "safeValue", "choiceKey", "choiceMade", "outcomeValue",
                           "choiceProbability", "choiceDifficulty", "choicePredicted", "blockNumber",
                           "choiceLocation", "riskSplitLocation", "gainLocation", "lossLocation", "safeLocation", "hideGainLocation", "hideLossLocation",
                           "choiceStart", "choiceEnd", "choiceTimeDur",
                           "isiStart", "isiEnd", "isiTimeDur", "isiTimeSet",
                           "outcomeStart", "outcomeEnd", "ocTimeDur", "outcomeTimeSet", 
                           "itiStart", "itiEnd", "itiTimeDur", "itiTimeSet"]] = [
                            trial, gainRounded, lossRounded, safeRounded, choiceKey, choiceMade, outcomeValue,
                            choiceP, difficulty, predicted, block,
                            loc, riskSplitLoc, gainTxtLoc, lossTxtLoc, safeTxtLoc, hideGainLoc, hideLossLoc,
                            choiceStart, choiceEnd, choiceDur,
                            isiStart, isiEnd, isiDur, isiTime,
                            outcomeStart, outcomeEnd, outcomeDur, ocTime,
                            itiStart, itiEnd, itiDur, itiTime]

# break between each dynamic block
#breakTimer = core.Clock()
#onBreak = True
## during the minute break
#while onBreak and breakTimer.getTime() < 60: # can't i just do core wait now?
#    breakTxt.draw()
#    win.flip()
#    breakStart = timer.getTime()
#    core.wait(0.01)
breakStartTxt.draw()
win.flip()
breakInstrStart = timer.getTime()
wait_OR_escape(breakTime)
# after the minute break
breakEndTxt.draw()
win.flip()
response = event.waitKeys(keyList = ['v', 'n', 'escape'], timeStamped = timer)
if 'escape' in event.getKeys(keyList=['escape']):
            endTask()
breakInstrEnd = response[0][1]
empty_data_appending()
bgpData[data_appending_index()][17:19] = [breakInstrStart, breakInstrEnd]

# Saving Prospect Theory Model Fitting Values and Time Values
breakInstrDur = breakInstrEnd - breakInstrStart
bgpDF.loc[len(bgpDF), ["instrStart", "instrEnd", "instrTimeDur"]] = [
                        breakInstrStart, breakInstrEnd, breakInstrDur]

# dynamic choice set task: block 2
for d in range(len(dynamicBlock02)):
    
    # Trial (Python starts at 0: This makes trials start at 1 and then continue from from dynamicBlock01)
    trial = d + 1 + len(dynamicBlock01)
    
    # Extract Choice Option Values
    gain = dynamicBlock02.riskyoption1[d]
    loss = dynamicBlock02.riskyoption2[d]
    safe = dynamicBlock02.safeoption[d]

    # Round Choice Option Monetary Values
    choice_value_rounding()

    # Randomize Choice Option Locations
    choice_location_randomizing()

    # Decision
    decision_Window()
    
    # ISI
    isi_Window(isiTime)
    
    # Outcome
    outcome_Window()
    
    # ITI
    itiTime = itiDynamic[d + len(dynamicBlock01)]
    iti_Window(itiTime)
    
    # Calculate Trial Time Data
    choiceDur = choiceEnd - choiceStart
    isiDur = isiEnd - isiStart
    outcomeDur = outcomeEnd - outcomeStart
    itiDur = itiEnd - itiStart
    
    # Stimuli File Data
    choiceP = dynamicBlock01.choiceP[d]
    difficulty = dynamicBlock01.type_e0i1d2[d]
    predicted = dynamicBlock01.reject0accept1[d]
    block = dynamicBlock01.dynamicblocknum[d]
    
    # Saving Data
    empty_data_appending()
    bgpData[data_appending_index()][0] = trial 
    bgpData[data_appending_index()][2:5] = [gainRounded, lossRounded, safeRounded] 
    bgpData[data_appending_index()][5:7] = [choiceP, difficulty]
    bgpData[data_appending_index()][7:17] = [choiceMade, choiceKey, outcomeValue, 
                                          loc, riskSplitLoc, gainTxtLoc, lossTxtLoc, safeTxtLoc, hideGainLoc, hideLossLoc]
    bgpData[data_appending_index()][19:27] = [choiceStart, choiceEnd, 
                                           isiStart, isiEnd, 
                                           outcomeStart, outcomeEnd,
                                           itiStart, itiEnd]

    # Saving Data
    bgpDF.loc[len(bgpDF), ["trialNumber", "gainValue", "lossValue", "safeValue", "choiceKey", "choiceMade", "outcomeValue",
                           "choiceProbability", "choiceDifficulty", "choicePredicted", "blockNumber",
                           "choiceLocation", "riskSplitLocation", "gainLocation", "lossLocation", "safeLocation", "hideGainLocation", "hideLossLocation",
                           "choiceStart", "choiceEnd", "choiceTimeDur",
                           "isiStart", "isiEnd", "isiTimeDur", "isiTimeSet",
                           "outcomeStart", "outcomeEnd", "ocTimeDur", "outcomeTimeSet", 
                           "itiStart", "itiEnd", "itiTimeDur", "itiTimeSet"]] = [
                            trial, gainRounded, lossRounded, safeRounded, choiceKey, choiceMade, outcomeValue,
                            choiceP, difficulty, predicted, block,
                            loc, riskSplitLoc, gainTxtLoc, lossTxtLoc, safeTxtLoc, hideGainLoc, hideLossLoc,
                            choiceStart, choiceEnd, choiceDur,
                            isiStart, isiEnd, isiDur, isiTime,
                            outcomeStart, outcomeEnd, outcomeDur, ocTime,
                            itiStart, itiEnd, itiDur, itiTime]





## evenly split task stimuli file (let's make it a function?)
#def equal_split_by_difficulty(fileDF, trialCount, choiceType = "type_e0i1d2", optionType = "reject0accept"):
#    
#    # equally split the choice types: easy (0), intermediate (1), difficult (2)
#    easy = fileDF[fileDF[choiceType] == 0]
#    intermediate = fileDF[fileDF[choiceType] == 1]
#    difficult = fileDF[fileDF[choiceType] == 2]
#    
#    equalSplit = trialCount // 3 # dividing by 3
#    
#    equalEasy = equalSplit
#    equalIntermediate = equalSplit
#    equalDifficult = equalSplit
#    
#    # equally split the choice type BY option type: reject (0: easy and intermediate), accept (1:easy and intermediate), 2s (difficult)
#    easyReject = easy[easy[optionType] == 0]
#    easyAccept = easy[easy[optionType] == 1]
#    
#    easyTrials = pd.concat([
#        easyReject.sample(n = equalSplit // 2),
#        easyAccept.sample(n = equalSplit // 2)
#        ])
#    
#    intermediateReject = intermediate[intermediate[optionType] == 0]
#    intermediateAccept = intermediate[intermediate[optionType] == 1]
#    
#    intermediateTrials = pd.concat([
#        intermediateReject.sample(n = equalSplit // 2),
#        intermediateAccept.sample(n = equalSplit // 2)
#        ])
#    
#    #difficultUnknown = difficult[difficult[optionType] == 2]
#    difficultTrials = difficult.sample(n = equalSplit)
#    
#    # put it all into a new data frame
#    return pd.concat([
#        easyTrials,
#        intermediateTrials,
#        difficultTrials
#        ]).sample(frac = 1).reset_index(drop = True)
#
## randomize trials 
##dynamicRandTrial = dynamicDF.sample(frac = 1).reset_index(drop = True) # use pandas to take all the rows and randomize them
#dynamicRandTrial = equal_split_by_difficulty(
#    dynamicDF,
#    dynamicSet,
#    choiceType = "type_e0i1d2",
#    optionType = "reject0accept1"
#    )
#
## break logic (a break in halfway through the trials)
#if dynamicSet % 2 == 0:
#    breakTime = dynamicSet // 2
#else:
#    breakTime = random.choice([
#        dynamicSet // 2,
#        dynamicSet // 2+1
#    ])
#
#breakTxt = visual.TextStim(
#    win,
#    text = 'Do not move your head from the mount.\n\n You may take a break for one minute.',
#    font = a,
#    height = instructionsH,
#    pos = center,
#    color = c2
#)
#
#breakDoneTxt = visual.TextStim(
#    win,
#    text = 'Do not move your head from the mount.\n\n You may take a break for one minute.\n\n Press "V" or "N" to move forward.',
#    font = a,
#    height = instructionsH,
#    pos = center,
#    color = c2
#)
#
## dynamic choice set task
#for d in range(dynamicSet):
#            
#    # Give the participant a break halfway through
#    if d == breakTime:
#        breakTimer = core.Clock()
#        onBreak = True
#        # during the minute break
#        while onBreak and breakTimer.getTime() < 60:
#            breakTxt.draw()
#            win.flip()
#            core.wait(0.01)
#        # after the minute break
#        breakDoneTxt.draw()
#        win.flip()
#        response = event.waitKeys(keyList = ['v', 'n'], timeStamped = timer)
#        pracInstructionsEnd = response[0][1]
#        empty_data_appending()
#        bgpData[data_appending_index()][17:19] = [pracInstructionsStart, pracInstructionsEnd]
#            
#    # Need to call in the practice file for the gain, loss, and safe text 
#    gain = dynamicRandTrial.riskyoption1[d]
#    loss = dynamicRandTrial.riskyoption2[d]
#    safe = dynamicRandTrial.safeoption[d]
#
#    # Adjusting Trial Start - Python starts at 0: This makes trials start at 1
#    trial = d + 1 
#
#    # Dynamic Choice Set Specific Data
#    choiceP = dynamicRandTrial.choiceP[d]
#    difficulty = dynamicRandTrial.type_e0i1d2[d]
#
#    # Round Choice Option Monetary Values to be Shown
#    choice_value_rounding()
#
#    # Randomize Choice Option Locations
#    choice_location_randomizing()
#
#    # Start of Trial
#    decision_window_starting()
#
#    # End Task if Wanted/Needed
#    endTask()
#
#    # Choice Made
#    decision_making()
#    
#    # ISI
#    isi_waiting()
#    
#    # Choice Outcome
#    outcome_showing()
#    
#    # ITI
#    iti_waiting()
#    
#    itiStart = timer.getTime()
#    core.wait(itiDynamic[d])
#    itiEnd = timer.getTime()
#    
#    # Saving Data
#    empty_data_appending()
#    bgpData[data_appending_index()][0] = trial 
#    bgpData[data_appending_index()][2:5] = [gainRounded, lossRounded, safeRounded] 
#    bgpData[data_appending_index()][5:7] = [choiceP, difficulty]
#    bgpData[data_appending_index()][7:17] = [choiceMade, choiceKey, outcomeValue, 
#                                          loc, riskSplitLoc, gainTxtLoc, lossTxtLoc, safeTxtLoc, hideGainLoc, hideLossLoc]
#    bgpData[data_appending_index()][19:27] = [choiceStart, choiceEnd, 
#                                           isiStart, isiEnd, 
#                                           outcomeStart, outcomeEnd,
#                                           itiStart, itiEnd]


## separate the dynamic choice set into blocks
#dynamicBlock01 = dynamicDF[dynamicDF["dynamicblocknum"] == 1].sample(frac=1).reset_index(drop=True)
#dynamicBlock02 = dynamicDF[dynamicDF["dynamicblocknum"] == 2].sample(frac=1).reset_index(drop=True)
#dynamicBlocks = [dynamicBlock01, dynamicBlock02]
#
## create dynamic choice set break text in between the blocks
#breakTxt = visual.TextStim(
#    win,
#    text='Do not move your head from the mount.\n\n You may take a break for one minute.',
#    font=a,
#    height=instructionsH,
#    pos=center,
#    color=c2
#)
#
#breakDoneTxt = visual.TextStim(
#    win,
#    text='Do not move your head from the mount.\n\n You may take a break for one minute.\n\n Press "V" or "N" to move forward.',
#    font=a,
#    height=instructionsH,
#    pos=center,
#    color=c2
#)
#
## dynamic choice set task
#trial = 0 # start the trials at 0 
#
#for dynamicBlocksIndex, dynamicRandTrial in enumerate(dynamicBlocks):
#
#    for d in range(len(dynamicRandTrial)):
#
#        # Adjusting Trial Start - Python starts at 0: This makes trials start at 1 & block 2 should carry on from block 1
#        trial += 1
#
#        # Need to call in the practice file for the gain, loss, and safe text
#        gain = dynamicRandTrial.riskyoption1[d]
#        loss = dynamicRandTrial.riskyoption2[d]
#        safe = dynamicRandTrial.safeoption[d]
#
#        # Dynamic Choice Set Specific Data
#        choiceP = dynamicRandTrial.choiceP[d]
#        difficulty = dynamicRandTrial.type_e0i1d2[d]
#
#        # Round Choice Option Monetary Values to be Shown
#        choice_value_rounding()
#
#        # Randomize Choice Option Locations
#        choice_location_randomizing()
#
#        # Start of Trial
#        decision_window_starting()
#
#        # End Task if Wanted/Needed
#        endTask()
#
#        # Choice Made
#        decision_making()
#
#        # ISI
#        isi_waiting()
#
#        # Choice Outcome
#        outcome_showing()
#
#        # ITI
#        iti_waiting()
#
#        itiStart = timer.getTime()
#        core.wait(itiDynamic[trial - 1])
#        itiEnd = timer.getTime()
#
#        # Saving Data
#        empty_data_appending()
#        bgpData[data_appending_index()][0] = trial
#        bgpData[data_appending_index()][2:5] = [gainRounded, lossRounded, safeRounded]
#        bgpData[data_appending_index()][5:7] = [choiceP, difficulty]
#        bgpData[data_appending_index()][7:17] = [choiceMade, choiceKey, outcomeValue,
#                                                 loc, riskSplitLoc, gainTxtLoc, lossTxtLoc, safeTxtLoc, hideGainLoc, hideLossLoc]
#        bgpData[data_appending_index()][19:27] = [choiceStart, choiceEnd,
#                                                  isiStart, isiEnd,
#                                                  outcomeStart, outcomeEnd,
#                                                  itiStart, itiEnd]
#
#    # Give the participant a break between the blocks
#    if blockIndex == 0:
#        breakTimer = core.Clock()
#        onBreak = True
#        # during the minute break
#        while onBreak and breakTimer.getTime() < 60:
#            breakTxt.draw()
#            win.flip()
#            core.wait(0.01)
#        # after the minute break
#        breakDoneTxt.draw()
#        win.flip()
#        response = event.waitKeys(keyList = ['v', 'n'], timeStamped = timer)
#        pracInstructionsEnd = response[0][1]
#        empty_data_appending()
#        bgpData[data_appending_index()][17:19] = [pracInstructionsStart, pracInstructionsEnd]

######################################################################################
##### TIME TO END THINGS! ############################################################
######################################################################################

# ------------------------------------------------------------------------------------
# CLOSING INSTRUCTIONS #
# ------------------------------------------------------------------------------------

## break instructions
#endStartTxt.draw()
#win.flip()
#endInstructionsStart = timer.getTime()
#endTask()
#response = event.waitKeys(keyList = ['return'], timeStamped = timer)
#endInstructionsEnd = response[0][1]
#empty_data_appending()
#bgpData[data_appending_index()][17:19] = [endInstructionsStart, endInstructionsEnd]
#
## Saving Time Values
#bgpTestDF.loc[len(bgpTestDF), ["instructionStart", "instructionEnd"]] = [
#                                endInstructionsStart, endInstructionsEnd]

# task done + call experimenter instructions
taskEndTxt.draw()
win.flip()
taskEndInstrStart = timer.getTime()
response = event.waitKeys(keyList = ['space', 'escape'], timeStamped = timer)
if response[0][0] == 'escape':
    endTask()
taskEndInstrEnd = response[0][1]
empty_data_appending()
bgpData[data_appending_index()][17:19] = [taskEndInstrStart, taskEndInstrEnd]

# Saving Prospect Theory Model Fitting Values and Time Values
endTaskDur = taskEndInstrEnd - taskEndInstrStart
bgpDF.loc[len(bgpDF), ["instrStart", "instrEnd", "instrTimeDur"]] = [
                        taskEndInstrStart, taskEndInstrEnd, endTaskDur]

# ------------------------------------------------------------------------------------
# SAVE DATA #
# ------------------------------------------------------------------------------------

#if doET:
#    et.sendMessage('BGP Recording Stopped')
#    et.sendMessage('post 100 pause')
#    pylink.pumpDelay(100)
#    et.stopRecording()
#    et.closeDataFile()
#    session_identifier = time.strftime("%Y%m%d-%H%M%S", time.localtime())
#    #session_identifier = edf_fname + time_str
#    #edf_dataDirName = os.path.join(BGP_eyeTracking_dataFolder, edf_fname + '_' + taskName +'_eyeTracking' + session_identifier + '.edf')
#    #et.receiveDataFile(edf_fname + '.edf', edf_dataDirName)
#    BGP_eyeTrackingData_fileName = os.path.join(BGP_eyeTracking_dataFolder, f"{studyName}{subID}_{taskName}_eyeTrackingData_{session_identifier}.edf")
#    #et.receiveDataFile(edf_fname + '.edf', os.path.join(BGP_eyeTracking_dataFolder, edf_fname + '_' + taskName +'_eyeTrackingData' + session_identifier + '.edf'))
#    et.receiveDataFile(edf_fname + '.edf', BGP_eyeTrackingData_fileName)
#    et.close()
#    
#    subprocess.run(["edf2asc.exe", BGP_eyeTrackingData_fileName])

save_eyeTrackingData()

#if doET:
 #   subprocess.run(["edf2asc.exe", edf_dataDirName])

#if doET:
 #   subprocess.run(["edf2asc.exe", os.path.join(BGP_eyeTracking_dataFolder, edf_fname + '_' + taskName +'_eyeTracking' + session_identifier + '.edf')])

win.close()


# saving out data
os.chdir(BGP_taskFolder)

bgpDF_oldFormat = pd.DataFrame(bgpData)
dateTime = time.strftime("%Y%m%d-%H%M%S")
BGP_choiceBehavior_fileName = os.path.join(BGP_choiceBehavior_dataFolder, f"{studyName}{subID}_{taskName}_choiceBehavior_{dateTime}.csv")
bgpDF_oldFormat.to_csv(BGP_choiceBehavior_fileName, header = False, index = False)

#dateTime = time.strftime("%Y%m%d-%H%M%S")
#BGP_choiceBehaviorData_fileName = os.path.join(BGP_choiceBehavior_dataFolder, f"{studyName}{subID}_TEST_{taskName}_choiceBehaviorData_{dateTime}.csv")
#bgpDF.to_csv(BGP_choiceBehaviorData_fileName, header = True, index = False)

save_choiceBehaviorData()

# make it so that it creates a data folder in the desktop too # use if state to find desktop pathway if it exists or at least creates one
#os.makedirs(qvfBGPdata_choiceBehavior_folder, exist_ok=True)
#os.makedirs(qvfBGPdata_eyeTracking_folder, exist_ok=True)



#qvfData_folder = os.path.join(qvfBGP_folder, "qvfData") # QVF Data Folder
#qvfBGPdata_choiceBehavior_folder = (qvfData_folder, "qvfBGP_choiceBehaviorData") # BGP Choice Behavior Data Folder
#qvfBGPdata_eyeTracking_folder = (qvfData_folder, "qvfBGP_eyeTrackingData") # BGP Eye-Tracking Data Folder


        


