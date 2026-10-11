
# --------------------------------------------------------------------------------------------------------------------------------------------
# QUITTING VS. FACILIATION (QVF) Study B: BESPOKE GAMBLING PARADIGM (BGP) - DISCRETE RANGES FULL CONTINUUM (MODEL PREDICTED TRINARY DIFFICULTY)
# --------------------------------------------------------------------------------------------------------------------------------------------

"""

Name: QVF Study B (intermediate difficult choices are defined by the model predicted output of probable changes in choice consistency)

Purpose: This script is designed to call (import) all the QVF Study A-related tasks, except for questionnaires on Qualtrics, as modules. 
         Through this method, only one script is required to sequentially launch all of the tasks. Additionally, 
         all participant-related information is automatically inputted into each task after an initial manual input.

Author: J. Von R. Monteza (2026/09/22)

"""

# SET IT OFF! 

# TO RUN THIS SCRIPT AND CALL THE TASK MODULES #
"""
Activate the PsychoPy Shell below by pressing enter. Then copy/paste code below into the Shell. Modify the code accordingly:
import os; os.chdir("/Users/Display/Desktop/Github/qvf/qvf_StudyB/"); import qvf_StudyB; qvf_StudyB.qvf_StudyB('XXX', 1, 3, 1, 1)
"""

def qvf_StudyB(subID, isReal, taskSet, doET): # define the function and specify the argument(s)
    
    # let us know things are starting...
    print('starting the study for participant', subID)
    print (isReal, taskSet, doET)

    # -------------------------------------------------------------------------------
    # IMPORT MODULES
    # -------------------------------------------------------------------------------
    import os
    import pandas as pd
    import sys
    #from psychopy import core

    # -------------------------------------------------------------------------------
    # SET WORKING DIRECTORY # (more flexible code)
    # -------------------------------------------------------------------------------
    # get the study folder directory
    dirName = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
    print (dirName)
    # get the study data folder directory
    dataDirName = os.path.abspath(os.path.join(dirName, "data"))
    print (dataDirName)
    # change the current directory
    os.chdir(dirName)

    # -------------------------------------------------------------------------------
    # IMPORT TASK SCRIPTS #
    # -------------------------------------------------------------------------------
    # BGP (Bespoke Gambling Paradigm)
    from bgp.bgpTask_modelPredicted_trinaryDifficulty import bgpTask
    # OSpan (Operation Span)
    from ospan.ospanTaskModule import ospanTask
    # SymSpan (Symmetry Span)
    from symspan.symSpanTaskModule import symSpanTask
    
    # -------------------------------------------------------------------------------
    # SETTING PRESENTATION ORDER & TASK VARIABLES #
    # -------------------------------------------------------------------------------
    # bgp, ospan, & symspan
    if taskSet == 1:
        
        bgpTask(subID, isReal, doET)
        
        ospanTask(subID, isReal, dirName, dataDirName)
        
        symSpanTask(subID, isReal, dirName, dataDirName)
    
    # ospan & sympspan
    elif taskSet == 2:
        
        ospanTask(subID, isReal, dirName, dataDirName)

        symSpanTask(subID, isReal, dirName, dataDirName)
    
    # sympspan
    elif taskSet == 3:
        
        symSpanTask(subID, isReal, dirName, dataDirName)

# -----------------------------------------------------------------------------------
# RUN THE MAIN SCRIPT (01. change the values accordingly 02. click save 03. click play)
# -----------------------------------------------------------------------------------
qvf_StudyB('XXX', 0, 1, 0)
    # qvf_StudyB(subID, isReal, taskSet, doET)
        # subID: must be three digits (e.g., 001, 093, 458, etc.)
        # isReal: 0 (test run) vs. 1 (real run)
        # taskSet: 1 (bgp, ospan, & symspan) vs. 2 (ospan & sympspan) vs. 3 (symspan)
        # doET: 0 (no eye-tracking) vs. 1 (do eye-tracking) - make sure to connect the eye-tracker
    
    
    
    
