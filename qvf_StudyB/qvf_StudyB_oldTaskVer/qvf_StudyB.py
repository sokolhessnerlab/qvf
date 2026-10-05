
# --------------------------------------------------------------------------------------------------------------------------------------------
# QUITTING VS. FACILIATION (QVF) Study A: BESPOKE GAMBLING PARADIGM (BGP) - DISCRETE RANGES FULL CONTINUUM (MODEL PREDICTED TRINARY DIFFICULTY)
# --------------------------------------------------------------------------------------------------------------------------------------------

"""

Name: QVF Study A (intermediate difficult choices are defined by the model predicted output of probable changes in choice consistency)

Purpose: This script is designed to call (import) all the QVF Study A-related tasks, except for questionnaires on Qualtrics, as modules. 
         Through this method, only one script is required to sequentially launch all of the tasks. Additionally, 
         all participant-related information is automatically inputted into each task after an initial manual input.

Author: J. Von R. Monteza (2026/09/22)

"""

# SET IT OFF! 

# Activate the PsychoPy Shell below by pressing enter. Then copy/paste code below into the Shell. Modify the code accordingly:
# import os; os.chdir("/Users/Display/Desktop/Github/qvf/qvf_StudyB/"); import qvf_StudyB; qvf_StudyB.qvf_StudyB('XXX', 1, 3, 1, 1)
# import os; os.chdir("/Users/shlab/Documents/GitHub/qvf/qvf_StudyB/"); import qvf_StudyB; qvf_StudyB.qvf_StudyB('XXX',1,3,1,1)


# import os; os.chdir("/Users/Display/Desktop/Github/qvf/qvf_StudyA/qvf_StudyA_tasks/"); import qvf_StudyA; qvf_StudyA.qvf_StudyA('XXX',1,3,1,1)

# DIRECTORY SETTING
# import os; os.chdir("/Users/shlab/Documents/GitHub/qvf/qvf_StudyB/"); import qvf_StudyB; qvf_StudyB.qvf_StudyB('XXX',1,2,1,1)
# import os; os.chdir(os.path.expanduser("~/Documents/GitHub/qvf/qvf_StudyB/")); import qvf_StudyB; qvf_StudyB.qvf_StudyB('XXX', 1, 2, 1, 1)
# import os; from pathlib import Path; os.chdir(next(Path.home().rglob("/Documents/GitHub/qvf/qvf_StudyB"))); import qvf_StudyB; qvf_StudyB.qvf_StudyB('XXX', 1, 2, 1, 1)




def qvf_StudyB(subID, isReal, compNum, taskSet, doET): # define the function and specify the argument(s)

    # subID
            # Must be three digits (e.g., 001, 093, 458, etc.)
    # isReal
            # 0 - for testing
            # 1 - for real
    # compNum:
            # 1 - VM Laptop
            # 2 - Chicharron/Kebab
            # 3 - tofu 
    # taskSet:
            # 1 - do all
            # 2 - do ospan and symspan only
            # 1 - do symspan only
    # doET:
            # 0 - No eye-tracking
            # 1 - Do eye-tracking
    
    # let us know things are starting...
    print('starting study for participant', subID)
    # print (isReal, compNum, taskSet, doET

    # IMPORT MODULES
    import os
    import pandas as pd
    import sys
    #from psychopy import core

    # SET WORKING DIRECTORY #
    if compNum ==1:
        dirName = ("C:\\Users\\jvonm\\Documents\\GitHub\\edi\\ediTasks\\day1_rdm_wmc\\ediRDM")
        dataDirName = ("\\GitHub\\edi\\ediTasks\\day1_rdm_wmc\\ediData")
    elif compNum ==2:
        dirName = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
        print (dirName) # ("/Users/shlab/Documents/Github/qvf/qvf_StudyB")
        dataDirName = os.path.abspath(os.path.join(dirName, "data"))
        print (dataDirName) # ("/Users/shlab/Documents/Github/qvf/qvf_StudyB/data")
    elif compNum ==3:
        dirName = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
        print (dirName) # ("/Users/Display/Desktop/Github/qvf/qvf_StudyB")
        dataDirName = os.path.abspath(os.path.join(dirName, "data"))
        print (dataDirName) # ("/Users/Display/Desktop/Github/qvf/qvf_StudyB/data")
    
    os.chdir(dirName)

    # IMPORT TASK SCRIPTS #
    # BGP
    import bgp.bgpTask_modelPredicted_trinaryDifficulty
    # OSpan
    from ospan.ospanTaskModule import ospanTask
    # SymSpan
    from symspan.symSpanTaskModule import symSpanTask
    
    # SETTING TASK VARIABLES & PRESENTATION ORDER
    if taskSet == 1:
        
        print("BEFORE BGP")
        print("cwd:", os.getcwd())
        print("dirName:", dirName)
        print("dataDirName:", dataDirName)
        
        # risky decision-making task (input arguments determined above) 
        bgp.bgpTask_modelPredicted_trinaryDifficulty
        
        print("CURRENT DIRECTORY:", os.getcwd())
        print("DIRNAME:", dirName)
        print("DATADIR:", dataDirName)
        print("ABOUT TO START OSPAN")
        
        # ospan instructions + instructions quiz + practice + task
        ospanTask(subID, isReal,dirName, dataDirName)
        
        print("OSPAN FINISHED")
        
        # symspan instructions + instructions quiz + practice + task
        symSpanTask(subID, isReal,dirName, dataDirName)
        
    elif taskSet == 2:
        
        ospanTask(subID, isReal,dirName, dataDirName)

        symSpanTask(subID, isReal,dirName, dataDirName)
        
    elif taskSet == 3:
        
        symSpanTask(subID, isReal,dirName, dataDirName)
    
    
    
    
    
    
