
# --------------------------------------------------------------------------------------------------------------------------------------------
# QUITTING VS. FACILIATION (QVF) Study A: BESPOKE GAMBLING PARADIGM (BGP) - DISCRETE RANGES FULL CONTINUUM (SPLIT DIFFERENCE TRINARY DIFFICULTY)
# --------------------------------------------------------------------------------------------------------------------------------------------

"""

Name: QVF Study A (intermediate difficult choices are defined by the split difference decision probabilities between easy and difficult choices)

Purpose: This script is designed to call (import) all the QVF Study A-related tasks, except for questionnaires on Qualtrics, as modules. 
         Through this method, only one script is required to sequentially launch all of the tasks. Additionally, 
         all participant-related information is automatically inputted into each task after an initial manual input.

Author: J. Von R. Monteza (2026/09/22)

"""

# SET IT OFF! 

# Activate the PsychoPy Shell below by pressing enter. Then copy/paste code below into the Shell. Modify the code accordingly:
# import os; os.chdir("/Users/Display/Desktop/Github/qvf/qvf_StudyA/qvf_StudyA_tasks/"); import qvf_StudyA; qvf_StudyA.qvf_StudyA('XXX',1,3,1,1)
# import os; os.chdir("/Users/shlab/Documents/qvf/qvf_StudyA/qvf_StudyA_tasks/"); import qvf_StudyA; qvf_StudyA.qvf_StudyA('XXX',1,3,1,1)

def qvf_StudyA(subID, isReal, compNum, taskSet, doET): # define the function and specify the argument(s)

    # subID
            # Must be three digits (e.g., 001, 093, 458, etc.)
    # isReal
            # 0 - for testing
            # 1 - for real
    # compNum:
            # 1 - VM Laptop
            # 2 - Chicharron
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
        dirName = ("/Users/shlab/Documents/Github/cge/CGE/")
        dataDirName = ("/Users/shlab/Documents/Github/cge/CGE/data")
    elif compNum ==3:
        #dirName = os.path.abspath(os.path.dirname(os.path.abspath(__file__))) # BGP Task Folder - to run the BGP (current directory)
        #print (dirName)
        #mainDirName = os.path.abspath(os.path.dirname(dirName)) # QVF Tasks Folder
        #print (mainDirName)
        #dataDirName = os.path.abspath(os.path.join(mainDirName, "data")) # QVF Data Folder
        #print (dataDirName)
        dirName = ("/Users/Display/Desktop/Github/qvf/qvf_StudyA/qvf_StudyA_tasks")
        dataDirName = ("/Users/Display/Desktop/Github/qvf/qvf_StudyA/qvf_StudyA_tasks/data")
    
    os.chdir(dirName)

    # IMPORT TASK SCRIPTS #
    # BGP
    import bgp.bgpTask_splitDifference_trinaryDifficulty
    # OSpan
    from ospan.ospanTaskModule import ospanTask
    # SymSpan
    from symspan.symSpanTaskModule import symSpanTask
    
    # SETTING TASK VARIABLES & PRESENTATION ORDER
    if taskSet ==1:
        
        # risky decision-making task (input arguments determined above) 
        bgp.bgpTask_splitDifference_trinaryDifficulty
        
        # ospan instructions + instructions quiz + practice + task
        ospanTask(subID, isReal,dirName, dataDirName)
        
        # symspan instructions + instructions quiz + practice + task
        symSpanTask(subID, isReal,dirName, dataDirName)
        
    elif taskSet==2:
        
        ospanTask(subID, isReal,dirName, dataDirName)

        symSpanTask(subID, isReal,dirName, dataDirName)
        
    elif taskSet==3:
        
        symSpanTask(subID, isReal,dirName, dataDirName)
    
    
    
    
    
    
