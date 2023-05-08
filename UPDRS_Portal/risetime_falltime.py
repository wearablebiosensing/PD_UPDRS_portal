import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.signal import find_peaks
import gspread as gs
import json
import glob
import plotly.express as px


# Returns a df with peak_indexes and valley_indexes , peak_indexes  and valley indexes.
def peaks_and_valleys(df):
    idx_norm = preprocessing.normalize([df["index"][25:500]]).ravel()
    valley_idxs, valley_properties_idx = find_peaks(idx_norm*-1, width=(2,20),distance=5)#  prominence=(0.01, 4)  
    iqr_idx = np.quantile(idx_norm, 0.50)
    peaks_idxs, peaks_properties_idx = find_peaks(idx_norm, height=iqr_idx)
    peak_points = [idx_norm[j] for j in peaks_idxs]
    valley_points = [idx_norm[j] for j in valley_idxs]
    #print("peaks_idx: ",peaks_idxs)
    #print("valley_idxs: ", valley_idxs)

    # print("df.shape",df.shape)
    peak_inctances = [] # create a list of peaks 
    valley_instances = []
    # initial;ize the peaks to "Na"
    for index,value in df.iterrows():
        peak_inctances.append(0)
    # if the index is a peak then set the value in the peak_inctances list to be 0.
    for peak_index in peaks_idxs:
       # print("peaks_idxs: ",peak_index)
        peak_inctances[peak_index] = 1
    #print("peak_inctances: ",len(peak_inctances),peak_inctances)
    # initial;ize the peaks to "Na"
    for index,value in df.iterrows():
        valley_instances.append(0)
    for valley_idx  in  valley_idxs:
        valley_instances[valley_idx] = 1

    df["peaks_index"] = peak_inctances
    df["valleys_index"] = valley_instances

    # return df, peaks_idxs, valley_idxs,peak_points

################################ -------------------- RISE TIME ----------------- ################################################################################################
# Params - Peak indexes and valley indexes returned by  peaks_and_valleys().
# def rise_time(peaks_idxs,valley_idxs):
    ##### Params needed (1) valley indexes, (2) peak indexes
    # Psudo Code
    rise_time_arr = [] # Len of this array ahould be the same as number of peaks. 
    found_valley_to_peak = 0
    valle_done = 0 # for breaking if one iter of valleys is done.
    for v_idx, valley_idx in enumerate(valley_idxs):
        for idx, peaks_idx in enumerate(peaks_idxs):
            print("peak to valley peaks_idxs[idx]  < valley_idxs[v_idx]:", peaks_idxs[idx]  , valley_idxs[v_idx])
            #print("valley to peak rise time :", valley_idxs[v_idx], peaks_idxs[idx], peaks_idxs[len(peaks_idxs)-1])  
            if valley_idxs[v_idx] < peaks_idxs[idx]: # RISE TIME
               # print("valley idx, peak idx: ",valley_idxs[v_idx],peaks_idxs[idx])
                rise_time = df["time"][peaks_idxs[idx]]/1000000000 -  df["time"][valley_idxs[v_idx]]/1000000000
                rise_time_arr.append(rise_time)
                found_valley_to_peak = 1
            # if we parserd through all the peaks then code is done running.
            elif peaks_idxs[idx] == peaks_idxs[len(peaks_idxs)-1]: # last value 
                valle_done = 0 # Finished finding valleis - peaks 
            if found_valley_to_peak == 1:
                idx =idx+1 
        if valle_done == 0:
            break
df = pd.read_csv("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_IOTEX/iotex-glove/PD/Participant1/2021-06-24/rg_20210624-170825.csv")
peaks_and_valleys(df)
#print("rise_time_arr: ",rise_time_arr,len(rise_time_arr))
    
################################ -------------------- FALL TIME -----------------################################################################################################
    # Problem because there are two valles in one segment.!!!!!!!
    # fall_time_arr = []
    # found_peak_to_valley = 0
    # ptovall_done = 0
    # print("len(valley_idxs),len(peaks_idxs): ",len(valley_idxs),len(peaks_idxs))
    # for v_idx, valley_idx in enumerate(valley_idxs):
    #     for idx, peaks_idx in enumerate(peaks_idxs):
    #         # FALL TIME
    #         if peaks_idxs[idx]  < valley_idxs[v_idx]:
    #             print("FALL TIME idx,peaks_idxs[idx]  , valley_idxs[v_idx]",peaks_idxs[idx]  , valley_idxs[v_idx])
    #             fall_time = df["time"][valley_idxs[v_idx]] - df["time"][peaks_idxs[idx]]
    #             fall_time_arr.append(fall_time)
    #             found_peak_to_valley = 1
    #         elif peaks_idxs[idx] == peaks_idxs[len(peaks_idxs)-1]: # la
    #             ptovall_done = 0
    #         if found_peak_to_valley  == 1:
    #             idx = idx +1
    #             v_idx = v_idx+1
    #     if ptovall_done == 1:
    #         break
    # print("fall_time_arr: ",fall_time_arr,len(fall_time_arr))
''' 
dset_path = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_IOTEX/iotex-glove/PD"
# file_id = []
for i in range(1):
    pno = i+1
    print("Part No: {}".format(pno))
    # csv_files = glob.glob('C:/Users/dan95/Desktop/Participant1/**/*.csv',recursive=True)
    dset_csv_fpath = dset_path +'/Participant'+str(pno)+'/**/*.csv'
    # print("dset_csv_fpath: ",dset_csv_fpath)
    label_text_fpath = dset_path +'/Participant'+str(pno)+'/**/*.txt'
    csv_files = glob.glob(dset_csv_fpath,recursive=True)
    #print("csv_files: ",csv_files)
    text_files = glob.glob(label_text_fpath,recursive=True)
    rg_files = [s for s in csv_files if "rg_" in s]
    lg_files = [s for s in csv_files if "lg_" in s]
    #print("rg_files: ",rg_files)
    #print("len(rg_files),rg_files: ",len(rg_files),rg_files)
    all_sessions = []
    for file, text_file in zip(rg_files,text_files):
        print("file: ",file)
        with open(text_file) as f:
            lines = f.readlines()
       # print("medication state: ",lines[5],lines[4])
        df = pd.read_csv(file)
        df =df[df["activity"]==1]
        df_peaks_valleys, peak_indexs, valley_idxs,peak_points = peaks_and_valleys(df)
        print("file.split [12]: ",file.split("/")[11])
        #df.to_csv(dset_path +'/Participant'+str(pno) + "/"+ file.split("/")[12] + "/" + file.split("/")[13] + "_peak_results.csv",index=0)
        rise_time_arr  = rise_time(peak_indexs,valley_idxs)
        print("peak_points,rise_time_arr: ",len(peak_points),len(rise_time_arr))
        # print("rise_time_arr: ",rise_time_arr)
       # print("rise_time_arr: ", rise_time_arr)
        # create a list of medication statius same length as rise_time array.
        med_status = [lines[4]] * len(peak_points) 
        # gets the dates of the participants. 
        date_list = [file.split("/")[11]] * len(peak_points)
        file_id = [file.split("/")[12]] *len(peak_points)
        session1 = pd.DataFrame(
            {#'rise_time_arr_seconds': rise_time_arr,
            'peak_amplitude': peak_points,
             'med_status': med_status,
            'date_list': date_list,
            'file_id_session_num': file_id
            })
        all_sessions.append(session1)
    all_sessions_df = pd.concat(all_sessions)
all_sessions_df.to_csv("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_IOTEX/iotex-glove/PD/Participant1/peak_amplitudes.csv",index=0)
'''