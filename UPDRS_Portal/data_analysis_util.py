import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import dash_bootstrap_components as dbc
import glob
from sklearn import preprocessing

# Params - all_files_txt- root file path of the dataset.
def get_file_path_meta_info(all_files_txt):
    counter_lg = 0
    participant_list = []
    date_list = []
    num_task = []
    file_paths_lg = []
    participant_list_lg = []
    participant_list_rg = []
    file_paths_rg = []
    date_list_lg_all = []
    date_list_rg_all = []
    file_name_lg_all = []
    file_name_rg_all = []
    patient_form = []
    # Read Files.
    for full_path in all_files_txt:
        #print("ROOT FOLDER PATH ",full_path)
        all_folder_dates =  glob.glob(full_path + "/*")
        #print(all_folder_dates) 
        # Get a list of dates based on Participant ID
        for dates_folder in all_folder_dates:
            d_split = dates_folder.split("/")
            #print("d_split == \n",d_split[9],d_split[10])
            participant_list.append(d_split[9])
            date_list.append(d_split[10])
            #list all .csvfiles in that dates folder.
            dates_folder_files = glob.glob(dates_folder + "/*")
            for i in dates_folder_files:
                print("dates_folder_files File names: - ",i.split("/")[12])
                file_name = i.split("/")[12]
                if file_name.startswith("lg"):
                    print("LG file_name",file_name)
                    print("LG",i.split("/")[9],i.split("/")[10] , file_name)
                    counter_lg+=1
                    # print("counter_lg = \n",counter_lg)
                    participant_list_lg.append(i.split("/")[10])
                    date_list_lg_all.append(i.split("/")[11])
                    file_name_lg_all.append(file_name)
                    file_paths_lg.append(i)
                if file_name.startswith("rg"):
                    participant_list_rg.append(i.split("/")[10])
                    date_list_rg_all.append(i.split("/")[11])
                    file_name_rg_all.append(file_name)
                    file_paths_rg.append(i)
                # patient_form.txt
                if file_name.endswith("patient_form.txt"):
                    pass 

            num_task.append(counter_lg)
            counter_lg = 0 # Reset counting fils after each date is visited.
    df_file_paths = pd.DataFrame(
        {'ParticipantList': participant_list_lg,
        'DateList': date_list_lg_all,
        "FileName_LeftGlove":file_name_lg_all,
       # "FileName_RightGlove":file_name_rg_all, 
        #'DateList_RightGlove': date_list_rg_all,
        })
    df_file_paths_rg = pd.DataFrame(
        {'ParticipantList': participant_list_rg,
        'DateList': date_list_rg_all,
        "FileName_LeftGlove":file_name_rg_all 
        })
    df_file_paths.to_csv("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_IOTEX/iotex-glove/all_file_path.csv",index=0)
    df_file_paths_rg.to_csv("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_IOTEX/iotex-glove/rg_file_path.csv",index=0)
    print(len(participant_list),len(date_list),len(num_task),len(file_paths_lg),len(file_paths_rg))
    df_dates_pid = pd.DataFrame(
        {'ParticipantList': participant_list,
        'DateList': date_list,
        "NumTasks":num_task 
        })
    # df_dates_pid.to_csv("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/iotex-glove/pd_dates_list.csv",index=0)
# list all files ending with a .txt.
all_files_txt = glob.glob("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_IOTEX/iotex-glove/PD/*")
# print("all_files_txt: ",all_files_txt)
get_file_path_meta_info(all_files_txt)