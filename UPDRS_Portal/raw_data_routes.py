from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import glob
from sklearn import preprocessing
import gspread as gs
from scipy.signal import find_peaks
import plotly.figure_factory as ff
import calendar
from scipy.signal import chirp, find_peaks, peak_widths
import scipy.fftpack                 # discrete Fourier transforms
from scipy.signal import find_peaks, peak_prominences
from scipy.signal import butter, lfilter
from scipy import signal



file_path_root = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_IOTEX/iotex-glove/PD/"
df_dates_file_path = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_IOTEX/iotex-glove/pd_dates_list.csv"
file_path_filenames = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_IOTEX/iotex-glove/file_paths.csv"
# Returns a list of filepaths and date IDs at which the exercises were performed.
def read_iotex_data():
    gc = gs.service_account(filename='/Users/shehjarsadhu/Desktop/carehub-361720-ebee0b4f8dfe.json')
    sh_dates_pid = gc.open_by_url("https://docs.google.com/spreadsheets/d/1e_pihWraVPzhqbrqVT_X-AsDJmLN1rWTaETXN9GPq6w/edit#gid=1243774476")
    ws_dates_pid = sh_dates_pid.worksheet('pd_dates_list')
    df_dates_pid = pd.read_csv(df_dates_file_path) #pd.DataFrame(ws_dates_pid.get_all_records())
    print("df_dates_pid.columns = ",df_dates_pid.columns)
    df_rg_paths = pd.read_csv("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_IOTEX/iotex-glove/rg_file_path.csv")
    sh_lg_paths = gc.open_by_url("https://docs.google.com/spreadsheets/d/168agcyGpMfihuWHku9zvvg6THCxsjmwB0Spnp2psH1Q/edit#gid=1483394103")
    ws_lg_paths = sh_lg_paths.worksheet('lg_file_path')
    df_lg_paths = pd.read_csv(file_path_filenames)
    print("df_lg_paths: ",df_lg_paths.head())
    return df_lg_paths,df_dates_pid,df_rg_paths

df_lg_paths,df_dates_pid,df_rg_paths = read_iotex_data()

# If left hand use following activity codes.
def activity_codes(file_name,activity_code):
    exercise_name = ""
    if file_name == "lg":
        if activity_code == 0:
            exercise_name = "FingerTap"
        if activity_code == 2:
            exercise_name = "CloseGrip"
        if activity_code == 4:
            exercise_name = "HandFlip"
        if activity_code == 6:
            exercise_name = "HoldHands"
        if activity_code == 7:
            exercise_name = "FingerToNose"
        if activity_code == 9:
            exercise_name = "RestingHands"
    return exercise_name

# Chained callback filer by patient ID and Device ID.
@callback(
    Output('dates_id', 'options'),
    Input('participant_id', 'value'))
def dates_dropdown(participant_id):
    global dates_json
    # dates_json['dates'] = df_dates_pid[df_dates_pid["ParticipantList"]==participant_id]["DateList"]
    return df_dates_pid[df_dates_pid["ParticipantList"]==participant_id]["DateList"]

# Second Chained Call. 
# Input is participant_id ,dates_id
@callback(
    Output('task_id', 'options'),
    Input('participant_id', 'value'),
    Input('dates_id', 'value'))
def sessions_dropdown(participant_id, dates_id):
    # Query by PID.
    pid_df = df_lg_paths[df_lg_paths["ParticipantList"] == participant_id]
    dates_query_df = pid_df[pid_df["DateList"] == dates_id]
    return dates_query_df["FileName"]

@callback(
    Output('activity_id', 'options'),
    Input('task_id', 'value'))
def activity_dropdown(task_id):
    lg_code = [0,2,4,6,7,9]
    rg_code = [1,3,5,7,9]
    if task_id.startswith("lg"):
        return lg_code
    if task_id.startswith("rg"):
        return rg_code
# Chained callback filer by patient ID and Device ID.
@callback(
    Output('column_id', 'options'),
    Input('filepath', 'value'),
    Input('participant_id', 'value'),
    Input('dates_id', 'value'),
    Input('task_id','value'),
    )
def column_id_dropdown(filepath,participant_id,dates_id,task_id):
    print("column_id_dropdown(): participant_id,dates_id,task_id,filepath: ",participant_id,dates_id,task_id,filepath)
    print("column_id_dropdown(): system_file_path accepted by program: ",filepath)
    system_file_path = ""
    if filepath != None and dates_id ==None:
        activity_id = 1
        print("update_graph2(): filepath entered by user: ",filepath)

        system_file_path = filepath
    else:

        system_file_path = file_path_root + str(participant_id) + "/" + str(dates_id) + "/"+ str(task_id)
    system_file_path_df = pd.read_csv(system_file_path)
    print("column_id_dropdown(): system_file_path: ",system_file_path)
    print("df columns:", system_file_path_df.columns.values.tolist())    
    return system_file_path_df.columns.values.tolist()

# Displays graphs.
@callback(
    Output('indicator-graphic-iotex1', 'figure'),
    Input('participant_id', 'value'),
    Input('dates_id', 'value'),
    Input('task_id','value'),
    Input('activity_id','value'),
  
    )
def update_graph(pid, dates_id, task_id,activity_id,):
    # Query for specfic patients. 
    #print("update_graph/ pid, dates_id, task_id: ",pid,dates_id,task_id)
    file_path = file_path_root + str(pid) + "/" + str(dates_id) + "/"+ str(task_id)
    df_lg = pd.read_csv(file_path)
    # Query by Activity.
    df_lg_activity = df_lg[df_lg["activity"]==activity_id] 
    exercise_name = activity_codes("lg",activity_id)
    df_dates_pid_p = df_dates_pid[df_dates_pid["ParticipantList"]==pid]
    fig1 = make_subplots(rows=1, cols=1, vertical_spacing = 0.13, # Vertical spacing cannot be greater than (1 / (rows - 1)) = 0.142857.
    subplot_titles=("Participant Adherence: All days")) # subplot_titles=("Participant Adherence " ," <b> Left Glove Activity Name: </b>" + str(exercise_name) + "<br> Index Finger","Middle","Thumb","Accelerometer x","Accelerometer y","Accelerometer z","Influx DB Timestamp"))
    fig1.add_trace(go.Bar( 
        x = df_dates_pid_p["DateList"], y = df_dates_pid_p["NumTasks"],
        text=df_dates_pid_p["NumTasks"],marker_color='teal'), 1, 1)
    large_rockwell_template = dict( 
    layout=go.Layout(title_font=dict(family="Rockwell")))
    # Update the axises
    fig1.update_xaxes(tickvals=df_dates_pid_p["DateList"] ,title_text="Dates , "+ " Total # Sessions : " +str(df_dates_pid_p["NumTasks"].sum()), title_font_family="IBM Plex San",row=1, col=1)
    fig1.update_yaxes(title_text="Number of Times Tasks Were Completed", title_font_family="IBM Plex San",row=1, col=1)
    fig1.update_layout(
        font_family="IBM Plex Sans",
        title= "" ,
        template=large_rockwell_template,height=500, width=1000) 
    return fig1

# Displays graphs.
@callback(
    Output('indicator-graphic-calendar-view', 'figure'),
    Input('participant_id', 'value'),
    Input('dates_id', 'value'),
    Input('task_id','value'),
    Input('activity_id','value'))
def update_graph_calendarview(pid, dates_id, task_id,activity_id):
    # Query for specfic patients.
    file_path = file_path_root + str(pid) + "/" + str(dates_id) + "/"+ str(task_id)
    # Get a  list of dates for the selected participant.
    # print("LIST OF Dates tasks were done- ",df_dates_pid[df_dates_pid["ParticipantList"]==pid]["DateList"])
    # print("LIST OF Dates tasks were done- ",dates_id)
    # df_lg = pd.read_csv(file_path)
    # # Query by Activity.
    # df_lg_activity = df_lg[df_lg["activity"]==activity_id]
    # exercise_name = activity_codes("lg",activity_id)
    # df_dates_pid_p = df_dates_pid[df_dates_pid["ParticipantList"]==pid]
    #Create synthetic data
    z = np.random.randint(0, 16,(12, 31)).astype(object) #needed to asign np.nan to some elems
    # print("Shape of Z: ",z.shape,z[6][25])
    nr_monthdays =[31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    # Initialalize the arrar to assign 0's.
    for k, d in enumerate(nr_monthdays[:7]):
        z[k] = 0
    for k in range(7, 12):
        z[k] = 0
    # Assign 1 if date is present.
    for idx,val in enumerate(df_dates_pid[df_dates_pid["ParticipantList"]==pid]["DateList"]):
        print("val: ",val.split("-"))
        row_idx = int(val.split("-")[1])
        col_idx = int(val.split("-")[2])
        print("row_idx,col_idx : ",row_idx,col_idx)
        if col_idx == 31: # Handle last value index edge case for columns.
            col_idx = 30
            z[row_idx][col_idx] += 1
        if row_idx == 12:  # Handle last value index edge case  for rows,
            row_idx = 11
            z[row_idx][col_idx] += 1 # Assign 1 if date is present.
        z[row_idx][col_idx] += 1 # Assign 1 if date is present.
    # print("MATRIX Z: ",z)
    xlabels = np.arange(1, 32).tolist()  #must be a list, otherwise error at the test if x:
    ylabels  = [name for name in calendar.month_name][1:]
    fig1 = ff.create_annotated_heatmap(z, x=xlabels, y=ylabels, annotation_text=z, colorscale='teal')

    fig1.update_traces(xgap=2, ygap=2, hoverongaps=False)
    fig1.update_layout(template='none')
    fig1.update_yaxes(autorange='reversed', 
                    showgrid=False, 
                    title='Month')
    fig1.update_xaxes(  
                    showgrid=False, 
                    showticklabels=True, 
                    side='top', # Default side is 'top' for annotated heatmap.
                    title='Total Unique Days' +",  " +str(df_dates_pid.shape[0]) + "<br> Total UPDRS Sessions: " + str(sum(df_dates_pid[df_dates_pid["ParticipantList"]==pid]["NumTasks"])))
    large_rockwell_template = dict(
    layout=go.Layout(title_font=dict(family="Rockwell")))
    fig1.update_layout(
        font_family="IBM Plex Sans",
        title= "" ,
        template=large_rockwell_template,height=500, width=1000) 
    return fig1

# Displays graphs.
@callback(
    Output('indicator-graphic-iotex2', 'figure'),
    Input('participant_id', 'value'),
    Input('dates_id', 'value'),
    Input('task_id','value'),
    Input('activity_id','value'),
    # [Input("width", "value")],
    # [Input("heigth", "value")],
    # [Input("prominance", "value")],
    # [Input("max_prominance", "value")],
    [Input("filepath", "value"),
      Input('column_id', 'value')
    ]
    )
def update_graph2(pid, dates_id, task_id,activity_id,filepath,column_id):
    print("update_graph2(): column_id - ",column_id )
    # print("update_graph2(): max_prominance",max_prominance)
    # Query for specfic patients.
    # print("update_graph2(): Width parameter entered by user: ",width,heigth,prominance)
    #print("update_graph2(): filepath entered by user: ",filepath)
    # TO get the correct file paths from either text of selected dropdown.
    system_file_path = ""
    if filepath != None and dates_id ==None:
        activity_id = 1
        print("update_graph2(): filepath entered by user: ",filepath)

        system_file_path = filepath
    else:
       
        system_file_path = file_path_root + str(pid) + "/" + str(dates_id) + "/"+ str(task_id)

    print("update_graph2(): system_file_path accepted by program: ",system_file_path)

    # print("file_path: ",file_path)
    df_lg = pd.read_csv(system_file_path)
    # Sbuset the data  by Activity.
    df_lg_activity = df_lg[df_lg["activity"]==activity_id] #df_lg#df_lg[df_lg["activity"]==activity_id]
    start_idx = 100
    end_idx = start_idx+160 
    df_lg_activity_subset = df_lg_activity[start_idx:end_idx]#df_lg_activity[:500] # windowed data
    print("df_lg_activity_subset.size[0]: ",len(df_lg_activity_subset))
    time_steps = np.linspace(0, 64, len(df_lg_activity_subset))
    print("time_steps: ----",len(time_steps),time_steps)
    normalized_arr = signal.detrend(preprocessing.normalize([df_lg_activity_subset[column_id]])) #df_lg_activity[column_id]#
    indices,properties_peaks = find_peaks(normalized_arr.ravel(), height=np.quantile(normalized_arr, 0.50))
    peaks =[normalized_arr.ravel()[j] for j in indices]
    time_peaks = [time_steps[j] for j in indices]
    results_half = peak_widths(normalized_arr.ravel(), indices, rel_height=0.5)
    # print("update_graph2(): PEAK WIDTHS - results_half: ",results_half[0],type(results_half[0]),len(results_half[0]))
    peak_prominence = peak_prominences(normalized_arr.ravel(), indices)
    # print("update_graph2(): PEAK PROMINANCES - results_half: ",peak_prominence[0],type(peak_prominence[0]),len(peak_prominence[0]))
    # valley_idx, valley_properties_idx = find_peaks(idx_norm*-1, width=(2,20),distance=5)     #  prominence=(0.01, 4)  
    valley_idx, properties_idx = find_peaks(normalized_arr.ravel()*-1,height=np.quantile(normalized_arr, 0.50))#     #  prominence=(0.01, 4) # height=np.quantile(normalized_arr, 0.50) 
    valley_widths = peak_widths(normalized_arr.ravel()*-1, valley_idx, rel_height=0.5)

    # print("valley_idx, properties_idx: ",valley_idx, properties_idx)
    #idx_thubm_norm_arr = preprocessing.normalize([df_lg_activity["index"][30:500]-df_lg_activity["thumb"][30:500]])
    quantiles_normalized_arr = np.quantile(normalized_arr, 0.50) # Threshold - get the 50% quantile for the index finger data.

    # print("update_graph2() len(split_index): ",len(split_index))
    frequency,amplitude,sample_freq = my_fft(normalized_arr)
    #print("len(frequency),len(amplitude):",len(frequency),len(amplitude))
    #print("f,P1: ",frequency,amplitude)
    #print("update_graph2():/ sig_fft_out: ",sig_fft_out)
    high_freq_fft = sample_freq.copy()

    fig2 = make_subplots(rows=2, cols=1, vertical_spacing = 0.13,subplot_titles=(
        #str(column_id)+": Number of Peaks- " +str(len(indices)) + "<br>" + "Width: " + str(width) + " Distance: "+ str(heigth) + " Prominance: "+ str(prominance),"Filtered Signal"," Peak Widthds <br> " + "Mean: "+str(results_half[0].mean()),"Peak Prominances <br> Mean: " +str(peak_prominence[0].mean()),"Valley Widths Mean: " + str(valley_widths[0].mean()),"Raw FFT","Filtered FFT "
        )
        )
        #signal.detrend(normalized_arr.ravel())
    fig2.add_trace(go.Scatter(y = normalized_arr.ravel() ,marker_color='teal'), 1, 1)
    fig2.add_trace(go.Scatter(
    x=time_peaks,
    y=peaks,
    mode='markers',
    marker=dict(
        size=5,
        color='blue',
        symbol='circle'
    ),
        name='Detectd Peaks'
    ),1,1),
    # fig2.add_trace(go.Scatter(
    # x=valley_idx,
    # y=[normalized_arr.ravel()[j] for j in valley_idx],
    # mode='markers',
    # marker=dict(
    #     size=5,
    #     color='red',
    #     symbol='circle'
    # ),
    #     name='Detectd valleys'
    # ),1,1)
    # indices
   
    # fig2.add_trace(go.Scatter(y = filtered_sig.real,marker_color='teal'), 2, 1)
    # fig2.add_trace(go.Scatter(
    # x=filt_indices,
    # y=filt_peaks,
    # mode='markers',
    # marker=dict(
    #     size=5,
    #     color='blue',
    #     symbol='circle'
    # ),
    #     name='Detectd Peaks'
    # ),2,1)
    # fig2.add_trace(go.Scatter(y =normalized_arr.ravel(),marker_color='teal',name=""), 3, 1)
    # fig2.add_trace(go.Scatter(
    #     x=filt_indices,
    #     y=[normalized_arr.ravel()[j] for j in filt_indices],
    #     mode='markers',
    #     marker=dict(
    #         size=5,
    #         color='blue',
    #         symbol='circle'
    #     ),
    #         name='Detectd Peaks'
    #     ),3,1)
    fig2.add_trace(go.Bar(x=np.arange(0,len(results_half[0])), y=results_half[0],text=results_half[0],
            textposition='auto',marker_color='blue', name='Detectd Peaks (Peak Width) , Mean: ' + str(results_half[0].mean())), 2, 1)
    #fig2.add_trace(go.Bar(x=np.arange(0,len(peak_prominence[0])), y=peak_prominence[0],text=peak_prominence[0],
    #        textposition='auto',marker_color='blue',name='Detectd Peaks'), 5, 1)
    # fig2.add_trace(go.Bar(x=np.arange(0,len(valley_widths[0])), y=valley_widths[0],text=valley_widths[0],
    #         textposition='auto',marker_color='red',name='Detectd Valleys'), 6, 1)
    # fig2.add_trace(go.Scatter(x = filtered_frequency,y = filtered_amplitude,marker_color='teal'), 4, 1)
    # fig2.add_trace(go.Scatter(x = filt_f,y = filt_P1,marker_color='teal'), 5, 1)
    # fig2.update_yaxes(title_text="Voltage", title_font_family="IBM Plex San", row=1, col=1)
    # fig2.update_xaxes(title_text="Number of Samples",row=1, col=1)
    # fig2.update_yaxes(title_text="Peak Width", title_font_family="IBM Plex San", row=3, col=1)
    # fig2.update_xaxes(title_text="Number of Peaks",row=2, col=1)
    # fig2.update_yaxes(title_text="Peak Prominance", title_font_family="IBM Plex San", row=4, col=1)
    # fig2.update_xaxes(title_text="Number of peaks",row=3, col=1)
    
    # fig2.update_yaxes(title_text="Number of Peaks", title_font_family="IBM Plex San", row=5, col=1)
    # fig2.update_xaxes(title_text="Frequency in Hz",row=4, col=1)
    large_rockwell_template = dict(
    layout=go.Layout(title_font=dict(family="Rockwell")))
    fig2.update_layout(
        font_family="IBM Plex Sans",
        title= "" ,
        template=large_rockwell_template,height=500, width=1000) 
    return fig2


# Displays graphs.
@callback(
    Output('indicator-graphic-iotex3', 'figure'),
    Input('participant_id', 'value'),
    Input('dates_id', 'value'),
    Input('task_id','value'),
    Input('activity_id','value'),
    [Input("width", "value")],
    [Input("heigth", "value")],
    [Input("prominance", "value")],
    [Input("max_prominance", "value")],
    [Input("filepath", "value"),
      Input('column_id', 'value')
    ]
    )
## Accelerometer Plots.
def update_graph3(pid, dates_id, task_id,activity_id,width,heigth,prominance,max_prominance,filepath,column_id):
    print("update_graph3(): column_id - ",column_id )
    print("update_graph3(): max_prominance",max_prominance)
    # Query for specfic patients.
    print("update_graph3(): Width parameter entered by user: ",width,heigth,prominance)
    #print("update_graph2(): filepath entered by user: ",filepath)
    # TO get the correct file paths from either text of selected dropdown.
    system_file_path = ""
    if filepath != None and dates_id ==None:
        activity_id = 1
        print("update_graph3(): filepath entered by user: ",filepath)

        system_file_path = filepath
    else:
       
        system_file_path = file_path_root + str(pid) + "/" + str(dates_id) + "/"+ str(task_id)

    print("update_graph3(): system_file_path accepted by program: ",system_file_path)

    # print("file_path: ",file_path)
    df_lg = pd.read_csv(system_file_path)
    # Sbuset the data  by Activity.
    df_lg_activity = df_lg[df_lg["activity"]==activity_id] #df_lg#df_lg[df_lg["activity"]==activity_id]
    start_idx = 100
    end_idx = start_idx+160 
    df_lg_activity_subset = df_lg_activity[start_idx:end_idx]#df_lg_activity[:500] # windowed data
   # print("df_lg_activity_subset.head() =  ",df_lg_activity_subset.head(),df_lg_activity_subset.shape)
    # exercise_name = activity_codes("lg",activity_id)
    # df_dates_pid_p = df_dates_pid[df_dates_pid["ParticipantList"]==pid]
   
    normalized_arr = preprocessing.normalize([df_lg_activity_subset[column_id]]) #df_lg_activity[column_id]#
    #print("update_graph3() : normalized_arr: ",normalized_arr,len(normalized_arr))
    split_index = np.array_split(normalized_arr[0], 3)
    #print("update_graph3() len(split_index): ",split_index,len(split_index),len(split_index[0]),len(split_index[1]),len(split_index[2]),type(split_index[0]))
    print()
    print("=====================split_index[0]=====================")
    f1,P1 = my_fft_split(split_index[0])
    indices1,properties_peaks1 = find_peaks(split_index[0], height=np.quantile(normalized_arr, 0.50) ,prominence=(prominance, max_prominance) ,width=width,distance= heigth)
    peaks1 = [split_index[0][j] for j in indices1]

    valley_idx1, properties_idx = find_peaks(split_index[0]*-1,prominence=(prominance, 4) ,width=(2,width),distance=heigth)#     #  prominence=(0.01, 4) # height=np.quantile(normalized_arr, 0.50) 
    valley1 = [split_index[0][j] for j in valley_idx1]

    print("=====================split_index[1]=====================")
    f2,P2 = my_fft_split(split_index[1])
    indices2,properties_peaks2 = find_peaks(split_index[1], height=np.quantile(normalized_arr, 0.50) ,prominence=(prominance, max_prominance) ,width=width,distance= heigth)
    peaks2 = [split_index[1][j] for j in indices2]
    valley_idx2, properties_idx = find_peaks(split_index[1]*-1,prominence=(prominance, 4) ,width=(2,width),distance=heigth)#     #  prominence=(0.01, 4) # height=np.quantile(normalized_arr, 0.50) 
    valley2 = [split_index[1][j] for j in valley_idx2]

    print("=====================split_index[2]=====================")
    f3,P3 = my_fft_split(split_index[2])
    indices3,properties_peaks3 = find_peaks(split_index[2], height=np.quantile(normalized_arr, 0.50) ,prominence=(prominance, max_prominance) ,width=width,distance= heigth)
    peaks3 = [split_index[2][j] for j in indices3]

    valley_idx3, properties_idx = find_peaks(split_index[2]*-1,prominence=(prominance, 4) ,width=(2,width),distance=heigth)#     #  prominence=(0.01, 4) # height=np.quantile(normalized_arr, 0.50) 
    valley3 = [split_index[2][j] for j in valley_idx3]

    fig3 = make_subplots(rows=3, cols=3, vertical_spacing = 0.13, # Vertical spacing cannot be greater than (1 / (rows - 1)) = 0.142857.
    subplot_titles = ("Index Raw 1","Index Raw 2","Index Raw 3","FFT 1","FFT 2","FFT 3")) 

    fig3.add_trace(go.Scatter(y = split_index[0],marker_color='teal'), 1, 1)
    fig3.add_trace(go.Scatter(x=valley_idx1, y=valley1,
    mode='markers',
    marker=dict(
        size=5,
        color='red',
        symbol='circle'
    ),
        name='Detectd valleys'
    ),1,1)
    # indices
    fig3.add_trace(go.Scatter(
    x=indices1,
    y=peaks1,
    mode='markers',
    marker=dict(
        size=5,
        color='blue',
        symbol='circle'
    ),
        name='Detectd Peaks'
    ),1,1)
    fig3.add_trace(go.Scatter(y = split_index[1],marker_color='teal'), 1, 2)
    fig3.add_trace(go.Scatter(x=valley_idx2, y=valley2,
    mode='markers',
    marker=dict(
        size=5,
        color='red',
        symbol='circle'
    ),
        name='Detectd valleys'
    ),1,2)
    # indices
    fig3.add_trace(go.Scatter(
    x=indices2,
    y=peaks2,
    mode='markers',
    marker=dict(
        size=5,
        color='blue',
        symbol='circle'
    ),
        name='Detectd Peaks'
    ),1,2)


    fig3.add_trace(go.Scatter(y = split_index[2],marker_color='teal'), 1, 3)
  
    fig3.add_trace(go.Scatter(x=valley_idx3, y=valley3,
    mode='markers',
    marker=dict(
        size=5,
        color='red',
        symbol='circle'
    ),
        name='Detectd valleys'
    ),1,3)
    # indices
    fig3.add_trace(go.Scatter(
    x=indices3,
    y=peaks3,
    mode='markers',
    marker=dict(
        size=5,
        color='blue',
        symbol='circle'
    ),
        name='Detectd Peaks'
    ),1,3)

    fig3.add_trace(go.Scatter(x = f1,y=P1,marker_color='teal'), 2, 1)
    fig3.add_trace(go.Scatter(x = f2,y=P2,marker_color='teal'), 2, 2)
    fig3.add_trace(go.Scatter(x = f3,y=P3,marker_color='teal'), 2, 3)

    return fig3

@callback(
    Output("out-all-types", "children"),
    [Input("width", "value")],
    [Input("heigth", "value")],
    [Input("prominance", "value")]
)
def cb_render(width,heigth,prominance):
    return u'Peak Width: {}  Peak Height: {} Peak Prominance: {}'.format(width,heigth,prominance)


def butter_highpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype = "high", analog = False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=2):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

   
# Takes in a numpy array and calculates fft using scipi .
# Returns: frequency component - f and P1: - the apmplitude. 
def my_fft(fft_data):
    Fs = 64 # Sampling Rate 
    T = 1/Fs
    L =len(fft_data) # Length of the dataset.
    #print("my_fft(): ",L)
    t = np.arange(start=0, stop=L-1, step=1) *T # Timestamp bins.
    f = np.arange(start=0, stop=L-1, step=1)/L * Fs # 
    #print("my_fft(): ",fft_data, type(fft_data))
    sig_fft_out = scipy.fftpack.fft(fft_data)
    # The corresponding frequencies
   # print("sig_fft_out: ",len(sig_fft_out))

    P2 = abs(sig_fft_out/L)
   # print("my_fft(fft_data): P2 - abs(sig_fft_out/L) ", P2,len(P2))
   # print("my_fft(fft_data): int(L/2+1): ",int(L/2+1))
    P1 = P2[1:int(L)]
    #print("my_fft(): P1: ",len(P1),P1)
    P1[2:len(P1)-1] = 2*P1[2:len(P1)-1]
    return f,P1,sig_fft_out

def my_fft_split(fft_data):
    Fs = 64 # Sampling Rate 
    T = 1/Fs
    L =len(fft_data) # Length of the dataset.
    #print("my_fft(): ",L)
    t = np.arange(start=0, stop=L-1, step=1) *T # Timestamp bins.
    f = np.arange(start=0, stop=L/2, step=1)/L * Fs # 
    #print("my_fft_split(): ",type(fft_data))
    Y = scipy.fftpack.fft(fft_data)
    P2 = abs(Y/L)
    #print("my_fft_split(): P2: ",P2,P2.size)
    P1 = P2[1:int(L/2+1)]
    #print("my_fft_split(): P1: ",P1)
    P1[2:len(P1)-1] = 2*P1[2:len(P1)-1]
    return f,P1
################################ -------------------- RISE TIME ----------------- ################################################################################################
# Params - Peak indexes and valley indexes returned by  peaks_and_valleys().
def rise_time(peaks_idxs,valley_idxs,df):
    ##### Params needed (1) valley indexes, (2) peak indexes
    # Psudo Code
    rise_time_arr = [] # Len of this array ahould be the same as number of peaks. 
    found_valley_to_peak = 0
    valle_done = 0 # for breaking if one iter of valleys is done.
    for v_idx, valley_idx in enumerate(valley_idxs):
        for idx, peaks_idx in enumerate(peaks_idxs):
            #print("peak to valley peaks_idxs[idx]  < valley_idxs[v_idx]:", peaks_idxs[idx]  , valley_idxs[v_idx])
            #print("valley to peak rise time :", valley_idxs[v_idx], peaks_idxs[idx], peaks_idxs[len(peaks_idxs)-1])  
            if valley_idxs[v_idx] < peaks_idxs[idx]: # RISE TIME
                print("rise_time(): idx,valley_idxs: ",idx,v_idx)
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
    return rise_time_arr