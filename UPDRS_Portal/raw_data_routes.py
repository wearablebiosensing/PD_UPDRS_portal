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
    # Input('dates_id', 'value'),
    # Input('task_id','value'),
    # Input('activity_id','value'),
    Input('filepath', 'value'),
    
    )
def column_id_dropdown(filepath):
    print("column_id_dropdown(): system_file_path accepted by program: ",filepath)

    df = pd.read_csv(filepath)
    print("df columns:", df.columns.values.tolist() )    
    return df.columns.values.tolist()

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
    [Input("width", "value")],
    [Input("heigth", "value")],
    [Input("prominance", "value")],
    [Input("filepath", "value"),
      Input('column_id', 'value')
    ]
    )
def update_graph2(pid, dates_id, task_id,activity_id,width,heigth,prominance,filepath,column_id):
    print("update_graph2(): column_id - ",column_id )
    # Query for specfic patients.
    print("update_graph2(): Width parameter entered by user: ",width,heigth,prominance)
    #print("update_graph2(): filepath entered by user: ",filepath)
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
    # Query by Activity.
    df_lg_activity = df_lg#df_lg[df_lg["activity"]==activity_id] #df_lg#df_lg[df_lg["activity"]==activity_id]
    exercise_name = activity_codes("lg",activity_id)
    df_dates_pid_p = df_dates_pid[df_dates_pid["ParticipantList"]==pid]
    normalized_arr = df_lg_activity[column_id][25:500]#preprocessing.normalize([df_lg_activity["index"][25:500]])
    indices,properties_peaks = find_peaks(normalized_arr.ravel(), prominence=(prominance, 4) ,width=(2,width),distance=heigth)
    print("properties_peaks: ",properties_peaks) 
    # valley_idx, valley_properties_idx = find_peaks(idx_norm*-1, width=(2,20),distance=5)     #  prominence=(0.01, 4)  
    valley_idx, properties_idx = find_peaks(normalized_arr.ravel()*-1,prominence=(prominance, 4) ,width=(2,width),distance=heigth)#     #  prominence=(0.01, 4)  
    # print("valley_idx, properties_idx: ",valley_idx, properties_idx)
    #idx_thubm_norm_arr = preprocessing.normalize([df_lg_activity["index"][30:500]-df_lg_activity["thumb"][30:500]])
    quantiles_normalized_arr = np.quantile(normalized_arr, 0.50) # Threshold - get the 50% quantile for the index finger data.
    rise_time_each_peak = rise_time(indices,valley_idx,df_lg_activity)
    
    # print("normalized_arr: ",normalized_arr.ravel(), type(normalized_arr))
    # idx_thubm_norm_arr_quatiles = idx_thubm_norm_arr.quantile(q=[0.25, 0.5, 0.75], axis=0, numeric_only=True)
    #quantiles_idx_thubm_norm_arr= np.quantile(idx_thubm_norm_arr, 0.50)
    # print("quantiles_idx_thubm_norm_arr: ",quantiles_idx_thubm_norm_arr)
    # Get the indicies of the peaks for plotting. Index only 
    # This only works fro P1 indices = find_peaks(normalized_arr.ravel(), height=quantiles_normalized_arr)[0]
    # # for P3 - width=4# prom_val= (0.01, None), width_val = (2,20) ,Dist_val=5
    # indices_idx_thumb = find_peaks(idx_thubm_norm_arr.ravel(), height=quantiles_idx_thubm_norm_arr)[0]
  
    # dist = np.linalg.norm(df_lg_activity["index"][:590]-df_lg_activity["thumb"][:590])
    # dist = np.sqrt(np.sum([(a-b)*(a-b) for a, b in zip(df_lg_activity["index"][:590], df_lg_activity["thumb"][:590])]))    
    #index_minus_thumb = df_lg_activity["index"][30:500]-df_lg_activity["thumb"][30:500]
    # index_minus_thumb_threshold = index_minus_thumb.diff().ravel().mean() + index_minus_thumb.diff().ravel().std()
    # indices_index_minus_thumb = find_peaks(index_minus_thumb.diff().ravel(), height=index_minus_thumb_threshold)[0]
    

    # Vertical spacing cannot be greater than (1 / (rows - 1)) = 0.142857.
    fig2 = make_subplots(rows=5, cols=1, vertical_spacing = 0.13,subplot_titles=(str(column_id)+": Number of Peaks- " +str(len(indices)) + "<br>" + "Width: " + str(width) + " Distance: "+ str(heigth) + " Prominance: "+ str(prominance),"Rise Times in (seconds)"))
    fig2.add_trace(go.Scatter(y = normalized_arr.ravel(),marker_color='teal'), 1, 1)
    fig2.add_trace(go.Scatter(
    x=valley_idx,
    y=[normalized_arr.ravel()[j] for j in valley_idx],
    mode='markers',
    marker=dict(
        size=5,
        color='red',
        symbol='circle'
    ),
        name='Detectd valleys'
    ),1,1)
    # indices
    fig2.add_trace(go.Scatter(
    x=indices,
    y=[normalized_arr.ravel()[j] for j in indices],
    mode='markers',
    marker=dict(
        size=5,
        color='blue',
        symbol='circle'
    ),
        name='Detectd Peaks'
    ),1,1)
    fig2.add_trace(go.Bar(x=np.arange(start=0, stop=len(rise_time_each_peak)+1, step=1),
                y=rise_time_each_peak,
                name='Rest of world',
                marker_color='rgb(55, 83, 109)'
                ), 2, 1)

    # normalized_arr = preprocessing.normalize([df_lg_activity["index"][30:500]])
    # #prom_val= (0.01, None), width_val = (2,20) ,Dist_val=5,detect_theshold=0
    # valley_idx, properties_idx 
    # fig2.add_trace(go.Scatter(y = df_lg_activity["middle"][30:500],marker_color='teal'), 2, 1)
    # fig2.add_trace(go.Scatter(y = df_lg_activity["thumb"][30:500],marker_color='teal'), 3, 1)     
    # fig2.add_trace(go.Scatter(y = preprocessing.normalize([df_lg_activity["index"][30:500]-df_lg_activity["thumb"][30:500]]).ravel(),marker_color='teal'), 4, 1)
    # fig2.add_trace(go.Scatter(y = index_minus_thumb.diff(),marker_color='teal'), 5, 1)     
    fig2.update_yaxes(title_text="Voltage", title_font_family="IBM Plex San", row=1, col=1)

    large_rockwell_template = dict(
    layout=go.Layout(title_font=dict(family="Rockwell")))
    fig2.update_layout(
        font_family="IBM Plex Sans",
        title= "" ,
        template=large_rockwell_template,height=900, width=1000) 
    return fig2

# Displays graphs.
@callback(
    Output('indicator-graphic-iotex3', 'figure'),
    Input('participant_id', 'value'),
    Input('dates_id', 'value'),
    Input('task_id','value'),
    Input('activity_id','value'),
    )
def update_graph3(pid, dates_id, task_id,activity_id):
    file_path = file_path_root + str(pid) + "/" + str(dates_id) + "/"+ str(task_id)
    df_lg = pd.read_csv(file_path)
    # Query by Activity.
    df_lg_activity = df_lg[df_lg["activity"]==activity_id]
    exercise_name = activity_codes("lg",activity_id)
    #print("update_graph/exercise_name",exercise_name)
    #print("update_graph/ file_path = \n",file_path,)
    df_dates_pid_p = df_dates_pid[df_dates_pid["ParticipantList"]==pid]

    fig3 = make_subplots(rows=3, cols=1, vertical_spacing = 0.13, # Vertical spacing cannot be greater than (1 / (rows - 1)) = 0.142857.
    subplot_titles = ("Accelerometer x","Accelerometer y","Accelerometer z")) # subplot_titles=("Participant Adherence " ," <b> Left Glove Activity Name: </b>" + str(exercise_name) + "<br> Index Finger","Middle","Thumb","Accelerometer x","Accelerometer y","Accelerometer z","Influx DB Timestamp"))

    fig3.add_trace(go.Scatter(y = df_lg_activity["ax"],marker_color='teal'), 1, 1)
    fig3.add_trace(go.Scatter(y = df_lg_activity["ay"],marker_color='teal'), 2, 1)
    fig3.add_trace(go.Scatter(y = df_lg_activity["az"],marker_color='teal'), 3, 1)


    large_rockwell_template = dict(
    layout=go.Layout(title_font=dict(family="Rockwell")))
    # Update the axises
    fig3.update_xaxes(title_text="Number of Samples <br>", title_font_family="IBM Plex San",row=3, col=1)
    fig3.update_xaxes(title_text="Frequency in Hz <br>", title_font_family="IBM Plex San",row=4, col=1)
    fig3.update_layout(
        font_family="IBM Plex Sans",
        title= "" ,
        template=large_rockwell_template,height=500, width=1000) 
    return fig3

@callback(
    Output("out-all-types", "children"),
    [Input("width", "value")],
    [Input("heigth", "value")],
    [Input("prominance", "value")]
)
def cb_render(width,heigth,prominance):
    return u'Peak Width: {}  Peak Height: {} Peak Prominance: {}'.format(width,heigth,prominance)

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