import dash
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

# Read data 
# Returns a list of filepaths and date IDs at which the exercises were performed.
def read_iotex_data():
    gc = gs.service_account(filename='/Users/shehjarsadhu/Desktop/carehub-361720-ebee0b4f8dfe.json')
    sh_dates_pid = gc.open_by_url("https://docs.google.com/spreadsheets/d/1e_pihWraVPzhqbrqVT_X-AsDJmLN1rWTaETXN9GPq6w/edit#gid=1243774476")
    ws_dates_pid = sh_dates_pid.worksheet('pd_dates_list')
    df_dates_pid = pd.DataFrame(ws_dates_pid.get_all_records())

    sh_lg_paths = gc.open_by_url("https://docs.google.com/spreadsheets/d/168agcyGpMfihuWHku9zvvg6THCxsjmwB0Spnp2psH1Q/edit#gid=1483394103")
    ws_lg_paths = sh_lg_paths.worksheet('lg_file_path')
    df_lg_paths = pd.DataFrame(ws_lg_paths.get_all_records())
    print("df_lg_paths: ",df_lg_paths.head())
    return df_lg_paths,df_dates_pid

df_lg_paths,df_dates_pid = read_iotex_data()

app_iotex_layout = html.Div([
    html.Div([
        html.Header(
            "IoTex Longitudinal Parkinsons Disease Dataset",style={ 'font-family': 'IBM Plex Sans','color':'white','font-size': '50pt'}
        ),
        html.Div([
            # dbc.Container(dbc.Row(dbc.Col([navbar])), id="nav_bar"),
            html.Br(),
            dbc.Row(style = {"align":"center"}, children=[
                dbc.Col([
                    dcc.Dropdown(
                        id='participant_id',
                        options=  [{'label': i, 'value': i} for i in df_dates_pid["ParticipantList"].unique()],
                        placeholder="Select Device ID ",
                        value = df_dates_pid["ParticipantList"].unique()[0],
                        # style = {"align":"center"}
                        ), html.Br()]),
                dbc.Col([
                        dcc.Dropdown(
                                id='task_id',
                                options=  [],
                                placeholder="Select Task ",
                            ),  
                            html.Br()])
                    ]),
            dbc.Row(style = {"align":"center"}, children=[
                #col1
                dbc.Col([
                     dcc.Dropdown(
                            id='dates_id',
                            options=  [],
                            placeholder="Select Patient ",
                            
                        ),
                    html.Br()]
                ),
                #col2
                dbc.Col([

                dcc.Dropdown(
                            id='activity_id',
                            options = [],
                            placeholder = "Activity Codes",    
                            ),
                html.Br(),
                ])
            ])
        ],
        style={'width': '48%', 'display': 'inline-block'}),
    ]),
    dcc.Graph(id='indicator-graphic-iotex'),
    html.Div(id='iotex_dash')
])
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
    return df_dates_pid[df_dates_pid["ParticipantList"]==participant_id]["DateList"]

# Second Chained Call.
@callback(
    Output('task_id', 'options'),
    Input('participant_id', 'value'),
    Input('dates_id', 'value'))
def sessions_dropdown(participant_id, dates_id):
    # Query by PID.
    pid_df = df_lg_paths[df_lg_paths["ParticipantList"] == participant_id]
    dates_query_df = pid_df[pid_df["DateList"] == dates_id]
    return dates_query_df["FileName_LeftGlove"]

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

# Displays graphs.
@callback(
    Output('indicator-graphic-iotex', 'figure'),
    Input('participant_id', 'value'),
    Input('dates_id', 'value'),
    Input('task_id','value'),
    Input('activity_id','value'))
def update_graph(pid, dates_id, task_id,activity_id):
    # Query for specfic patients.
    #print("update_graph/ pid, dates_id, task_id: ",pid,dates_id,task_id)
    file_path = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_IOTEX/iotex-glove/PD/" + str(pid) + "/" + str(dates_id) + "/"+ str(task_id)
    df_lg = pd.read_csv(file_path)
    # Query by Activity.
    df_lg_activity = df_lg[df_lg["activity"]==activity_id]
    exercise_name = activity_codes("lg",activity_id)
    #print("update_graph/exercise_name",exercise_name)
    #print("update_graph/ file_path = \n",file_path,)
    df_dates_pid_p = df_dates_pid[df_dates_pid["ParticipantList"]==pid]
    fig = make_subplots(rows=8, cols=1, #vertical_spacing = 0.11
    subplot_titles=("Participant Adherence " ," <b> Left Glove Activity Name: </b>" + str(exercise_name) + "<br> Index Finger","Middle","Thumb","Accelerometer x","Accelerometer y","Accelerometer z","Influx DB Timestamp"))
    fig.add_trace(go.Bar(
        x = df_dates_pid_p["DateList"], y = df_dates_pid_p["NumTasks"],
        text=df_dates_pid_p["NumTasks"],marker_color='teal'), 1, 1)
    fig.add_trace(go.Scatter(y = df_lg_activity["index"],marker_color='teal'), 2, 1)
    fig.add_trace(go.Scatter(y = df_lg_activity["middle"],marker_color='teal'), 3, 1)
    fig.add_trace(go.Scatter(y = df_lg_activity["thumb"],marker_color='teal'), 4, 1)
    fig.add_trace(go.Scatter(y = df_lg_activity["ax"],marker_color='teal'), 5, 1)
    fig.add_trace(go.Scatter(y = df_lg_activity["ay"],marker_color='teal'), 6, 1)
    fig.add_trace(go.Scatter(y = df_lg_activity["az"],marker_color='teal'), 7, 1)
    fig.add_trace(go.Scatter(y = df_lg_activity["time"],marker_color='teal'), 8, 1)

    large_rockwell_template = dict(
    layout=go.Layout(title_font=dict(family="Rockwell")))
    fig.update_xaxes( tickvals=df_dates_pid_p["DateList"] ,title_text="Dates , "+ " Total # Sessions : " +str(df_dates_pid_p["NumTasks"].sum()), title_font_family="IBM Plex San",row=1, col=1)
    # fig.update_yaxes(title_text="Number of Times Tasks Were Completed", title_font_family="IBM Plex San",row=1, col=1)
    # fig.update_xaxes(title_text="Frequency in Hz <br>", title_font_family="IBM Plex San",row=2, col=1)
    # fig.update_yaxes(title_text="Amplitude", title_font_family="IBM Plex San", row=4, col=1)
    # fig.update_xaxes(title_text="Number of Samples <br>", title_font_family="IBM Plex San",row=3, col=1)
    # fig.update_xaxes(title_text="Frequency in Hz <br>", title_font_family="IBM Plex San",row=4, col=1)

    fig.update_layout(
        font_family="IBM Plex Sans",
        title= "" ,
        template=large_rockwell_template,height=1000, width=1300) 
    return fig