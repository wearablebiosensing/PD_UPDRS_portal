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
from raw_data_routes import df_lg_paths,df_dates_pid,df_rg_paths

# Read data 
app_iotex_layout = html.Div([
    html.Div([
        html.Header(
            "IoTex Longitudinal Parkinsons Disease Dataset",style={ 'font-family': 'IBM Plex Sans','color':'white','font-size': '50pt'}
        ),
        html.Div([
            html.Br(),
            dbc.Row(style = {"align":"center"}, children=[
                dbc.Col([
                    dcc.Dropdown(
                        id='participant_id',
                        options=  [{'label': i, 'value': i} for i in df_dates_pid["ParticipantList"].unique()],
                        placeholder="Select Device ID ",
                        # Gets all the unique participants. 
                        value = df_dates_pid["ParticipantList"].unique()[0],
                        ), html.Br()]),
                dbc.Col([
                        dcc.Dropdown(
                                id='task_id',
                                options=  [],
                                placeholder="Select Glove",
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
    dcc.Graph(id='indicator-graphic-iotex1'),
    dcc.Graph(id='indicator-graphic-calendar-view'),
    dcc.Graph(id='indicator-graphic-iotex2'),
    dcc.Graph(id='indicator-graphic-iotex3'),
    html.Div(id='iotex_dash')
])
