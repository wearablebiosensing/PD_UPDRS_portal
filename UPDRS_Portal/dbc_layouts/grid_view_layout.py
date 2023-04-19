from dash import Input, Output, dcc, html,dash_table
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from color_variables import *
from dbc_styles.grid_view_styles import *
from raw_data_routes import df_lg_paths,df_dates_pid


#######################################################
# This view gets the Calander view for each exercise. #
#######################################################

# # Chained callback filer by patient ID and Dates ID.
@callback(
    Output('dates_output', 'children'),
    Input('participant_id_grid_view', 'value'))
def dates_dropdown_ft(participant_id_grid_view):
    #     #df_dates_pid[df_dates_pid["ParticipantList"]==participant_id]["DateList"]
    dates_json  = {'dates':[]}
    dates_json['dates'] = df_dates_pid[df_dates_pid["ParticipantList"]==participant_id_grid_view]["DateList"]
    print("dates_dropdown_ft(): Dates callback values:",dates_json)
    return  dates_json



columns_seven = [] # Create a list of 7 columns.
columns_seven_content = []

# Creates a column of 7 rows.
for index,val in enumerate(range(7)):
    columns_seven_content.append(val)
    columns_seven.append(dbc.Col(html.Div(columns_seven_content[val],
               ),style=calendar_view_style,width=1)) #  (width=1) Check the width parameter for sizing columns : https://dash-bootstrap-components.opensource.faculty.ai/docs/components/layout/

calander_view = html.Div(
    [
        dcc.Store(id='dates_value_grid_view'),
         dcc.Dropdown(
                        id='participant_id_grid_view',
                        options=  [{'label': i, 'value': i} for i in df_dates_pid["ParticipantList"].unique()],
                        placeholder="Select Device ID ",
                        value = df_dates_pid["ParticipantList"].unique()[0],
                        style=drop_down_styles 
                        # style = {"align":"center"}
                        ),
        # For calendar view dropdown.
        dcc.Dropdown(['January', 'February', 'March', 'April','May','June','July','August','September','October', 'November','December'], style=drop_down_styles ,id='demo-dropdown'),
    # dash_table.DataTable(
    #         id='table_options',
    #         columns=[]
    #     ),
        html.Div(id='dd-output-container'),
        html.Div(id='dates_output'),

        dbc.Row( # Header of the calander view for displaying the days.
            [ dbc.Col(html.Div("Sunday",
                ),style=calendar_view_header_style,width=1),
            dbc.Col(html.Div("Monday",
               ),style=calendar_view_header_style,width=1),
            dbc.Col(html.Div("Tuesday",
                ),style=calendar_view_header_style,width=1),
            dbc.Col(html.Div("Wednesday",
                ),style=calendar_view_header_style,width=1),
            dbc.Col(html.Div("Thursday",
                ),style=calendar_view_header_style,width=1),
            dbc.Col(html.Div("Friday",
               ),style=calendar_view_header_style,width=1),
            dbc.Col(html.Div("Saturday",
               ),style=calendar_view_header_style,width=1),
            ],
            style = rows_style
        ),
        dbc.Row( # Days Rows
            columns_seven,
            style = rows_style,
            ),
        dbc.Row( # Days Rows
            columns_seven,
            style = rows_style
            ),
        dbc.Row( # Days Rows
           columns_seven,style = rows_style
        ),
        dbc.Row( # Days Rows
            columns_seven,style =rows_style
        ),
        dbc.Row( # Days Rows
           columns_seven,style = rows_style
        ),

    ],
    style = {'align' : 'center'}
)
