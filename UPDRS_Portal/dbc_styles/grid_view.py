import dash
from dash import Input, Output, dcc, html
import dash_bootstrap_components as dbc
from color_variables import *

calendar_view_style = {'color': grid_view_text, 
                'fontSize': 14,
                  'padding': '0x',
                  'border-radius': 5, #  'border':'1px solid',
                'background-color': grid_view_bg_background,
                'margin-right': '10px',
                'margin-left': '2px',
                'no_gutters':'True',
        'height': 'auto', 'width':'auto'
                }
calendar_view_header_style = {
                'color': 'black', 
                'fontSize': 14,
                  'padding': '0x',
                  'border-radius': 5, 
                  'margin-right': '10px',
                'margin-left': '2px',
                'no_gutters':'True',
                 
}
columns_seven = [] # create a list of 7 columns.
columns_seven_content = []
for index,val in enumerate(range(7)):
    columns_seven_content.append(val)
    columns_seven.append(dbc.Col(html.Div(columns_seven_content[val],
                style = {'height': '50px','width':'50px'}),style=calendar_view_style))

row = html.Div(
    [
        dbc.Row( # header of the calander view for displaying the days.
            [
            dbc.Col(html.Div("Sunday",
                style = {'height': '80px', 'width':'50px'}),style=calendar_view_header_style),
            dbc.Col(html.Div("Monday",
                style = {'height': '80px', 'width':'50px'}),style=calendar_view_header_style),
            dbc.Col(html.Div("Tuesday",
                style = {'height': '80px', 'width':'50px'}),style=calendar_view_header_style),
            dbc.Col(html.Div("Wednesday",
                style = {'height': '80px', 'width':'50px'}),style=calendar_view_header_style),
            dbc.Col(html.Div("Thursday",
                style = {'height': '80px', 'width':'50px'}),style=calendar_view_header_style),
            dbc.Col(html.Div("Friday",
                style = {'height': '80px', 'width':'50px'}),style=calendar_view_header_style),
            dbc.Col(html.Div("Saturday",
                style = {'height': '80px', 'width':'50px'}),style=calendar_view_header_style),
            ],
        style = {'width':"auto",'margin-bottom': '10px'}
        ),

        dbc.Row(
            columns_seven,
            style = {'width':"auto",'margin-bottom': '10px'}
            ),
        dbc.Row(
            columns_seven,
            style = {'width':"auto",'margin-bottom': '10px'}
            ),
        dbc.Row(
           columns_seven,style = {'width':"auto",'margin-bottom': '10px'}
            # align="center",
        ),
        dbc.Row(
            columns_seven,style = {'width':"auto",'margin-bottom': '10px'}
            # align="end",
        ),
        dbc.Row(
           columns_seven,style = {'width':"auto",'margin-bottom': '10px'}
        ),
    ],
)