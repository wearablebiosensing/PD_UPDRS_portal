import dash
from dash import Input, Output, dcc, html
import dash_bootstrap_components as dbc
from dbc_styles.sidebar_styles import *

sidebar = html.Div(
    [
        html.H2("Patient #", className="display-4"),
        html.Hr(),
        html.P(
            "UPDRS Motor Exam Exercises", className="lead"
        ),
        dbc.Nav(
            [
                dbc.NavLink("Raw Data", href="/", active="exact"),
                dbc.NavLink("Finger Tapping", href="/page-1", active="exact"),
                dbc.NavLink("Close Grip", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)
