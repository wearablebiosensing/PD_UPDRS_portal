import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, callback,State
from dash_bootstrap_components._components.Container import Container
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
import glob
from sklearn import preprocessing

search_bar = dbc.Row(
    [
        dbc.Col(dbc.Input(type="search", placeholder="Search")),
        dbc.Col(
            dbc.Button(
                "Search", color="primary", className="ms-2", n_clicks=0
            ),
            width="auto",
        ),
    ],
    className="g-0 ms-auto flex-nowrap mt-3 mt-md-0",
    align="center",
)
# NavBar Component.
navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(dbc.NavbarBrand("CareHub", className="ms-2")),
                    ],
                    align="center",
                    className="g-0",
                ),
                href="https://plotly.com",
                style={"textDecoration": "none"},
            ),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            dbc.Collapse(
                search_bar,
                id="navbar-collapse",
                is_open=False,
                navbar=True,
            ),
        ]
    ),
    className = "nav-bar"
    # color="dark",
    # dark=True,
)
# For navigating through different projects.

card_content1 = [
    # dbc.CardHeader("IoTex"),
    dbc.CardBody(
        [
            html.H5("IoTex", className="card-title"),
            html.P(
                "A Smart Glove Digital Health Platform for Monitoring Parkinsons Disease Symptoms",
                className="card-text",
            ),
            dbc.CardLink("Go to iotex dashboard", href="/iotex"),

        ]
    ),
]
card_content2 = [
    # dbc.CardHeader("Carewell"),
    dbc.CardBody(
        [
            html.H5("Carewell", className="card-title"),
            html.P(
                "A Digital Health Platform for Caregivers of Dementia Patients",
                className="card-text",
            ),
        dbc.CardLink("Go to carewell dashboard", href="/carewell"),
        ]
    ),
]

card_content3 = [
    # dbc.CardHeader("CarePortal"),
    dbc.CardBody(
        [
            html.H5("CarePortal", className="card-title"),
            html.P(
                "A Clinician Centered Data Visvulization Platform for Wearable Sensor Data",
                className="card-text",
            ),
        dbc.CardLink("External link", href="https://google.com"),
        ]
    ),
]

row_1 = dbc.Row(
    [
        dbc.Col(dbc.Card(card_content1, color="", outline=True)),
        dbc.Col(dbc.Card(card_content2, color="", outline=True)),
        dbc.Col(dbc.Card(card_content3, color="", outline=True)),
    ],
    className="mb-4",
)

row_2 = dbc.Row(
    [
        dbc.Col(dbc.Card(card_content1, color="", outline=True)),
        dbc.Col(dbc.Card(card_content2, color="", outline=True)),
        dbc.Col(dbc.Card(card_content3, color="", outline=True)),
    ],
    className="mb-4",
)

cards = html.Div([row_1])
