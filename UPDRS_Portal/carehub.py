import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
from dbc_layouts import *
from dbc_styles.grid_view_styles import *
from dbc_styles.sidebar_styles import *
from dbc_layouts.sidebar_layout import * 
from dbc_layouts.grid_view_layout import *
from dbc_layouts.raw_data_layout import *
from raw_data_routes import *

# external_stylesheets=[dbc.themes.BOOTSTRAP] needed 
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP]) 

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

# Side bar navigation routing.
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return app_iotex_layout
    elif pathname == "/page-1":
        return html.P(calander_view)
    elif pathname == "/page-2":
        return html.P("Oh cool, this is page 2!")
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )

json_selected_moth = {'selected_month':''}

# Calendar view dropdown.
@app.callback(
    Output('dd-output-container', 'children'),
    Input('demo-dropdown', 'value'))
def update_output(value):
    return f'Month: {value}'
    
if __name__ == "__main__":
    app.run_server(port=8888)