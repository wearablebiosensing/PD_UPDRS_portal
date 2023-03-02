import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
from raw_data_layout import app_iotex_layout
from dbc_styles.sidebar import *
from dbc_styles.grid_view import *

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return app_iotex_layout
    elif pathname == "/page-1":
        return html.P(row)
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


if __name__ == "__main__":
    app.run_server(port=8888)