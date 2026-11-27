import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc


@njit
def map_loop(
    x0: float,
    base_loop: np.ndarray[tuple[int], np.dtype[np.float64]],
    loop_iters: int = 1,
) -> float:
    x = x0
    N = 0
    logs = np.empty(base_loop.shape[0] * loop_iters, np.float64)
    for ITER in range(loop_iters):
        for base in base_loop:
            x *= base * (1.0 - x)
            if not np.isfinite(x):
                break

            # if ITER > 2:
            log = np.log(np.abs(base * (1.0 - 2 * x)))
            if np.isfinite(log):
                logs[N] = log
                N += 1

    if N > base_loop.shape[0] * 2:
        lyap_exp = np.nanmean(logs[:N])
    else:
        lyap_exp = np.nan
        print(base_loop)

    return lyap_exp


@njit(parallel=True)
def map_grid(
    x0: float,
    loop_iters: int,
    bool_loop: np.ndarray[tuple[int], np.dtype[np.bool]],
    Ngrid: int,
    arange: tuple[float, float] = (0.0, 4.0),
    brange: tuple[float, float] = (0.0, 4.0),
):
    out_lyap = np.empty((Ngrid, Ngrid), np.float64)

    amin, amax = arange
    bmin, bmax = brange
    assert amax > amin
    assert bmax > bmin
    assert Ngrid > 2

    da = (amax - amin) / (Ngrid + 1)
    db = (bmax - bmin) / (Ngrid + 1)

    A_inds = np.where(~bool_loop)[0]
    B_inds = np.where(bool_loop)[0]

    for i in prange(Ngrid):
        float_loop = np.empty((bool_loop.shape[0],), np.float64)
        a = amin + i * da
        for ind in A_inds:
            float_loop[ind] = a
        for j in range(Ngrid):
            b = bmin + j * db
            for ind in B_inds:
                float_loop[ind] = b
            out_lyap[j, i] = map_loop(x0, float_loop, loop_iters)

    return out_lyap


def gui(debug: bool = False, port: int = 8000, jupyter_mode: str = "inline"):
    DEFAULTCODE = "AB"
    DEFAULTCODE_BOOL = [c == "B" for c in DEFAULTCODE.upper()]
    NGRID0 = 250
    NLOOP0 = 50

    data = np.zeros((10, 10))

    img = go.Heatmap(
        z=data,
        x=[0, 4],
        y=[0, 4],
        colorscale="Viridis",
        showscale=False,
        zsmooth="best",
    )  # , hoverinfo="skip"

    fig = go.Figure(data=[img])

    # set layout
    fig.update_layout(
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=0, r=0, b=0, t=0),
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
    )

    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, range=(0, 4), autorange=False),
        yaxis=dict(showgrid=False, zeroline=False, range=(0, 4), autorange=False),
    )

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.themes.CYBORG])

    #fmt: off
    app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            html.Div([dbc.FormText("AB Code"), dbc.Input(value=DEFAULTCODE, type='text',id="code-input", debounce=True)]),
            width=3
        ),
        dbc.Col(
            html.Div([
                dbc.FormText("Colormap"),
                dbc.Select(px.colors.named_colorscales(), "viridis", id="cmap-dropdown")
            ]),
            width=3
        ),
        dbc.Col(
            html.Div([dbc.FormText("Colormap Scale"), dcc.Slider(0, 3, step=0.01, marks=None, value=0., tooltip=False, allow_direct_input=False,id='color-scale')]),
            width=6
        ),
    ]),
    
    dbc.Row([
        dbc.Col(html.Div([dcc.Graph(figure=fig, id="display", style={"height": "80vh"})]), width=12)
    ]),
    
    
    dbc.Row([
        dbc.Col(
            html.Div([dbc.FormText("Grid Density"), dbc.Input(value=NGRID0, type='number',step=1, min=100,id="NGRID", debounce=True)]),
            width=5
        ),
        dbc.Col(
            html.Div([dbc.FormText("Iteration Count"), dbc.Input(value=NLOOP0, type='number',step=1, min=25,id="NLOOP", debounce=True)]),
            width=5
        ),
        dbc.Col(
            html.Div([dbc.FormText("Export"), dbc.Button("Save", id="export-btn")]),
            width=1
        ),
    ]),
    

    html.Div([
        dcc.Store(id="data-store", storage_type="memory"),
        dcc.Store(id="code-store", storage_type="memory", data=DEFAULTCODE_BOOL),
    ])
    ], fluid=False)
    # fmt: on

    @callback(
        Output("code-store", "data"),
        Output("display", "figure"),
        Input("code-input", "value"),
        State("display", "figure"),
    )
    def set_code(code: str, fig: dict):

        fig["layout"]["xaxis"]["range"] = [0, 4]
        fig["layout"]["yaxis"]["range"] = [0, 4]

        code_bool = [c == "B" for c in code.upper()]


        return code_bool, fig

    @callback(
        Output("data-store", "data"),
        Input("display", "figure"),
        Input("NGRID", "value"),
        Input("NLOOP", "value"),
        State("code-store", "data"),
    )
    def recompute(fig: dict, NGRID: int, NLOOP: int, code_bool: list[bool]):
        A0, A1 = fig["layout"]["xaxis"]["range"]
        B0, B1 = fig["layout"]["yaxis"]["range"]

        A0 = max(0.0, A0)
        B0 = max(0.0, B0)
        A1 = min(A1, 4.0)
        B1 = min(B1, 4.0)

        data = map_grid(
            0.5, int(NLOOP), np.array(code_bool), int(NGRID), (B0, B1), (A0, A1)
        )

        return data

    @callback(
        Output("display", "figure", allow_duplicate=True),
        Input("data-store", "data"),
        Input("cmap-dropdown", "value"),
        Input("color-scale", "value"),
        State("display", "figure"),
        State("NGRID", "value"),
        prevent_initial_call=True,
    )
    def replot(data: list, cmap: str, colorscale: float, fig: dict, NGRID: int):
        A0, A1 = fig["layout"]["xaxis"]["range"]
        B0, B1 = fig["layout"]["yaxis"]["range"]

        data_plot = np.array(data, dtype=np.float64)
        data_plot[np.isnan(data_plot)] = 0.0

        A0 = max(0.0, A0)
        B0 = max(0.0, B0)
        A1 = min(A1, 4.0)
        B1 = min(B1, 4.0)

        # fig["layout"]["xaxis"]["range"] = (A0, A1)
        # fig["layout"]["yaxis"]["range"] = (B0, B1)

        # data_plot[data_plot > 0] = 0.0
        data_plot = np.asinh(data_plot / 10 ** (-colorscale))
        fig["data"][0]["z"] = data_plot.T
        fig["data"][0]["x"] = np.linspace(A0, A1, int(NGRID))
        fig["data"][0]["y"] = np.linspace(B0, B1, int(NGRID))

        fig["data"][0]["colorscale"] = px.colors.get_colorscale(cmap)

        return fig

    @callback(
        Input("export-btn", "n_clicks"),
        State("code-store", "data"),
        State("cmap-dropdown", "value"),
        State("color-scale", "value"),
        State("display", "figure"),
        State("NGRID", "value"),
        State("NLOOP", "value"),
        prevent_initial_call=True,
    )
    def save_image(
        _,
        code: list[bool],
        cmap_name: str,
        colorscale: float,
        fig: dict,
        NGRID: int,
        NLOOP: int,
    ):
        cmap = plt.get_cmap(cmap_name)

        A = fig["layout"]["xaxis"]["range"]
        B = fig["layout"]["yaxis"]["range"]

        data = map_grid(0.5, int(NLOOP * 3), np.array(code), int(NGRID * 5), B, A)
        data_plot = np.array(data, dtype=np.float64)
        data_plot[np.isnan(data_plot)] = 0.0

        data_plot[data_plot > 0.0] = 0.0
        data_plot = np.asinh(data_plot / 10 ** (-colorscale))
        data_plot -= data_plot.min()
        data_plot /= data_plot.max()

        rgba_image = cmap(data_plot.T)

        code_str = "".join([chr(ord("A") + v) for v in code])
        plt.imsave(
            f"image_{code_str}_{A[0]:.4f}-{A[1]:.4f}_{B[0]:.4f}-{B[1]:.4f}.png",
            rgba_image,
            origin="lower",
            dpi=300,
        )
        print("SAVED!")

    _ = set_code("AB", fig.to_dict())
    app.run(debug=debug, use_reloader=False, jupyter_mode=jupyter_mode, port=port)


out = gui(jupyter_mode="external")
