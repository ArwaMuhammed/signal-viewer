import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import base64
import io
import os

from models.earthquake_predictor import predict_damage
from models.audio_classifier import classify_audio

# Import the SarPy-based analyzer
from models.sar_analyzer import analyze_sar_file   # <-- Save your code into sarpy_analyzer.py


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([

    # Title
    dbc.Row([
        dbc.Col([
            html.H1("🌍 Disaster Detection System", className="text-center my-4"),
            html.Hr()
        ])
    ]),

    # Three Frames: SAR Earthquake, Audio, and SAR TIFF Analysis
    dbc.Row([

        # SAR Earthquake Detection frame
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H3("🛰️ SAR Earthquake Damage")),
                dbc.CardBody([
                    html.P("Upload Sentinel-1 SAR image (.npy format)"),
                    html.P("4 channels: [pre_VV, pre_VH, post_VV, post_VH]",
                           className="text-muted small"),

                    # SAR Upload
                    dcc.Upload(
                        id='upload-sar',
                        children=html.Div([
                            '🖼️ Drag & Drop or ',
                            html.A('Select SAR Image (.npy)')
                        ]),
                        style={
                            'width': '100%',
                            'height': '80px',
                            'lineHeight': '80px',
                            'borderWidth': '2px',
                            'borderStyle': 'dashed',
                            'borderRadius': '10px',
                            'textAlign': 'center',
                            'margin': '10px 0',
                            'backgroundColor': '#f8f9fa'
                        },
                        accept='.npy'
                    ),

                    # SAR Result
                    html.Div(id='sar-result', className="mt-3")
                ])
            ], className="shadow-sm")
        ], md=4),

        # Audio frame
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H3("🎵 Audio Classification")),
                dbc.CardBody([
                    html.P("Upload audio file for drone/bird detection"),
                    html.P("Supported formats: .wav, .mp3",
                           className="text-muted small"),

                    dcc.Upload(
                        id='upload-audio',
                        children=html.Div([
                            '🔊 Drag & Drop or ',
                            html.A('Select Audio File')
                        ]),
                        style={
                            'width': '100%',
                            'height': '80px',
                            'lineHeight': '80px',
                            'borderWidth': '2px',
                            'borderStyle': 'dashed',
                            'borderRadius': '10px',
                            'textAlign': 'center',
                            'margin': '10px 0',
                            'backgroundColor': '#f8f9fa'
                        },
                        accept='.wav,.mp3'
                    ),

                    html.Div(id='audio-result', className="mt-3")
                ])
            ], className="shadow-sm")
        ], md=4),

        # SAR TIFF Analysis frame
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H3("📡 SAR Signal Analysis")),
                dbc.CardBody([

                    dcc.Upload(
                        id='upload-vv',
                        children=html.Div(['📄 Select VV TIFF']),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '5px 0',
                            'backgroundColor': '#f8f9fa',
                            'fontSize': '14px'
                        },
                        accept='.tif,.tiff'
                    ),
                    html.Div(id='vv-status', className="small text-muted"),

                    dcc.Upload(
                        id='upload-vh',
                        children=html.Div(['📄 Select VH TIFF (Optional)']),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '5px 0',
                            'backgroundColor': '#f8f9fa',
                            'fontSize': '14px'
                        },
                        accept='.tif,.tiff'
                    ),
                    html.Div(id='vh-status', className="small text-muted"),

                    dbc.Button("Analyze", id="analyze-tiff-btn", color="primary",
                               className="mt-2 w-100", size="sm", disabled=True),
                ])
            ], className="shadow-sm")
        ], md=4)

    ], className="mb-4"),

    # Results
    dbc.Row([
        dbc.Col([html.Div(id='tiff-result')])
    ]),

    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("Powered by PyTorch + Dash + SarPy",
                   className="text-center text-muted")
        ])
    ]),

    # Hidden storage
    dcc.Store(id='vv-store'),
    dcc.Store(id='vh-store'),

], fluid=True)



@app.callback(
    Output('sar-result', 'children'),
    Input('upload-sar', 'contents'),
    State('upload-sar', 'filename')
)
def classify_sar(contents, filename):
    if contents is None:
        return html.Div("No file uploaded", className="text-muted")
    try:
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        sar_data = np.load(io.BytesIO(decoded))
        result = predict_damage(sar_data)
        return dbc.Alert(f"Result: {result}", color="info")
    except Exception as e:
        return dbc.Alert(f"❌ Error: {e}", color="danger")


@app.callback(
    Output('audio-result', 'children'),
    Input('upload-audio', 'contents'),
    State('upload-audio', 'filename')
)
def classify_audio_file(contents, filename):
    if contents is None:
        return html.Div("No file uploaded", className="text-muted")
    try:
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        temp_path = f"temp_{filename}"
        with open(temp_path, 'wb') as f:
            f.write(decoded)
        result = classify_audio(temp_path)
        os.remove(temp_path)
        return dbc.Alert(f"Result: {result}", color="success")
    except Exception as e:
        return dbc.Alert(f"❌ Error: {e}", color="danger")


@app.callback(
    [Output('vv-status', 'children'),
     Output('vv-store', 'data'),
     Output('analyze-tiff-btn', 'disabled')],
    Input('upload-vv', 'contents'),
    State('upload-vv', 'filename')
)
def save_vv(contents, filename):
    if contents is None:
        return "", None, True
    if ',' in contents:
        _, content_string = contents.split(',')
    else:
        content_string = contents
    decoded = base64.b64decode(content_string)
    os.makedirs("temp_tiff", exist_ok=True)
    path = os.path.join("temp_tiff", filename)
    with open(path, 'wb') as f:
        f.write(decoded)
    return f"✓ {filename}", path, False


@app.callback(
    [Output('vh-status', 'children'),
     Output('vh-store', 'data')],
    Input('upload-vh', 'contents'),
    State('upload-vh', 'filename')
)
def save_vh(contents, filename):
    if contents is None:
        return "", None
    if ',' in contents:
        _, content_string = contents.split(',')
    else:
        content_string = contents
    decoded = base64.b64decode(content_string)
    os.makedirs("temp_tiff", exist_ok=True)
    path = os.path.join("temp_tiff", filename)
    with open(path, 'wb') as f:
        f.write(decoded)
    return f"✓ {filename}", path


@app.callback(
    Output('tiff-result', 'children'),
    Input('analyze-tiff-btn', 'n_clicks'),
    [State('vv-store', 'data'),
     State('vh-store', 'data')],
    prevent_initial_call=True
)
def analyze_tiff(n_clicks, vv_path, vh_path):
    try:
        results = analyze_sar_file(vv_path, vh_path)
        return html.Div([
            dbc.Card([
                dbc.CardHeader("📊 SAR Analysis Results"),
                dbc.CardBody([
                    html.H6("Statistics:"),
                    html.Ul([html.Li(f"{k}: {v}") for k, v in results['statistics'].items()]),
                    html.Hr(),
                    html.Img(src=f"data:image/png;base64,{results['image1']}", style={'width': '100%'}),
                    html.Br(),
                    html.Img(src=f"data:image/png;base64,{results['image2']}", style={'width': '100%'})
                ])
            ])
        ])
    except Exception as e:
        return dbc.Alert(f"❌ Error analyzing SAR: {e}", color="danger")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
