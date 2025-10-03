import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import base64
import io

from models.earthquake_predictor import predict_damage
from models.audio_classifier import classify_audio

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([

    # Title
    dbc.Row([
        dbc.Col([
            html.H1("🌍 Disaster Detection System", className="text-center my-4"),
            html.Hr()
        ])
    ]),

    # Two Frames: SAR and Audio
    dbc.Row([

        # SAR frame
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H3("🛰️ SAR Image Analysis")),
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
        ], md=6),

        # Audio frame
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H3("🎵 Audio Classification")),
                dbc.CardBody([
                    html.P("Upload audio file for drone/bird detection"),
                    html.P("Supported formats: .wav, .mp3",
                           className="text-muted small"),

                    # Audio Upload
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

                    # Audio Result
                    html.Div(id='audio-result', className="mt-3")
                ])
            ], className="shadow-sm")
        ], md=6)

    ], className="mb-4"),

    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("Powered by PyTorch + Dash",
                   className="text-center text-muted")
        ])
    ])

], fluid=True)

# Callback: SAR classification
@app.callback(
    Output('sar-result', 'children'),
    Input('upload-sar', 'contents'),
    State('upload-sar', 'filename')
)
def classify_sar(contents, filename):
    if contents is None:
        return html.Div("No file uploaded", className="text-muted")

    try:
        # Decode uploaded .npy file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        sar_data = np.load(io.BytesIO(decoded))

        # Validate channels (allow any height/width)
        if sar_data.ndim != 3 or sar_data.shape[0] != 4:
            return dbc.Alert(
            f"❌ Invalid shape: {sar_data.shape}. Expected (4, H, W)",
            color="danger"
            )

        # Validate dtype
        if sar_data.dtype != np.float32:
            sar_data = sar_data.astype(np.float32)

        # PREDICT
        result = predict_damage(sar_data)

        # Display result
        if result == "Damage":
            return dbc.Alert([
                html.H4("🚨 DAMAGE DETECTED", className="alert-heading"),
                html.Hr(),
                html.P(f"File: {filename}"),
                html.P("Earthquake damage identified in SAR imagery", className="mb-0")
            ], color="danger")
        else:
            return dbc.Alert([
                html.H4("✅ No Damage", className="alert-heading"),
                html.Hr(),
                html.P(f"File: {filename}"),
                html.P("No significant damage detected", className="mb-0")
            ], color="success")

    except Exception as e:
        return dbc.Alert(
            f"❌ Error processing file: {str(e)}",
            color="danger"
        )


# Callback: audio classification
@app.callback(
    Output('audio-result', 'children'),
    Input('upload-audio', 'contents'),
    State('upload-audio', 'filename')
)
def classify_audio_file(contents, filename):
    if contents is None:
        return html.Div("No file uploaded", className="text-muted")

    try:
        # Decode uploaded audio file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        # Save temporarily (classify_audio needs file path)
        temp_path = f"temp_audio_{filename}"
        with open(temp_path, 'wb') as f:
            f.write(decoded)

        # CLASSIFY using your function
        result = classify_audio(temp_path)  # Returns: "drone", "bird", or "other"

        # Clean up temp file
        import os
        os.remove(temp_path)

        # Display result
        if result.lower() == "drone":
            return dbc.Alert([
                html.H4("🚁 DRONE DETECTED", className="alert-heading"),
                html.Hr(),
                html.P(f"File: {filename}"),
                html.P("Drone sound identified in audio", className="mb-0")
            ], color="warning")

        elif result.lower() == "bird":
            return dbc.Alert([
                html.H4("🐦 BIRD DETECTED", className="alert-heading"),
                html.Hr(),
                html.P(f"File: {filename}"),
                html.P("Bird sound identified in audio", className="mb-0")
            ], color="info")

        else:  # "other"
            return dbc.Alert([
                html.H4("🔊 OTHER SOUND", className="alert-heading"),
                html.Hr(),
                html.P(f"File: {filename}"),
                html.P("Sound classified as 'other'", className="mb-0")
            ], color="secondary")

    except Exception as e:
        return dbc.Alert(
            f"❌ Error processing audio: {str(e)}",
            color="danger"
        )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)