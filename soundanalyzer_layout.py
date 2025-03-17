import os
import plotly.graph_objs as go
import plotly.express as px
import dash_bootstrap_components as dbc
from dash_extensions.enrich import dcc, html
from dash.dash_table import DataTable

# --- Sound directory and wav files ---

SOUNDDIR = 'soudns'
DDOPTIONS = [
    {
        'label': item.replace("_", " ").title(), 
        'value': os.path.join(SOUNDDIR, item)
    } 
    for item in os.listdir(SOUNDDIR)
]

# TODO: resolve/parameterize this hard coded PC count of 15

PC_OPTIONS = [{'label': f'PC{i + 1}','value': i} for i in range(15)]
PLOT_OPTIONS = [
    {'label': 'Spectrogram', 'value': 'fft'},
    {'label': '3D Points', 'value': '3d'},
    {'label': 'Scatter Matrix', 'value': 'matrix'},
    {'label': 'PC Time Series', 'value': 'pc_ts'}
]
DEFAULT_PLOT = 'fft'  # default plot type

# to make plot background transparent

FIGURE_LAYOUT_SETTINGS = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    dragmode='select'
)

# these settings are used with update_scenes on a 3d scatter plot
FIGURE_3D_AXIS_INVISIBLE = dict(
    xaxis_visible=False, 
    yaxis_visible=False,
    zaxis_visible=False    
)

# - LAYOUT - 

PC_INPUTS = [
    dbc.ListGroupItem([
        dbc.Input(type="number", value=2 + i, min=1, max=16, step=1, id=f'{x}coord')
    ]) for i, x in enumerate(('x', 'y', 'z'))
]

# - nav items -

LOAD_SOUND = dbc.AccordionItem([
    dbc.Row([
        dcc.Dropdown(options=DDOPTIONS, id='filepicker')
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Input(type='number', placeholder='Max Seconds...', min=10, max=60, step=5, value=None, id='maxseconds')
        ]),
        dbc.Col([
            dbc.Input(type='number', placeholder='Sample Rate...', min=22050, max=44100, step=22050, value=22050, id='samplerate')
        ])
    ]),
    dbc.Row([
        dbc.Button('Load...', id='load_sound', style={'width':'100%'})
    ]),
    html.Div(id='datasummary')
], title='Load Sound...')
    

SPECTRAL_ANALYSIS = dbc.AccordionItem([
    dbc.Row([
        dbc.Col([
            html.H4('FFT Window (samples)'),
            dbc.Input(type="number", value=1000, min=200, max=2000, step=100, id='window')
        ]),
        dbc.Col([
            html.H4('Step Size (samples)'),
            dbc.Input(type="number", value=100, min=20, max=200, step=10, id='hop')
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Button('Run FFT...', id='runfft')
        ]),
        dbc.Col([
            dbc.RadioItems(
                options=[{'label': 'PSD', 'value': 'psd'}, {'label': 'Magnitude', 'value': 'mag'}],
                value='mag',
                id='psd_or_magnitude'
            )
        ])
    ]),
    dbc.Row([
        dbc.Col([dbc.Button('Plot Spectrogram...', id='plotspectrogram')])
    ]),
    dbc.Row([
        html.Div(id='fft_result_info')
    ])
], title='Spectral Analysis...')


DATA_LABELING = dbc.AccordionItem([
    dbc.Row([
        dbc.Input(placeholder='Data Label', id='label')
    ]),
    dbc.Row([
        dbc.Button('Store Event',id='store_event'),
        dbc.Button('Show Events', id='showevents')
    ]),
    dbc.Row([DataTable(id='event_table', columns=[
        {'name': 'Label', 'id': 'label'},
        {'name': 'Start', 'id': 't1'},
        {'name': 'Duration (s)', 'id': 'duration'}
    ])])
], title='Data Labeling...')


DIMENSION_REDUCTION = dbc.AccordionItem([
    dbc.Row([
        html.H4('Number of Dims'),
        dbc.Input(type="number", value=16, min=1, max=30, step=1, id='npc')
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Button('Run PCA...', id='runpca')
        ]),
        dbc.Col([
            dbc.Checklist(
                options=[{'label': 'Whiten', 'value': 'True'}],
                id='whiten_pca', value=['True']
            )
        ]),
    ]),
    dbc.Row([
        dbc.Col([dbc.Button('Show Components...', id='showcomponents')]),
        dbc.Col([dbc.Button('Plot 3D...', id='plot3d')]),
        dbc.Col([dbc.Button('Matrix Plot...', id='matrix')]),
        dbc.Col([dbc.Button('Show PC Timeseries...',id='pc_ts')]),
    ]),
    dbc.Row([html.Div(id='pca_result_info')])
], title='Dimension Reduction...')


CLUSTERING = dbc.AccordionItem([
    dbc.Row([
        dbc.Col([
            html.H4('Number of Clusters'),
            dbc.Input(type="number", value=12, min=1, max=24, step=1, id='nclust')]),
        dbc.Col([
            html.H4('Exclude First Dims'),
            dbc.Input(type="number", value=1, min=0, max=7, step=1, id='pc1')
        ]),
    ]),
    dbc.Row([dbc.Button('Cluster Sounds', id='runclustering')]),
    dbc.Row([html.Div(id='clustering_result_info')]),
    dbc.Row([DataTable(
        id='clustertable', row_selectable='single', 
        columns=[
            {'name': 'C', 'id': 'cluster_'},
            {'name': 'Start (s)', 'id': 'ts_min'},
            {'name': 'Duration (s)', 'id': 'dt_'}
        ]
    )])
], title='Clustering...')


FIT_CLASSIFIER = dbc.AccordionItem([
    dbc.Button('Fit Classifier', id='fitclassifier'),
    dbc.Button('View Feature Importance', id='view_feature_importance', disabled=True),
    dbc.Button('Save Classifier', id='savemodel', disabled=True),
    dbc.Button('Show Predictions', id='showprediction', disabled=True),
    DataTable(id='modeltable')
], title='Classification Modeling')


# - main layout -

ANALYZER_LAYOUT = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1('Sound Analyzer')
        ]),
        dbc.Col([
            dbc.Button('Play Selection...', id='playsound'),
            dcc.Store(id='abc123')
        ])
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Accordion([
                LOAD_SOUND,
                SPECTRAL_ANALYSIS,
                DATA_LABELING,
                DIMENSION_REDUCTION,
                CLUSTERING,
                FIT_CLASSIFIER
            ], id='navbar')
        ], width=3),
        dbc.Col([
            dcc.Graph(
                id='graph1', 
                figure=go.Figure(layout=go.Layout(**FIGURE_LAYOUT_SETTINGS))
            )
        ], width=9)
    ]),
    dbc.Toast(
        'console displayed here',
        header="Console", dismissable=True,
        id='console', is_open=False,
        style={"position": "fixed", "bottom": 20, "right": 20, "width": 400, 'max-height': 600, 'overflow-y': 'auto'}
    ),
    dcc.Store(id='soundprocess'),
    dcc.Store(id='path'),
    dcc.Store(id='events', data=[]),
    dcc.Store(id='classifiermodel'),
    dcc.Store(id='graphtype')
], fluid=True, id='main-layout')
