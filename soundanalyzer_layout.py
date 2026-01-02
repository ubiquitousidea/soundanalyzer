import os
import plotly.graph_objs as go
import plotly.express as px
import dash_bootstrap_components as dbc
from dash_extensions.enrich import dcc, html
from dash.dash_table import DataTable
from dash_ag_grid import AgGrid

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

EVENT_COLUMNS = [
    {'name': 'Label', 'id': 'label'},
    {'name': 'Start', 'id': 'ts_min'},
    {'name': 'Duration (s)', 'id': 'duration'}
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
        dbc.Button('Load Events', id='load_events', color='primary'),
        dbc.Button('Store Event', id='store_event')
    ]),
    dbc.Row([DataTable(id='event_table', columns=EVENT_COLUMNS, row_selectable='single')])
], title='Data Labeling...')


DIMENSION_REDUCTION = dbc.AccordionItem([
    dbc.Row([
        html.H4('Number of Dims'),
        dbc.Input(type="number", value=16, min=1, max=30, step=1, id='npc')
    ]),
    dbc.Row([
        html.H4('Component Number'),
        dcc.Dropdown(
            options=[{'label': f'PC {i + 1}', 'value': i} for i in range(16)],
            value=1,
            id='component_number'
        )
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
    dbc.Button('Start Modeling', id='chooseevents'),
    dbc.Button('View Feature Importance', id='view_feature_importance', disabled=True),

    dbc.Button('Show Predictions', id='showprediction', disabled=True),

], title='Classification Modeling')

MODAL_CHOOSE_EVENTS = dbc.Modal([
    dbc.ModalHeader([html.H2('Choose Training Data')]),
    dbc.ModalBody([
        html.P('Choose events for model training data'),
        AgGrid(
            id='event_chooser_table',
            rowData=None,
            columnDefs=[
                {'headerName': 'Label', 'field': 'label', "checkboxSelection": True, "headerCheckboxSelection": True},
                {'headerName': 'Start', 'field': 'ts_min'},
                {'headerName': 'Duration (s)', 'field': 'duration'}
            ],
            style={"height": '30vh', "width": '100%'},
            columnSize="sizeToFit",
            dashGridOptions={"rowSelection": 'multiple'}
        ),
        AgGrid(
            id='model_selector_table',
            rowData=None,
            columnDefs=[
                {'headerName': 'Kind', 'field': 'Kind'},
                {'headerName': 'Features', 'field': 'Num Features'},
                {'headerName': 'Samples', 'field': 'Num Samples'},
                {'headerName': 'Labels', 'field': 'Labels'},
                {'headerName': 'Type', 'field': 'Classifier Type'},
            ],
            style={"height": '30vh', "width": '100%'},
            columnSize="sizeToFit",
            dashGridOptions={"rowSelection": 'multiple'}
        )
    ]),
    dbc.ModalFooter([
        dbc.ButtonGroup([
            dbc.Button('Remove Event', id='remove_event', color='warning'),
            dbc.Button('Fit Classifier', id='fitclassifier', color='success'),
            dbc.Button('Save Classifier', id='savemodel', disabled=True, color='primary')
        ])
    ])
], id='chooseevents_modal', size='xl', is_open=False)

CONSOLE = dbc.Toast(
    html.Pre(id='console_text'),
    header="Console", dismissable=True,
    id='console', is_open=False,
    style={"position": "fixed", "bottom": 20, "right": 20, "width": 400, 'max-height': 600, 'overflow-y': 'auto'}
)

# - main layout -

ANALYZER_LAYOUT = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1('Sound Analyzer')
        ], width=6),
        dbc.Col([
            dbc.Button('Play Selection...', id='playsound')
        ], width=3),
        dbc.Col([
            dbc.Checklist(
                id='showconsole', 
                options=[{'label': 'Show Console', 'value': 'True'}], 
                value=['True']
            )
        ], width=3)
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
    CONSOLE,
    MODAL_CHOOSE_EVENTS,
    dcc.Store(id='soundprocess'),
    dcc.Store(id='path'),
    dcc.Store(id='events', data=[]),
    dcc.Store(id='classifiermodel'),
    dcc.Store(id='graphtype'),
    dcc.Store(id='abc123'),
    dcc.Store(id='model_index')
], fluid=True, id='main-layout')
