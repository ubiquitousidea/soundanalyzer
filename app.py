import os
import sys
import json
import pickle
import getpass
import cloudpickle
import numpy as np
import pandas as pd
from uuid import uuid4
import dash
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import (
    DashProxy, ServersideOutputTransform, 
    Serverside, FileSystemBackend, callback, 
    html, Input, Output, State
)
from util.soundprocess import SoundProcess
from util.classifier import XGClassifier, RFClassifier
from util.plotter import SoundProcessPlotter
from util.util import delete_all_files
from soundanalyzer_layout import (
    ANALYZER_LAYOUT, FIGURE_LAYOUT_SETTINGS,
    FIGURE_3D_AXIS_INVISIBLE
)

user = getpass.getuser()

if sys.platform == 'win32':
    cachedir = f'C:/users/{user}/dashcache'
else:
    cachedir = f'/users/{user}/dashcache'

delete_all_files(cachedir)

backend = FileSystemBackend(cache_dir=cachedir)
app = DashProxy(
    __name__, 
    transforms=[ServersideOutputTransform(backends=[backend])], 
    external_stylesheets=[dbc.themes.LUX])
app.title = 'Sound Analyzer'

# --- LAYOUT ---

app.layout = ANALYZER_LAYOUT

# --- Callbacks ---

@callback(
    Output('soundprocess', 'data'),
    Output('path', 'data'),
    Output('datasummary', 'children'),
    Output('fft_result_info', 'children'),
    Output('pca_result_info', 'children'),
    Output('clustering_result_info', 'children'),
    Input('load_sound', 'n_clicks'),
    Input('runfft', 'n_clicks'),
    Input('runpca', 'n_clicks'),
    Input('runclustering', 'n_clicks'),
    State('filepicker', 'value'),
    State('soundprocess', 'data'),
    State('window', 'value'),
    State('hop', 'value'),
    State('nclust', 'value'),
    State('pc1', 'value'),
    State('npc', 'value'),
    State('maxseconds', 'value'),
    State('whiten_pca', 'value'),
    State('samplerate', 'value'),
    State('psd_or_magnitude', 'value'),
)
def main_action(
    n1, n2, n3, n4, path, sp_data, window, hop, 
    nclust, pc1, npc, maxseconds, whiten, sr, scalingmethod):
    
    if not any([n1, n2, n3, n4]):
        raise PreventUpdate
    _id = dash.callback_context.triggered_id
    if _id == 'load_sound':
        sp = SoundProcess.from_wav(path=path, max_seconds=maxseconds, sr=sr)
    else:
        sp = pickle.loads(sp_data)
        assert isinstance(sp, SoundProcess)
        if _id == 'runfft':
            sp.compute_fft(window, hop, psd=scalingmethod == 'psd')
        elif _id == 'runpca':
            sp.compute_pca(npc, whiten='True' in whiten)
        elif _id == 'runclustering':
            sp.compute_clusters(clusters=nclust, pc_min=pc1 + 1)
        else:
            raise PreventUpdate
    return (
        Serverside(cloudpickle.dumps(sp)), 
        Serverside(path),
        html.Pre(sp.wav_description),
        html.Pre(sp.fft_result),
        html.Pre(sp.pca_result),
        html.Pre(sp.clustering_result)
    )

# - Generate Graph - 

@callback(
    Output('graph1', 'figure'),
    Output('graphtype', 'data'),
    Input('soundprocess', 'data'),
    Input('plotspectrogram', 'n_clicks'),
    Input('plot3d', 'n_clicks'),
    Input('view_feature_importance', 'n_clicks'),
    Input('showprediction', 'n_clicks'),
    Input('showcomponents', 'n_clicks'),
    Input('matrix', 'n_clicks'),
    Input('pc_ts', 'n_clicks'),
    State('classifiermodel', 'data')
)
def make_graph(sp_data, n1, n2, n3, n4, n5, n6, n7, model_data):
    try:
        sp = pickle.loads(sp_data)
        plotter = SoundProcessPlotter(sp)
    except:
        raise PreventUpdate
    try:
        m = pickle.loads(model_data)
    except:
        m = None
    
    caller = dash.callback_context.triggered_id
    
    match caller:
        case 'plot3d':
            fig = plotter.plot_pca_3d(1, 2, 3)
            graphtype = '3d'
            fig.update_scenes(**FIGURE_3D_AXIS_INVISIBLE)
        case 'plotspectrogram':
            fig = plotter.plot_sound_fft()
            graphtype = 'timeseries'
        case 'matrix':
            fig = plotter.plot_pca_matrix()
            graphtype = 'matrix'
        case 'pc_ts':
            fig = plotter.plot_wav_and_pcs()
            graphtype = 'timeseries'
        case 'showprediction':
            fig = m.plot_predictions(sp)
            graphtype = 'timeseries'
        case 'showcomponents':
            fig = plotter.plot_principal_spectra(log=True)
            graphtype = 'spectra'
        case 'view_feature_importance':
            fig = m.plot_variable_importance()
            graphtype = 'bar'
        case _:
            raise PreventUpdate
    
    fig.update_layout(**FIGURE_LAYOUT_SETTINGS)
    return fig, graphtype

# - Play Sound - 

@callback(
    Output('abc123', 'data'),
    Input('playsound', 'n_clicks'),
    Input('graph1', 'clickData'),
    Input('clustertable', 'selected_rows'),
    Input('event_table', 'selected_rows'),
    State('soundprocess', 'data'),
    State('graph1', 'selectedData'),
    State('graphtype', 'data'),
    State('clustertable', 'data'),
    State('event_table','data')
)
def play_sound(n, cd, sel_rows, sel_rows2, sp_data, graph_selection, gt, clustertable, event_table):
    try:
        sp=pickle.loads(sp_data)
        assert isinstance(sp, SoundProcess)
    except:
        raise PreventUpdate
    caller = dash.callback_context.triggered_id
    if caller == 'clustertable':
        sp.play_selection(selected_row=clustertable[sel_rows[0]])
    elif  caller == 'event_table':
        sp.play_selection(selected_row=event_table[sel_rows2[0]])
    elif gt == 'timeseries':
        sp.play_selection(dash_selection=graph_selection)
    elif gt == 'spectra':
        sp.play_principal_spectrum(cd['points'][0]['curveNumber'])
    return None

# - fit classification model

@callback(
    Output('classifiermodel', 'data'),
    Input('fitclassifier', 'n_clicks'),
    State('events', 'data'),
    State('soundprocess', 'data')
)
def build_the_classifier_model(n, events, sp_data):
    """
    get events, encode labels onto spectral data, fit classifier
    """
    if not n:
        raise PreventUpdate
    sp = pickle.loads(sp_data)
    sp.set_target(events)
    m = RFClassifier.from_fftpc(sp)
    m.fit()
    return Serverside(cloudpickle.dumps(m))

# - Save classifier model to file 

@callback(
    Output('modeltable', 'data'),
    Output('modeltable', 'columns'),
    Input('savemodel', 'n_clicks'),
    State('path', 'data'),
    State('classifiermodel', 'data')
)
def save_model_and_fill_model_table(n, path, classifier):
    try:
        m = pickle.loads(classifier)
    except:
        raise PreventUpdate
    name = os.path.split(path)[-1].replace('.wav', '')
    m.save_binary(f'models/classifier/{name}.bin')
    return [{'name': name} for name in os.listdir('models/classifier')], [{'name': 'Filename', 'id': 'name'}]
    
# - store event data from selection - 

@callback(
    Output('events', 'data'),
    Input('store_event', 'n_clicks'),
    Input('showevents', 'n_clicks'),
    State('path', 'data'),
    State('graph1', 'selectedData'),
    State('events', 'data'),
    State('label', 'value')
)
def store_event(n, n2, path, selection, data, label):
    if not n:
        # on start up, attempt to load stored events
        try:
            with open('../events_temp.json', 'r') as f:
                data = json.load(f)
            return data
        except:
            raise PreventUpdate
    try:
        # then try to add event from range selection
        s = selection['range']['x']
        new_event = dict(
            filename=path,
            t1=s[0],
            t2=s[1],
            label=label,
            uuid=str(uuid4())
        )
        data.append(new_event)
    except:
        pass
    # data is returned, possibly modified
    return data

# - show events in the event table - 

@callback(
    Output('event_table', 'data'),
    Input('events', 'data'),
    State('path', 'data')
)
def show_events(events, path):
    table_data = [
        {
            'label': item['label'],
            'ts_min': f"{float(item['t1']):1.3f}", 
            'ts_max': f"{float(item['t2']):1.3f}",
            'duration': f"{float(item['t2']) - float(item['t1']):1.3f}"
        } 
        for item in events 
        if item['filename'] == path
    ]
    with open('../events_temp.json', 'w') as f:
        json.dump(events, f, indent=4)
    return table_data


@callback(
    Output('view_feature_importance','disabled'),
    Output('savemodel', 'disabled'),
    Output('showprediction', 'disabled'),
    Input('classifiermodel', 'data'),
)
def enable_view_importance_button(cm_data):
    if cm_data:
        return False, False, False
    else:
        return True, True, True


@callback(
    Output('clustertable','data'),
    Input('soundprocess', 'data')
)
def fill_cluster_table(sp_data):
    try:
        sp = pickle.loads(sp_data)
        assert isinstance(sp, SoundProcess)
        assert sp.run_complete
    except:
        raise PreventUpdate
    return sp.cluster_table


@callback(
    Output('console', 'children'),
    Output('console', 'is_open'),
    Input('graph1', 'clickData'),
    Input('graph1', 'selectedData'),
    Input('clustertable', 'selected_rows'),
    State('graphtype', 'data')
)
def show_console_data(cd, sd, rows, gt):
    propids = dash.callback_context.triggered_prop_ids
    try:
        caller = list(propids.keys())[0]
    except:
        raise PreventUpdate
    match caller:
        case 'graph1.clickData':
            data = json.dumps(cd, indent=4)
        case 'graph1.selectedData':
            data = json.dumps(sd, indent=4)
        case 'clustertable.selected_rows':
            data = json.dumps(rows, indent=4)
        case _:
            raise PreventUpdate
    if data == 'null':
        raise PreventUpdate
    output = html.Pre(data)
    return output, True


if __name__ == '__main__':
    app.run_server(debug=True)
