import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from .soundprocess import SoundProcess
from time import strftime, gmtime



class SoundProcessPlotter(object):
    """
    class for making plots based on a SoundProcess object
    """
    def __init__(self, soundprocess:SoundProcess):
        self.sp = soundprocess
        self.interval = None

    def set_interval(self, interval):
        """
        set the interval of the plot
        interval: dict with 'ts_min' and 'ts_max' keys for min and max timestamp
        """
        assert 'ts_min' in interval
        assert 'ts_max' in interval
        self.interval = [interval['ts_min'], interval['ts_max']]

    @property
    def t1(self):
        """
        minimum timestamp of the plot (time in seconds)
        """
        return self.interval[0]
    
    @property
    def t2(self):
        """
        maximum timestamp of the plot (time in seconds)
        """
        return self.interval[1]
    
    @property
    def data(self):
        return self.sp.get_selection_from_interval(self.interval)

    @property
    def duration(self):
        return len(self.data) / self.sp.fs

    @property
    def duration_str(self):
        return strftime("%H:%M:%S", gmtime(self.duration))

    @property
    def description(self):
        return (
            f"Sound file: {self.sp.path}\n"
            f"Duration: {self.duration_str}\n"
            f"{self.sp.fs} samples per second\n"
            f"{self.data.__len__()} Total Samples"
        )

    def plot_wave(self):
        """
        plot the wave form as a line graph
        returns a plotly Scatter object
        """
        df = pd.DataFrame(dict(
            ts=self.sp.sound_times,
            y=self.sp.sound
        ))
        return go.Scatter(
            x=df.ts, y=df.y, mode='lines', 
            line=dict(color='white', width=1))

    def plot_heatmap(self):
        return go.Heatmap(
            z=self.sp.zarray, x=self.sp.fft_times, y=self.sp.stft.f,
            colorscale='viridis')

    def plot_pc_timeseries(self, *pcs, smoothed=True):
        """
        return a list of line plots for each PC in pcs
        *pcs: indices of pcs to plot
        """
        colors = px.colors.qualitative.Light24
        if not pcs:
            pcs = list(range(7))
        scores = self.sp.pc_smoothed.to_numpy() if smoothed else self.sp.pc_scores
        return [
            go.Scatter(
                x=self.sp.fft_times, 
                y=scores[:, i], 
                mode='lines', 
                line=dict(color=colors[i])
            )
            for i in pcs
        ]

    def plot_sound_fft(self):
        """
        plot the frequency spectra and sound wave
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(self.plot_heatmap(), secondary_y=False)
        fig.add_trace(self.plot_wave(), secondary_y=True)
        
        fig.add_trace(go.Scatter(
            x=self.sp.fft_times, 
            y=[0] * len(self.sp.fft_times), mode='markers', 
            marker=dict(
                size=5, opacity=.5, 
                color=self.sp.class_table.color if self.sp.run_complete else '#fff'
            )
        ), secondary_y=True)
        return fig

    def plot_pca_3d(self, i, j, k):
        return px.scatter_3d(
            data_frame=self.sp.get_pc_scores(i, j, k),
            x='x', y='y', z='z', 
            color='cluster', opacity=.5, 
            category_orders={'cluster': sorted(self.sp.cluster_labels, key=int)} if self.sp.run_complete else None,
            color_discrete_map=self.sp.colormap
        )

    def plot_pca_matrix(self):
        df = self.sp.pc_table
        fig = px.scatter_matrix(
            df,
            dimensions=[col for col in df.columns if col.startswith('PC')],
            color='cluster'
        )
        return fig

    def plot_wav_and_pcs(self):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(self.plot_heatmap(), secondary_y=False)
        fig.add_traces(self.plot_pc_timeseries(0, 1, 2, 3, 4, 5, 6), secondary_ys=[True] * 7)
        return fig
    
    def plot_principal_spectra(self, log=False):
        """
        plot the principal components (spectra) of the sound
        log: bool, plot the log of the fft coefficient?
        """
        b = self.sp.pc_model.inverse_transform(np.eye(self.sp.npc))
        
        if log is True:
            x = np.log(self.sp.stft.f)
            spectra = b
            SF = 1
            laballer = lambda x: f"{np.exp(x):1.2f}"
        else:
            x = self.sp.stft.f
            spectra = np.exp(b) - self.sp.FUCKER
            SF = 2
            laballer = lambda x: f"{x:1.3f}"
        
        fig = go.Figure()
        
        for i in range(self.sp.npc):
            fig.add_trace(go.Scatter(
                x=x,
                y=spectra[i],
                mode='lines',
                name=f"PC{i + 1}"
            ))
        
        fig.update_layout(
            xaxis = dict(
                tickmode = 'array',
                tickvals = np.linspace(
                    (10 ** SF) * int(np.min(x[~np.isinf(x)]) / (10 ** SF)), 
                    (10 ** SF) * int(np.max(x[~np.isinf(x)]) / (10 ** SF)),
                    num=11
                ),
                ticktext = [laballer(_x) for _x in x]
            )
        )
        return fig
        
