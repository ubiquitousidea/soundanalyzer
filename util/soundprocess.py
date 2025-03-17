import os
import re
import json
import numpy as np
import pandas as pd
import plotly.express as px
import librosa
import soundfile
import simpleaudio
import cloudpickle
from time import strftime, gmtime
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA


class SoundProcess(object):
    
    EPOCH = pd.Timestamp('1970-01-01T00:00:00+0000')
    PC_SMOOTH_HALFLIFE = 4
    CLUSTER_SMOOTH_HALFLIFE = 4
    FUCKER = .000001
    
    def __init__(self, sound, fs, path=None):
        """
        sound: vector of sound amplitudes
        fs: int, sample frequency
        """
        self.window_size = None
        self.hop = None
        self.pc1 = None
        self.npc = None
        self.n_cluster = None
        self.setup_complete = False
        self.run_complete = False
        self.fs = fs
        # - data -
        self.path = path
        self.sound = np.array(sound)
        self.duration = len(self.sound) / self.fs
        self.duration_str = strftime("%H:%M:%S", gmtime(self.duration))
        self.target = None  # target variable for supervised learning
        # - models -
        self.stft = None
        self.pc_model = None
        self.cluster_model = None
        # - computed data -
        self.fft = None
        self.window = None
        self.pc_scores = None
        self.cluster_labels = None
        self.colormap = None
        self.class_table = None
    
    def save_binary(self, filename):
        """
        save binary file that can be unpickled later
        """
        data = cloudpickle.dumps(self)
        with open(filename, 'wb') as f:
            f.write(data)

    @classmethod
    def from_wav(cls, path, max_seconds=None, sr=None):
        """
        initialize an object of this class from a wave file
        save the wave file path in the object
        """
        sound, fs = librosa.load(path, sr=sr)
        if max_seconds is not None:
            n = fs * max_seconds
            if n < len(sound):
                sound = sound[-n:]
        obj = cls(sound, fs)
        obj.path = path
        return obj

    @property
    def filename(self):
        """
        the file name part of the path
        """
        return os.path.split(self.path)[-1]

    @property
    def zarray(self):
        """
        log fourier magnitudes for the heatmap and dimension reduction model
        np.abs() converts complex fft outputs to real values representing coefficient magnitude
            phase information from z = a + b i could be recovered by arctan(b / a), enabling
            study of human ear ability to detect phase. 
        np.log() is used to equalize noise variance that occurs over many orders of magnitude
        """
        return np.log(np.abs(self.fft) + self.FUCKER)

    @property
    def sound_times(self):
        """
        time scale of the sound in seconds
        """
        return np.arange(len(self.sound)).astype(np.float64) / self.fs

    @property
    def fft_times(self):
        """
        time scale for the columns of the FFT matrix
        """
        return self.stft.t(len(self.sound))

    @property
    def unique_cluster_labels(self):
        """
        names of all the clusters, in ascending order
        """
        return sorted(list(set(self.cluster_labels)))

    @property
    def pc_smoothed(self):
        """
        return data frame of temporally smoothed pc scores
        pc scores of spectra can oscillate, making classification results unstable
        smoothing the pc scores temporally using exponential weighted moving average
        could allow a model to make smoother predictions
        """
        df = pd.DataFrame(
            self.pc_scores, 
            columns=self.pc_names, 
            index=range(self.pc_scores.shape[0]))
        return df.ewm(halflife=self.PC_SMOOTH_HALFLIFE).mean()

    @property
    def wav_description(self):
        return (
            f"Sound file: {self.path}\n"
            f"Duration: {self.duration_str}\n"
            f"{self.fs} samples per second\n"
            f"{self.sound.__len__()} Total Samples"
        )

    @property
    def fft_result(self):
        if self.fft is None:
            return ''
        else:
            m, n = self.fft.shape
            return f"FFT Matrix: {m} by {n}"

    @property
    def pca_result(self):
        try:
            m, n = self.pc_scores.shape
            return f"PC Shape: {m} x {n}"
        except:
            return ''

    @property
    def clustering_result(self):
        try:
            output = (
                self.class_table
                .label.value_counts()
                .reset_index().to_string()
            )
            return output
        except:
            return ''
    
    def get_pc_scores(self, i, j, k):
        return pd.DataFrame({
            'x': self.pc_scores[:, i],
            'y': self.pc_scores[:, j],
            'z': self.pc_scores[:, k],
            'cluster': self.cluster_labels
        })

    @property
    def pc_names(self):
        return [f'PC{i + 1}' for i in range(self.npc)]

    @property
    def pc_table(self):
        data = self.pc_scores
        df = pd.DataFrame(
            data=data,
            columns=self.pc_names,
            index=range(data.shape[0])
        )
        try:
            return df.assign(cluster=self.cluster_labels)
        except:
            return df

    def setup(self, window=1200, hop=100, nclust=24, pc_min=2, npc=16):
        """
        window: int, window size in samples
        hop: int, hop size in samples
        nclust: number of clusters
        pc_min: lowest pc to include in clustering
        npc: number of principal components extracted
        """
        self.window_size=window
        self.window = gaussian(
            self.window_size, 
            std=.5 * self.window_size)
        self.hop=hop
        self.n_cluster = nclust
        self.npc = npc
        self.pc1 = pc_min
        self.setup_complete = True

    def run(self):
        """
        setup the model parameters, run fft and pca
        window: window size (samples)
        hop: hop size (samples)
        """
        self.compute_fft()
        self.compute_pca()
        self.compute_clusters()

    def set_target(self, event_data):
        """
        encode target variable from event data labels
        populate self.target
        event_data: list of dictionaries with keys {t1, t2, label}
        """
        target = pd.DataFrame(dict(
            ts=self.fft_times
        )).assign(label='')
        for event in event_data:
            if event['filename'] == self.path:
                t1 = float(event['t1'])
                t2 = float(event['t2'])
                ii = target.ts.gt(t1) & target.ts.lt(t2)
                target.loc[ii, 'label'] = event['label']
        self.target = target

    def compute_fft(self, window=None, hop=None, mfft=None, psd=True):
        """
        set up the short time fft and calculate the magnitudes
        """
        if window is not None:
            self.window_size=window
            # self.window = gaussian(window, .5 * window)
            self.window = np.ones(window)
        if hop is not None:
            self.hop = hop
        
        self.stft = ShortTimeFFT(
            win=self.window, 
            hop=self.hop, fs=self.fs, mfft=mfft,
            scale_to='psd' if psd is True else 'magnitude')
        self.fft = self.stft.stft(self.sound)
    
    def compute_pca(self, npc=None, whiten=False):
        """
        fit pca model on fft magnitudes
        """
        if npc is not None:
            self.npc = npc
        self.pc_model = PCA(self.npc, whiten=whiten)
        self.pc_scores = self.pc_model.fit_transform(self.zarray.T)

    def compute_clusters(self, clusters=None, pc_min=None):
        """
        separate the fft spectra into clusters based on pc scores, skipping the first k components
        clusters: number of clusters
        pc_min: minimum pc included in the clustering (zero based index)
        """
        if clusters is not None:
            self.n_cluster = int(clusters)
        if pc_min is not None:
            self.pc1 = int(pc_min)
        self.cluster_model = AgglomerativeClustering(n_clusters=self.n_cluster)
        self.cluster_model.fit(self.pc_scores[:, self.pc1:])
        self.cluster_labels = [str(item) for item in self.cluster_model.labels_]
        self.colormap = {
            cluster: px.colors.qualitative.Light24[i]
            for i, cluster in enumerate(list(set(self.cluster_labels)))
        }
        self.class_table = pd.DataFrame(dict(
            ts=self.fft_times,
            label=self.cluster_labels
        )).assign(label_smoothed=lambda x: self.smoothed_cluster_labels(x))
        
        self.class_table['color'] = self.class_table['label_smoothed'].apply(lambda x: self.colormap[x])
        self.run_complete = True

    def smoothed_cluster_labels(self, df):
        """
        resolve the oscillation between cluster label with temporal smoothing
        encode one binary variable for each class
        apply smoothing
        calculate most likely class label across smoothed binary indicators
        """
        for label in self.unique_cluster_labels:
            df = df.assign(**{f"binary_{label}": lambda x: x.label.eq(label)})
        for label in self.unique_cluster_labels:
            df = df.assign(**{
                f"binary_{label}_smoothed": 
                lambda x: x[f"binary_{label}"].ewm(halflife=self.CLUSTER_SMOOTH_HALFLIFE).mean()})
        df2 = df[[name for name in df.columns if re.match(r'.*_smoothed', name)]]
        most_likely_index = df2.apply(lambda row: np.where(row == row.max())[0][0], axis=1)
        return most_likely_index.astype(str)

    @property
    def sound_groups(self):
        """
        return a dictionary of dataframes of sound data in each cluster
        """
        output = {}
        df1 = self.class_table.reset_index().rename(columns={'index': 'oldindex'})
        for label in self.unique_cluster_labels:
            df = df1.loc[df1.label_smoothed.eq(label)]
            df = df.assign(
                newgroup=lambda x: x.oldindex != x.oldindex.shift(1) + 1,
                groupnum=lambda x: x.newgroup.cumsum()
            )
            output[label] = df
        return output
    
    @property
    def cluster_table(self):
        """
        create table data for cluster table (data_table.DataTable) in the navbar
        """
        output_data = []
        for cluster_label, df in self.sound_groups.items():
            df_ = (
                df.groupby('groupnum')
                .agg({'ts': ['min', 'max']})
                .reset_index()
                .assign(
                    dt=lambda x: (x[('ts','max')] - x[('ts','min')]),
                    cluster=cluster_label,
                    cluster_int=int(cluster_label)
                )
            )
            output_data.append(df_[df_.dt >= np.quantile(df_.dt, .8)])
        data = (
            pd.concat(output_data, axis=0)
            .sort_values(['cluster_int', ('ts', 'min')])
            .reset_index(drop=True)
        )
        data.columns = ['_'.join(map(str, col)) for col in data.columns]
        output = (
            data[['cluster_', 'groupnum_', 'ts_min', 'ts_max', 'dt_']]
            .assign(
                ts_min=lambda x: x.ts_min.apply(lambda x: f"{x:1.3f}"),
                ts_max=lambda x: x.ts_max.apply(lambda x: f"{x:1.3f}"),
                dt_=lambda x: x.dt_.apply(lambda x: f"{x:1.3f}")
            )
            .to_dict('records')
        )
        return output

    def get_selection_from_dash_graph_selection(self, selection):
        """
        return the sound data that was selected
        selection: list of timestamps from ['range']['x'] of the dash graph selection
        """
        return self.get_selection_from_interval([
            selection['range']['x'][0], 
            selection['range']['x'][1]
        ])
        
    def get_selection_from_table_row(self, selected_row):
        return self.get_selection_from_interval([
            selected_row['ts_min'], 
            selected_row['ts_max']
        ])

    def get_selection_from_interval(self, interval):
        if interval is None:
            return self.sound
        else:
            ii = np.where((
                self.sound_times <= float(interval[1])
                ) & (
                self.sound_times >= float(interval[0])
            ))
            return self.sound[ii]

    def playsound(self, data):
        """
        play audio data. 
        sample rate is set when this object is initialized
        data: numpy array of sound data
        """
        simpleaudio.play_buffer(
            audio_data=(data * (2 ** 15 - 1)).astype(np.int16), 
            num_channels=1, 
            bytes_per_sample=2, 
            sample_rate=self.fs
        ).wait_done()

    def play_selection(self, dash_selection=None, selected_row=None):
        """
        play a selection using simpleaudio (does not write a wav file)
        """
        if dash_selection is not None:
            data = self.get_selection_from_dash_graph_selection(selection=dash_selection)
        elif selected_row is not None:
            data = self.get_selection_from_table_row(selected_row=selected_row)
        else:
            raise ValueError('provide dash_selection or selected_row')
        self.playsound(data)

    def play_principal_spectrum(self, i):
        b = self.pc_model.inverse_transform(np.eye(self.npc))[i, :]
        c = np.array(np.exp(b) - self.FUCKER, ndmin=2)
        c = np.concat([c] * 100, axis=0)
        data = self.stft.istft(c.T)
        self.playsound(data)

    def split(self, output_dir='split_sounds'):
        """
        split the sound into clusters
        writes out the sounds into individual wav files in separate directories
        directories are numbered clusters
        individual files are instances of contiguous sound in that cluster
        """
        if not self.run_complete:
            print('Need to run clustering first')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        output_path = os.path.join(output_dir, self.filename.lower().replace('.wav', ''))
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        
        output_summary = []
        for label, df in self.sound_groups.items():
            output_label_path = os.path.join(output_path, str(label))
            if not os.path.isdir(output_label_path):
                os.mkdir(output_label_path)
            for grpnum, df_grp in df.groupby('groupnum'):
                t1 = df_grp.ts.min()
                t2 = df_grp.ts.max()
                ii = np.where((self.sound_times <= t2) & (self.sound_times >= t1))
                sound_data = self.sound[ii]
                if len(sound_data) == 0:
                    continue
                filename_ = os.path.join(output_label_path, f"soundbite_cluster_{label}_item_{grpnum}.wav")
                soundfile.write(filename_, sound_data, self.fs, subtype='PCM_24')
                output_summary.append({
                    "path": filename_,
                    "duration": t2 - t1,
                    "samples": int(np.max(ii)) - int(np.min(ii)),
                    "cluster_label": label,
                    "instance": grpnum
                })
        with open(os.path.join(output_path, 'output_summary.json'), 'w') as f:
            json.dump(output_summary, f, indent=4)
        fig = pd.DataFrame(output_summary).duration.hist()
        return output_summary, fig

