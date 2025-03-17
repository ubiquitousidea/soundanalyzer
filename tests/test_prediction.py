from ..util.soundprocess import SoundProcess
from ..util.classifier import XGClassifier, RFClassifier, ClassifierModel
from ..util.util import load_events, load_object
import plotly.express as px
import json


sp = SoundProcess.from_wav("soudns/gmajor-scale-2octave-up-down-up-down.wav", sr=22050)
sp.compute_fft(window=1000, hop=100)
sp.compute_pca(npc=20, whiten=True)
sp.set_target(load_events('events_temp.json'))
# sp.plot_principal_spectra(log=True).show()
m = RFClassifier.from_fftpc(sp)
m.fit()
# m.plot_variable_importance().show()

pred = m.predict(sp)
pred_long = pred.melt(id_vars=['ts'])
fig = px.line(pred_long,x='ts', y='value', color='variable')
fig.show()