from ..util.soundprocess import SoundProcess
from ..util.classifier import XGClassifier, RFClassifier, ClassifierModel
from ..util.util import load_events, load_object
import json


# sp = SoundProcess.from_wav('soudns/diatonicseventhchords.wav', sr=44100)
# sp = SoundProcess.from_wav('soudns/g-g#-a-major-scales-with-mistakes.wav', sr=44100)
# sp.compute_fft(window=2000, hop=50)
# sp.compute_pca(npc=20, whiten=False)
# sp.compute_clusters(clusters=9, pc_min=1)
# split_output = sp.split()  # split the sound into clusters, write out wav files

# sp.set_target(load_events('events_temp.json'))

sp = load_object('models/diatonicseventhchords_soundprocess.bin')
# sp = load_object('models/g-g#-a-major-scales-with-mistakes_soundprocess.bin')

sp.plot_principal_spectra(log=True).show()

# m = RFClassifier.from_fftpc(sp)
# m.fit()
m = load_object('models/diatonicseventhchords.bin')

m.plot_variable_importance().show()

