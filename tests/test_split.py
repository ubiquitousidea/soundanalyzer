import matplotlib.pyplot as plt
from ..util.soundprocess import SoundProcess


sp = SoundProcess.from_wav('soudns/g-g#-a-major-scales-with-mistakes.wav', sr=22050)
sp.setup(window=800, hop=100, nclust=13, pc_min=2, npc=13)
sp.run()

sp.plot_sound_fft().show()

output_summary, histogram = sp.split()

plt.show()
