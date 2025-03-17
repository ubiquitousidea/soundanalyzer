from ..util.soundprocess import SoundProcess

sp = SoundProcess.from_wav('soudns/diatonicseventhchords.wav')
sp.setup(window=1200, hop=30, nclust=9, pc_min=1, npc=15)
sp.run()

output = sp.pc_smoothed
print(output)