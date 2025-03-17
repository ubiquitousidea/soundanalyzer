from ..util.soundprocess import SoundProcess

sp = SoundProcess.from_wav('soudns/diatonicseventhchords.wav')

print(sp.wav_peaks)