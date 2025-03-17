import os
import json
import pickle
from pydub import AudioSegment


def delete_all_files(directory):
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory path.")
        return
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                print(f"Deleted: {file_path}")
            elif os.path.isdir(file_path):
                print(f"Found directory (not deleted): {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def convert_m4a_to_wav(input_file, output_file):
    # Load your M4A file
    audio = AudioSegment.from_file(input_file, format='m4a')
    
    # Export as WAV
    audio.export(output_file, format='wav')
    print(f"Converted {input_file} to {output_file}")


def convert_m4a_files_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".m4a"):
            input_file = os.path.join(directory_path, filename)
            output_file = os.path.join(directory_path, filename.replace('.m4a', '.wav'))
            convert_m4a_to_wav(input_file, output_file)


def load_events(fname='events_temp.json'):
    with open(fname, 'r') as f:
        d = json.load(f)
    return d


def load_object(binary_file):
    with open(binary_file, 'rb') as f:
        bytes = f.read()
    m = pickle.loads(bytes)
    return m