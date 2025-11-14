from sound import SoundClass as sc
import os
import random

class DataCollection:
    def __init__(self):
        self.name_path = "src/NAMES.txt"
        self.audio_path = "src/NAMES_audio_wav"
        pass

    def get_names(self):
        names = []
        with open(self.name_path, "r") as f:
            string = f.read()
            names = string.split("\n")
        return names


    def collect_audio(self):


        names = self.get_names()
        missing_names_audio = []
        for name in names:

            expected_files = [f"{name}{i:03d}" for i in range(1, 20 + 1)]
            existing_files = os.listdir(self.audio_path)
            missing_files = [f for f in expected_files if f"{f}.wav" not in existing_files]

            if missing_files:
                print(f"Missing files found for {name}, appending...")
                missing_names_audio += missing_files
        print(f"Found {len(missing_names_audio)} missing files.")

        #Shuffles list so same data isn't collected in a go to introduce variation
        if len(missing_names_audio) > 20:
            random.shuffle(missing_names_audio)

        for file in missing_names_audio:
            audio_data,fs = sc.record_ns()
            sc.save(audio_data, fs, f"{self.audio_path}/{file}")
            confirmation = input("Record next? y/n")
            if confirmation == "y":
                continue
            else:
                break






