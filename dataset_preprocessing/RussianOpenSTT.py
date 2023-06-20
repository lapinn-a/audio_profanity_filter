import sys
import os
import wave
import json
import pandas as pd
from vosk import Model, KaldiRecognizer, SetLogLevel

words = ["\u0431\u043B\u044F", "\u043F\u0438\u0437\u0434", "\u0445\u0443\u0439", "\u0445\u0443\u044F", "\u0445\u0443\u0435", "\u0445\u0443\u0451", "\u0445\u0443\u044E", "\u0445\u0443\u0451\u043C", "\u0435\u0431\u0430", "\u0435\u0431\u043B", "\u0435\u0431\u0438", "\u0435\u0431\u044B", "\u0451\u0431\u044B", "\u0435\u0431\u0443", "\u0435\u0431\u0435\u0442", "\u0435\u0431\u0451\u0442", "\u0435\u0431\u0435\u0439", "\u0435\u0431\u0435\u043D", "\u043E\u0435\u0431", "\u043E\u0451\u0431", "\u0430\u0451\u0431", "\u0430\u0435\u0431"]

dir = os.path.dirname(__file__)
if not os.path.isdir(os.path.join(dir, 'processed')):
  os.mkdir(os.path.join(dir, 'processed'))
data_dir = os.path.join(dir, 'processed', '0')
if not os.path.isdir(data_dir):
  os.mkdir(data_dir)

def get_data():
  i = 0
  sp = pd.read_csv(os.path.join(sys.argv[1], "df.csv"))
  files = sp[sp['text'].str.contains('|'.join(words))]['audio_id'].tolist()
  length = len(files)
  for file in files:
    wf = wave.open(os.path.join(sys.argv[1], "audio_files", file + ".wav"), "rb")
    rec = KaldiRecognizer(model, 16000)
    rec.SetWords(True)
    rec.SetPartialWords(True)
    while True:
      data = wf.readframes(2000)
      if len(data) == 0:
        break
      rec.AcceptWaveform(data)
    jres = json.loads(rec.FinalResult())
    if "result" in jres:
      for word in jres["result"]:
        if word["conf"] >= 0.5 and any(w in word["word"] for w in words):
          wf.setpos(int(word["start"] * wf.getframerate()))
          data = wf.readframes(int((word["end"] - word["start"]) * wf.getframerate()))
          if not os.path.isdir(os.path.join(data_dir, word["word"])):
            os.mkdir(os.path.join(data_dir, word["word"]))
          with wave.open(os.path.join(data_dir, word["word"], str(i) + ".wav"), 'w') as outfile:
            outfile.setnchannels(wf.getnchannels())
            outfile.setsampwidth(wf.getsampwidth())
            outfile.setframerate(wf.getframerate())
            outfile.setnframes(int(len(data) / wf.getsampwidth()))
            outfile.writeframes(data)
          i += 1

get_data()