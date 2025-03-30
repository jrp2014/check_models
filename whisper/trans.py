import mlx_whisper

speech_file = "/Users/jrp/Documents/Jaro Pavel - Life Story.wav"
#speech_file = "/Users/jrp/.cache/whisper/alice.mp3"


result = mlx_whisper.transcribe(speech_file,path_or_hf_repo=f"mlx-community/whisper-large-v3-mlx",verbose=True, fp16=True, condition_on_previous_text=True,  word_timestamps=True, hallucination_silence_threshold=2)

f=open("result.txt","w+")

for segment in result["segments"]:
     print(segment["text"], file=f)

f.close()
