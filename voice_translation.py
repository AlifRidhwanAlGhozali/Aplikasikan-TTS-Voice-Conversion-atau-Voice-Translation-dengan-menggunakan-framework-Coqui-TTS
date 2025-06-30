import whisper
from transformers import MarianMTModel, MarianTokenizer
from TTS.api import TTS

# STEP 1: Speech Recognition dengan Whisper
print("🔊 [1/3] Running Speech Recognition (ASR)...")
model_asr = whisper.load_model("small")
result = model_asr.transcribe("source.wav")
text_id = result["text"]
print("✅ Transkripsi (ID):", text_id)

# STEP 2: Translate ID -> EN
print("🔊 [2/3] Running Translation ID → EN ...")
model_name = 'Helsinki-NLP/opus-mt-id-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

translated = model.generate(**tokenizer([text_id], return_tensors="pt", padding=True))
text_en = tokenizer.decode(translated[0], skip_special_tokens=True)
print("✅ Hasil Terjemahan (EN):", text_en)

# STEP 3: TTS English dengan model Fairseq
print("🔊 [3/3] Generating TTS output in English (Fairseq model)...")
tts = TTS("tts_models/deu/fairseq/vits", progress_bar=True, gpu=False)
tts.tts_to_file(text=text_en, file_path="output_translate.wav")

print("✅ Voice Translation Pipeline selesai! File: output_translate.wav")