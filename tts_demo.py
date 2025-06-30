from TTS.api import TTS

# Gunakan model Fairseq (contoh: bahasa Jerman, ganti ke bahasa lain jika ingin)
tts = TTS("tts_models/deu/fairseq/vits", progress_bar=True, gpu=False)

# Generate TTS
tts.tts_to_file(text="Hallo, dies ist ein Test mit Fairseq TTS.", file_path="output_tts.wav")
