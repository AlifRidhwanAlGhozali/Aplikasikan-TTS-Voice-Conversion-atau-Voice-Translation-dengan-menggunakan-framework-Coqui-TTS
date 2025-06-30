from TTS.api import TTS

# Inisialisasi model voice conversion
tts = TTS("voice_conversion_models/multilingual/vctk/freevc24", progress_bar=True, gpu=False)

# Voice Conversion: source.wav → output_vc.wav
tts.voice_conversion_to_file(
    source_wav="source.wav",  # file suara input kamu
    target_wav="source.wav",  # file suara target (bisa sama untuk test)
    file_path="output_vc.wav"
)