from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs

# Print all supported languages
print(f"Total supported languages: {len(supported_langs)}")
print(supported_langs)

for lang in supported_langs:
    if "Arab" in lang:
        print(lang)

# Check if a specific language is supported
if "eng_Latn" in supported_langs:
    print("English (Latin script) is supported!")