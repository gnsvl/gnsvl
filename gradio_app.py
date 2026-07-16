"""Sesle konuşmacı tanıma için basit Gradio uygulaması.

Yalnızca açık rızayla kaydedilmiş kişilerle kullanılmalıdır.
"""

from pathlib import Path
import tempfile

import gradio as gr
import joblib
import numpy as np
import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier


TARGET_SAMPLE_RATE = 16_000
UNKNOWN_THRESHOLD = 0.60
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Hazır ses modeli yükleniyor. İlk açılışta birkaç dakika sürebilir.")
encoder = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_spkrec",
    run_opts={"device": DEVICE},
)


def extract_embedding(audio_path: str) -> np.ndarray:
    """Ses dosyasını ECAPA-TDNN ile sayısal ses imzasına dönüştürür."""
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0, keepdim=True)  # stereo ise mono yap
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, TARGET_SAMPLE_RATE
        )

    with torch.inference_mode():
        embedding = encoder.encode_batch(waveform.to(DEVICE))
    return embedding.squeeze().cpu().numpy().reshape(1, -1)


def predict(model_file, audio_file):
    if model_file is None:
        return "Önce speaker_model.joblib dosyasını yükle.", []
    if audio_file is None:
        return "Sonra test etmek istediğin ses dosyasını seç.", []

    saved_model = joblib.load(model_file)
    model = saved_model["model"]
    embedding = extract_embedding(audio_file)
    scores = model.predict_proba(embedding)[0]
    ranking = sorted(
        zip(model.classes_, scores), key=lambda item: item[1], reverse=True
    )
    best_person, best_score = ranking[0]

    if best_score < UNKNOWN_THRESHOLD:
        result = (
            f"Sonuç: Bilinmiyor\n\n"
            f"En yakın eşleşme: {best_person} (%{best_score * 100:.1f})\n"
            "Bu ses, kayıtlı kişilerden biriyle yeterince güçlü eşleşmedi."
        )
    else:
        result = f"Sonuç: {best_person}\n\nGüven: %{best_score * 100:.1f}"

    chart = {person: float(score) for person, score in ranking}
    return result, chart


with gr.Blocks(title="Sesle Konuşmacı Tanıma") as demo:
    gr.Markdown(
        "# Sesle Konuşmacı Tanıma\n"
        "Eğitilmiş modelinle yeni bir ses kaydının kayıtlı kişilerden hangisine "
        "daha çok benzediğini kontrol et. Yalnızca izinli ses kayıtlarını kullan."
    )
    with gr.Row():
        model_input = gr.File(
            label="Eğitilmiş model (speaker_model.joblib)", file_types=[".joblib"]
        )
        audio_input = gr.Audio(
            label="Test sesi", type="filepath", sources=["upload", "microphone"]
        )
    check_button = gr.Button("Sesi kontrol et", variant="primary")
    result_output = gr.Textbox(label="Tahmin", lines=4)
    scores_output = gr.Label(label="Olasılık skorları", num_top_classes=10)

    check_button.click(
        predict,
        inputs=[model_input, audio_input],
        outputs=[result_output, scores_output],
    )


if __name__ == "__main__":
    demo.launch()
