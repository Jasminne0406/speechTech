import os
import re
import uuid
import shutil
import tempfile
import logging
import numpy as np
import torch
import librosa
import soundfile as sf

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from transformers import Wav2Vec2BertProcessor, Wav2Vec2BertForCTC
from TTS.utils.synthesizer import Synthesizer
from TTS.tts.utils.text import cleaners


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KhmerSpeechAPI")

app = FastAPI(
    title="Khmer ASR and TTS API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

speech_models = {}

os.makedirs("outputs", exist_ok=True)


def khmer_cleaners(text):
    text = " ".join(text.split())
    text = re.sub(
        r"[^\u1780-\u17FF\u19E0-\u19FF0-9\s.,!?;:\-()'\"]",
        "",
        text
    )
    return text.strip()


setattr(cleaners, "khmer_cleaners", khmer_cleaners)


@app.on_event("startup")
async def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = torch.cuda.is_available()

    # ASR_REPO = "./models/khmer-wav2vec-bert"
    # TTS_REPO = "./models/khmer-vits"

    ASR_REPO = "Prakmlis/w2v-bert-2.0-khmer-customize-data-final"
    TTS_REPO = "Prakmlis/khmer-vits-mms"

    TTS_CHECKPOINT = (
        "vits_khmer/"
        "vits_khmer-July-14-2026_02+08AM-0000000/"
        "best_model.pth"
    )

    TTS_CONFIG = (
        "vits_khmer/"
        "vits_khmer-July-14-2026_02+08AM-0000000/"
        "config.json"
    )

    logger.info(f"Using device: {device}")

    logger.info("Loading ASR model...")
    speech_models["asr_processor"] = Wav2Vec2BertProcessor.from_pretrained(ASR_REPO)
    speech_models["asr_model"] = Wav2Vec2BertForCTC.from_pretrained(ASR_REPO).to(device)
    speech_models["asr_model"].eval()
    speech_models["device"] = device

    logger.info("Downloading TTS model...")
    # checkpoint_path = "./models/khmer-vits/best_model.pth"
    # config_path = "./models/khmer-vits/config.json"
    checkpoint_path = hf_hub_download(repo_id=TTS_REPO, filename=TTS_CHECKPOINT)
    config_path = hf_hub_download(repo_id=TTS_REPO, filename=TTS_CONFIG)


    logger.info("Loading TTS synthesizer...")
    synth = Synthesizer(
        tts_checkpoint=checkpoint_path,
        tts_config_path=config_path,
        use_cuda=use_cuda
    )

    synth.tts_model.length_scale = 0.90
    synth.tts_model.noise_scale = 0.3
    synth.tts_model.inference_noise_scale = 0.3
    synth.tts_model.noise_scale_dp = 0.55
    synth.tts_model.inference_noise_scale_dp = 0.55

    speech_models["tts_synth"] = synth

    logger.info("Models loaded successfully.")


@app.get("/")
def root():
    return {
        "message": "Khmer ASR and TTS API is running",
        "asr_api": "/api/v1/speech/asr",
        "tts_api": "/api/v1/speech/tts",
        "docs": "/docs"
    }


class ASRResponse(BaseModel):
    transcription: str


@app.post("/api/v1/speech/asr", response_model=ASRResponse)
async def asr_endpoint(file: UploadFile = File(...)):
    if "asr_model" not in speech_models:
        raise HTTPException(status_code=503, detail="ASR model is not loaded.")

    suffix = os.path.splitext(file.filename)[1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        shutil.copyfileobj(file.file, temp_audio)
        temp_audio_path = temp_audio.name

    try:
        speech_array, _ = librosa.load(temp_audio_path, sr=16000)

        processor = speech_models["asr_processor"]
        model = speech_models["asr_model"]
        device = speech_models["device"]

        inputs = processor(
            speech_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        decoded_text = processor.batch_decode(predicted_ids)[0]

        return ASRResponse(transcription=decoded_text.strip())

    except Exception as e:
        logger.error(f"ASR error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


class TTSRequest(BaseModel):
    text: str = Field(..., example="សួស្តី")


@app.post("/api/v1/speech/tts")
async def tts_endpoint(payload: TTSRequest):
    if "tts_synth" not in speech_models:
        raise HTTPException(status_code=503, detail="TTS model is not loaded.")

    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")

    try:
        synth = speech_models["tts_synth"]

        clean_text = payload.text.replace(" ", ", ")

        wav = synth.tts(clean_text)
        wav = np.array(wav, dtype=np.float32)

        if wav.size == 0:
            raise HTTPException(status_code=422, detail="Generated waveform is empty.")

        if np.isnan(wav).any():
            raise HTTPException(status_code=422, detail="Generated waveform contains NaN.")

        peak = np.max(np.abs(wav))
        if peak > 0:
            wav = wav / peak

        wav = wav * 0.9
        wav = np.clip(wav, -1.0, 1.0)

        output_path = f"outputs/tts_{uuid.uuid4()}.wav"

        sf.write(
            output_path,
            wav,
            synth.output_sample_rate,
            format="WAV"
        )

        logger.info("TTS generated successfully")
        logger.info(f"Text: {clean_text}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Duration: {len(wav) / synth.output_sample_rate:.2f}s")

        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="tts_output.wav"
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
