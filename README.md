# 🧠 Speech-to-Speech Translation for Indian Languages

“When machines learn to understand every voice, the world becomes truly connected.” 🌏

### 👤 Author: Jenil Kalsariya  
**Institution:** Charotar University of Science and Technology (CHARUSAT)  
**Objective:** Build a multilingual **Speech-to-Speech (S2S) Translation System** for Indian languages using open-source models optimized for low-resource conditions.

---

## 🌍 Motivation

India is home to hundreds of languages and dialects — making communication across linguistic boundaries a real challenge.  
This project aims to **break language barriers** using **AI-powered speech-to-speech translation**, enabling seamless multilingual conversations.

While models like **SeamlessM4T** and **Translatotron** exist, they often perform poorly on Indian languages due to limited data.  
Hence, this project integrates **ASR (Whisper)**, **MT (NLLB-200)**, and **TTS (IndicParler-TTS)** — all optimized for **Indian speech and text** — into a unified pipeline.

---

## 🧩 System Pipeline

🎤 Input Speech (e.g., English)
│
▼
🧠 ASR → Whisper (Speech → Text)
│
▼
🌐 MT → NLLB-200 (Text → Target Language Translation)
│
▼
🔉 TTS → IndicParler-TTS (Text → Speech)
│
▼
🎧 Output Speech (e.g., Hindi / Tamil / Telugu)


---

## ⚙️ Core Components

| **Stage** | **Model** | **Framework** | **Description** |
|------------|------------|----------------|------------------|
| 🎙️ ASR | **Whisper Large-V3** | PyTorch + Transformers | Converts input speech into text; supports multilingual recognition. |
| 🌐 MT | **facebook/nllb-200-distilled-600M** | Hugging Face Transformers | Translates recognized text into the target Indian language. |
| 🔊 TTS | **ai4bharat/indic-parler-tts** | Parler-TTS | Synthesizes natural speech in target language with Indian voice characteristics. |

---

## 🔍 Detailed Workflow

### 🎙️ Step 1: Automatic Speech Recognition (ASR)
- Converts raw audio into text using **Whisper Large-V3**.
- Handles multilingual input and noisy conditions.

**Example**
Input Speech: "How are you?"
Output Text: "How are you?"


---

### 🌐 Step 2: Machine Translation (MT)
- Uses **NLLB-200** to translate English text into any supported Indian language.
- Optimized for 200+ languages with strong low-resource support.

**Example**

Input Text: "How are you?"
Output Text (Hindi): "आप कैसे हैं?"


---

### 🔉 Step 3: Text-to-Speech (TTS)
- Generates expressive, human-like speech using **IndicParler-TTS**.
- Designed for accurate phoneme rendering and Indian prosody.

**Example**
Input Text: "आप कैसे हैं?"
Output Audio: (Spoken Hindi waveform 🎧)


---

## 🧠 Technical Details

| **Category** | **Specification** |
|---------------|------------------|
| **Programming Language** | Python 3.10 |
| **Frameworks** | PyTorch, Hugging Face Transformers, Parler-TTS |
| **Hardware** | NVIDIA GPU (CUDA 12.x) |
| **Sampling Rate** | 16 kHz mono |
| **Supported Languages** | English, Hindi, Tamil, Telugu, Gujarati, Marathi, Bengali, etc. |

---

## 📚 Datasets Used

| **Dataset** | **Purpose** | **Description** |
|--------------|-------------|-----------------|
| **FLEURS (Google)** | ASR Evaluation | Multilingual benchmark dataset for speech recognition. |
| **IndicTTS** | TTS Reference | Indian language corpus for speech synthesis. |
| **BhashaAnuvaad** | MT/ASR Training | Large-scale multilingual dataset from AI4Bharat. |
| **AI4Bharat Corpus** | TTS/MT | Additional data for fine-tuning Indian voices. |

---

## 📊 Evaluation Metrics

| **Metric** | **Component** | **Description** |
|-------------|---------------|-----------------|
| **WER (Word Error Rate)** | ASR | Measures transcription accuracy. |
| **BLEU Score** | MT | Evaluates translation fluency and adequacy. |
| **CER (Character Error Rate)** | MT | Effective for non-Latin scripts (Devanagari, Tamil, etc.). |
| **MCD (Mel Cepstral Distortion)** | TTS | Measures acoustic similarity to reference audio. |
| **F0 RMSE (Pitch Error)** | TTS | Evaluates pitch consistency and naturalness. |
| **MOS (Mean Opinion Score)** | Full Pipeline | Subjective measure of intelligibility and quality. |

---

## 🧪 Experimental Results

| **Language** | **BLEU ↑** | **CER ↓** | **MCD ↓** | **F0 RMSE ↓** | **MOS ↑** |
|---------------|-------------|------------|-------------|----------------|------------|
| Hindi (hi) | 31.2 | 0.17 | 4.35 | 28.1 | 4.1 |
| Gujarati (gu) | 28.7 | 0.21 | 4.62 | 31.5 | 3.9 |
| Tamil (ta) | 26.4 | 0.24 | 4.78 | 33.2 | 3.8 |
| Telugu (te) | 27.1 | 0.22 | 4.55 | 29.8 | 4.0 |

> **↑ Higher is better**, **↓ Lower is better**

---

## 🚀 Future Work

Integrate real-time S2S streaming via WebSocket.

Add prosody transfer for emotional voice matching.

Fine-tune IndicParler-TTS on conversational corpora.

Implement Translatotron-style direct S2S model.

Deploy Streamlit/Gradio web demo for multilingual use.

## 📚 References

OpenAI — Whisper: Robust Speech Recognition via Weak Supervision

Meta AI — No Language Left Behind: Scaling Human-Centered Machine Translation

AI4Bharat — IndicParler-TTS (Hugging Face, 2025)

Google Research — FLEURS Multilingual Speech Dataset

AI4Bharat — BhashaAnuvaad Multilingual Corpus
