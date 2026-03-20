# 🎬 Video Keyframe Extractor

Un tool Python che estrae automaticamente i frame più rilevanti da video educativi/presentazioni e li inserisce nel punto corretto della trascrizione, generando un documento HTML interattivo.

## ✨ Funzionalità

- **🎙️ Trascrizione Automatica**: Utilizza Whisper di OpenAI per trascrivere l'audio con timestamp precisi
- **🖼️ Estrazione Frame Intelligente**: Campiona frame dal video e seleziona quelli più rilevanti usando AI
- **🧠 Segmentazione Semantica**: Divide la trascrizione in sezioni logiche usando LLM
- **🎯 Selezione VLM**: Utilizza modelli Vision-Language (Gemini, Claude, GPT-4) per selezionare i frame migliori
- **📄 Output HTML Interattivo**: Genera documenti HTML con testo e immagini inline, navigabili e responsive
- **🛡️ Rendering Sicuro HTML**: Escape automatico del contenuto testuale nel report finale
- **🌐 Networking più robusto**: Timeout e gestione errori HTTP esplicita su integrazione OpenRouter
- **🎞️ Gestione FPS difensiva**: Validazione FPS per evitare crash su video malformati

## 🚀 Installazione

### Prerequisiti

- Python 3.8+
- FFmpeg installato sul sistema
- (opzionale, per URL/YouTube) `yt-dlp` (installato via `pip install -r requirements.txt`)
- (per Vertex AI) credenziali Google Cloud (ADC), ad esempio: `gcloud auth application-default login`

### Installazione FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**Windows:**
Scarica da [ffmpeg.org](https://ffmpeg.org/download.html) e aggiungi al PATH

### Installazione Progetto

1. Clona il repository:
```bash
git clone <repository-url>
cd videoanalisi
```

2. Crea e attiva l'ambiente virtuale:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oppure
venv\Scripts\activate  # Windows
```

3. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

4. Configura le API keys:
```bash
cp .env.example .env
# Modifica .env con le tue API keys
```

## ⚙️ Configurazione

Crea un file `.env` nella root del progetto:

```env
# === Provider ===
# Usa Vertex AI (consigliato per video-native e chunking)
USE_VERTEX_AI=true
GOOGLE_CLOUD_PROJECT=your_gcp_project
GOOGLE_CLOUD_LOCATION=us-central1
VIDEOANALISI_GCS_BUCKET=your_gcs_bucket

# In alternativa (senza Vertex): usa API key Gemini Developer API
GOOGLE_API_KEY=your_google_api_key_here

# OpenRouter (opzionale)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# === Modelli ===
GEMINI_MODEL=gemini-3-flash-preview

# === Video-native (Gemini su Vertex) ===
VIDEO_NATIVE_ENABLED=true
VIDEO_MAX_CHUNK_SECONDS_WITH_AUDIO=2700
VIDEO_MAX_CHUNK_SECONDS_NO_AUDIO=3600
VIDEO_CHUNK_OVERLAP_SECONDS=15
VIDEO_TIMELINE_MAX_EVENTS=60
VIDEO_MEDIA_RESOLUTION=LOW

# === Trascrizione ===
# Backend per la trascrizione audio (groq | local | lascia vuoto per auto-detect)
# "groq"  → Groq Whisper API (veloce, richiede GROQ_API_KEY)
# "local" → faster-whisper locale (offline, nessuna API key, richiede: pip install faster-whisper)
# Se non impostato: usa "groq" se GROQ_API_KEY è presente, altrimenti "local"
# TRANSCRIBER=local
GROQ_API_KEY=your_groq_api_key_here

# === Pipeline legacy ===
FRAME_SAMPLE_RATE=1
CLIP_SIMILARITY_THRESHOLD=0.90
MIN_SEGMENT_DURATION=30.0
```

**Ottieni le API keys:**
- [Google AI Studio](https://aistudio.google.com/app/apikey) - Gemini
- [OpenAI](https://platform.openai.com/api-keys) - GPT-4
- [Anthropic](https://console.anthropic.com/settings/keys) - Claude
- [OpenRouter](https://openrouter.ai/keys) - Accesso multi-modello

## 🖥️ Utilizzo

### Interfaccia Web (Gradio)

```bash
python app.py
```

Apri il browser all'indirizzo mostrato (di solito `http://127.0.0.1:7860`).

Puoi caricare un file locale oppure fornire un URL (YouTube/direct).

### Backend API opzionali

- **Gemini / Vertex AI**: percorso principale consigliato per analisi video e selezione frame.
- **OpenRouter**: opzionale per accesso multi-modello dove previsto dal progetto.
- **Groq**: opzionale per integrazioni di trascrizione/fallback compatibili con Groq Whisper. Se `GROQ_API_KEY` non è configurata, non selezionare un backend Groq: il percorso base del progetto continua a funzionare usando i provider già configurati (Gemini/Vertex, OpenAI, Anthropic o altri backend supportati).

### CLI (Command Line)

```bash
# Video locale (Vertex AI + chunking automatico se >45m)
python cli.py path/to/video.mp4 --use-vertex-ai --gcs-bucket YOUR_BUCKET --gcp-project YOUR_PROJECT

# Video da URL (YouTube o direct) -> download automatico
python cli.py --video-url "https://www.youtube.com/watch?v=..." --use-vertex-ai --gcs-bucket YOUR_BUCKET --gcp-project YOUR_PROJECT

# Usa trascrizione locale offline (faster-whisper, nessuna API key)
python cli.py path/to/video.mp4 --no-fast --transcriber local

# Usa trascrizione Groq (richiede GROQ_API_KEY)
python cli.py path/to/video.mp4 --no-fast --transcriber groq
```

### Semantic Search

Dopo aver generato un report HTML, puoi interrogare offline il file `data.json` prodotto nella cartella output.
La query viene codificata localmente con `EmbeddingsModel` e confrontata con le sezioni del report usando cosine similarity.

```bash
python cli.py --output-json output_documents/nome_progetto/data.json --query "cosa fa il personaggio a 2:30"
```

Output atteso:
- top-3 sezioni ordinate per rilevanza
- timestamp di inizio/fine
- score di similarità
- trascritto della sezione

La ricerca semantica funziona offline: usa solo embedding locali e non effettua chiamate API.

### Trascrizione locale (offline)

Per usare la trascrizione offline senza API key:

```bash
pip install faster-whisper
python cli.py path/to/video.mp4 --no-fast --transcriber local
```

`faster-whisper` è un'implementazione ottimizzata di Whisper basata su CTranslate2.
Funziona su CPU e GPU, non richiede connessione Internet né API key.
Se non è installato e si usa `--transcriber local`, il programma mostrerà un messaggio di errore chiaro con le istruzioni per l'installazione.

### Come Libreria Python

```python
from video_keyframe_extractor.core.orchestrator import FrameSelectorOrchestrator
from video_keyframe_extractor.vlm_providers.gemini_provider import GeminiProvider
from video_keyframe_extractor.output.html_generator import HTMLGenerator

# Inizializza
vlm_provider = GeminiProvider()
orchestrator = FrameSelectorOrchestrator(vlm_provider)
html_generator = HTMLGenerator()

# Processa video
result_data = orchestrator.process_video("path/to/video.mp4")

# Genera HTML
output_path = html_generator.generate_document(result_data, "my_project")
print(f"Documento generato: {output_path}")
```

## 📁 Struttura del Progetto

```
videoanalisi/
├── app.py                          # Interfaccia Gradio
├── cli.py                          # Interfaccia a linea di comando
├── requirements.txt                # Dipendenze Python
├── .env                            # Configurazione (non committare)
├── video_keyframe_extractor/
│   ├── config.py                   # Configurazione
│   ├── core/
│   │   ├── audio_extractor.py      # Estrazione audio (FFmpeg)
│   │   ├── transcriber.py          # Trascrizione (Whisper)
│   │   ├── frame_sampler.py        # Campionamento frame
│   │   ├── scene_detector.py       # Rilevamento scene
│   │   ├── embeddings.py           # CLIP embeddings
│   │   ├── candidate_selector.py   # Selezione candidati
│   │   ├── text_segmenter.py       # Segmentazione testo
│   │   └── orchestrator.py         # Pipeline principale
│   ├── vlm_providers/
│   │   ├── base.py                 # Classe base VLM
│   │   ├── gemini_provider.py      # Provider Google Gemini
│   │   └── openrouter_provider.py  # Provider OpenRouter
│   ├── output/
│   │   ├── html_generator.py       # Generatore HTML
│   │   ├── templates/
│   │   │   └── document.html       # Template HTML
│   │   └── assets/
│   │       └── style.css           # Stili CSS
│   └── utils/
│       └── __init__.py
└── output_documents/               # Output generati
```

## 🎬 Formati Video Supportati

- MP4
- AVI
- MOV
- MKV
- WEBM
- FLV
- WMV

## 🔧 Architettura

```
Video Input
    ↓
┌─────────────────────────────────────┐
│  Processing Pipeline                │
│  ┌──────────────┐  ┌─────────────┐ │
│  │ Audio        │  │ Video       │ │
│  │ Extraction   │  │ Sampling    │ │
│  │ (FFmpeg)     │  │ (OpenCV)    │ │
│  └──────┬───────┘  └──────┬──────┘ │
│         ↓                  ↓        │
│  ┌──────────────┐  ┌─────────────┐ │
│  │ Whisper      │  │ CLIP        │ │
│  │ Transcription│  │ Embeddings  │ │
│  └──────┬───────┘  └──────┬──────┘ │
│         └────────┬────────┘        │
│                  ↓                  │
│         ┌──────────────┐           │
│         │ VLM Selector │           │
│         │ (Gemini/     │           │
│         │  Claude/     │           │
│         │  GPT-4)      │           │
│         └──────┬───────┘           │
└────────────────┼────────────────────┘
                 ↓
┌─────────────────────────────────────┐
│  HTML Output Generator              │
│  Documento interattivo con          │
│  testo + immagini inline            │
└─────────────────────────────────────┘
```

## 🛠️ Pipeline di Processing

1. **Estrazione Audio**: FFmpeg estrae la traccia audio in WAV
2. **Trascrizione**: Whisper trascrive con timestamp a livello di parola
3. **Segmentazione**: LLM divide la trascrizione in sezioni semantiche
4. **Campionamento Frame**: OpenCV estrae frame a intervalli regolari
5. **Filtraggio**: CLIP rimuove frame troppo simili
6. **Selezione VLM**: Il modello VLM sceglie il frame migliore per ogni sezione
7. **Generazione HTML**: Jinja2 genera il documento finale

## ⚡ Ottimizzazioni

- **Video-native su Vertex**: Chunking automatico per video lunghi con overlap configurabile
- **Deduplica frame**: Filtraggio CLIP per ridurre ridondanza prima della selezione VLM
- **Batch embedding**: Encoding immagini in batch nel selettore candidati
- **Hardening I/O**: timeout HTTP, `raise_for_status`, validazione FPS e errori FFmpeg espliciti

## ✅ Test

Esegui la suite minima di regressione compatibile con la CI:

```bash
python -m pytest \
  tests/test_cli_range_handler.py \
  tests/test_frame_sampler.py \
  tests/test_html_generator.py \
  tests/test_openrouter_provider.py \
  -q
```

Nota CI: `requirements.txt` installa anche dipendenze pesanti per la pipeline completa.
La pipeline GitHub Actions usa comunque questo comando esplicito di regressione e installa anche `ffmpeg`, `libgl1` e `libglib2.0-0` sul runner Ubuntu, necessari per i test che importano OpenCV.

## 🐛 Troubleshooting

### Errore: "FFmpeg not found"
Assicurati che FFmpeg sia installato e nel PATH di sistema.

### Errore: "No module named 'whisper'"
Attiva l'ambiente virtuale prima di eseguire: `source venv/bin/activate`

### Errore: "CUDA out of memory"
Riduci il batch size o usa il modello Whisper "base" invece di "large".

### Errore: "API Key not found"
Verifica che il file `.env` esista e contenga le API keys valide.

## 🤝 Contributi

I contributi sono benvenuti! Per favore:

1. Fork il repository
2. Crea un branch per la feature (`git checkout -b feature/AmazingFeature`)
3. Commit le modifiche (`git commit -m 'Add some AmazingFeature'`)
4. Push al branch (`git push origin feature/AmazingFeature`)
5. Apri una Pull Request

## 📝 License

Questo progetto è rilasciato sotto licenza MIT.

## 🙏 Riconoscimenti

- [OpenAI Whisper](https://github.com/openai/whisper) per la trascrizione
- [Google Gemini](https://ai.google.dev/) per i modelli VLM
- [Sentence Transformers](https://www.sbert.net/) per CLIP embeddings
- [Gradio](https://gradio.app/) per l'interfaccia web
- [PySceneDetect](https://scenedetect.com/) per il rilevamento scene

---

**Creato con ❤️ per semplificare l'analisi di video educativi**
