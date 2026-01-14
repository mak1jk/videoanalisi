# Video Keyframe Extractor - Piano di Progetto

## 🎯 Obiettivo
Creare un tool che estrae automaticamente i frame più rilevanti da un video educativo/presentazione e li inserisce nel punto corretto della trascrizione, generando un documento HTML interattivo.

---

## 🏗️ Architettura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                        VIDEO INPUT                               │
│                    (MP4, MKV, AVI, etc.)                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PROCESSING PIPELINE                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Audio     │  │   Video     │  │   Scene Detection       │  │
│  │ Extraction  │  │  Sampling   │  │   (PySceneDetect)       │  │
│  │  (FFmpeg)   │  │  (OpenCV)   │  │                         │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
│         │                │                      │                │
│         ▼                ▼                      ▼                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  WhisperX   │  │    CLIP     │  │   Candidate Frames      │  │
│  │Transcription│  │ Embeddings  │  │      Selection          │  │
│  │ + Timestamps│  │             │  │                         │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
│         │                │                      │                │
│         └────────────────┼──────────────────────┘                │
│                          ▼                                       │
│              ┌───────────────────────┐                          │
│              │   VLM Frame Selector  │                          │
│              │ (Gemini/Claude/GPT-4) │                          │
│              └───────────┬───────────┘                          │
│                          │                                       │
└──────────────────────────┼──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     HTML OUTPUT GENERATOR                        │
│         Documento interattivo con testo + immagini inline        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Struttura Moduli

```
video_keyframe_extractor/
│
├── main.py                     # Entry point + Gradio UI
├── requirements.txt            # Dipendenze
├── config.py                   # Configurazione API keys, parametri
│
├── core/
│   ├── __init__.py
│   ├── audio_extractor.py      # Estrazione audio da video (FFmpeg)
│   ├── transcriber.py          # WhisperX transcription + alignment
│   ├── frame_sampler.py        # Campionamento frame (OpenCV)
│   ├── scene_detector.py       # Rilevamento scene (PySceneDetect)
│   ├── embeddings.py           # CLIP embeddings per frame e testo
│   └── frame_selector.py       # VLM-based frame selection
│
├── vlm_providers/
│   ├── __init__.py
│   ├── base.py                 # Abstract base class
│   ├── gemini_provider.py      # Google Gemini API
│   ├── claude_provider.py      # Anthropic Claude API
│   ├── openai_provider.py      # OpenAI GPT-4V API
│   └── local_provider.py       # Qwen2-VL locale (opzionale)
│
├── output/
│   ├── __init__.py
│   ├── html_generator.py       # Generatore HTML interattivo
│   ├── templates/
│   │   └── document.html       # Template Jinja2
│   └── assets/
│       └── style.css           # Stili CSS
│
└── utils/
    ├── __init__.py
    ├── video_utils.py          # Utility video (durata, fps, etc.)
    └── image_utils.py          # Resize, compressione immagini
```

---

## 🖥️ Interfaccia Gradio

### Layout Proposto

```
┌─────────────────────────────────────────────────────────────────┐
│  🎬 Video Keyframe Extractor                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  📁 Upload Video                              [Browse]   │    │
│  │  ─────────────────────────────────────────────────────  │    │
│  │  Drag & drop or click to upload (MP4, MKV, AVI, MOV)    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌──────────────────────┐  ┌──────────────────────────────┐     │
│  │ 🤖 VLM Provider      │  │ ⚙️ Impostazioni              │     │
│  │ ○ Gemini Flash       │  │                              │     │
│  │ ○ Claude Sonnet      │  │ Frame rate sampling: [1] fps │     │
│  │ ○ GPT-4o             │  │ Max frames per segment: [5]  │     │
│  │ ○ Qwen2-VL (locale)  │  │ Lingua trascrizione: [auto]  │     │
│  └──────────────────────┘  │ Soglia similarità: [0.3]     │     │
│                            └──────────────────────────────┘     │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ 🔑 API Key (se necessaria)                              │    │
│  │ [________________________________________________]      │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│          [ 🚀 Processa Video ]                                  │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  📊 Progresso                                                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ ████████████░░░░░░░░░░░░░░░░░░░░  35%                   │    │
│  │ Status: Transcribing audio with WhisperX...             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  📄 Output                                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                                                          │    │
│  │  [Preview HTML]              [Download HTML] [Download   │    │
│  │                                              Images ZIP] │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │                                                  │    │    │
│  │  │   (Anteprima documento HTML generato)           │    │    │
│  │  │                                                  │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Tabs Aggiuntive (opzionali)

1. **📝 Trascrizione**: Mostra la trascrizione completa con timestamp
2. **🖼️ Frame Estratti**: Gallery di tutti i keyframe selezionati
3. **⚙️ Configurazione Avanzata**: Parametri dettagliati

---

## 📄 Schema Output HTML

### Struttura Documento

```html
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <title>{video_title} - Trascrizione con Keyframe</title>
    <style>
        /* Stili embedded per portabilità */
        body { font-family: 'Segoe UI', sans-serif; max-width: 900px; margin: 0 auto; }
        .segment { margin: 2rem 0; padding: 1rem; border-left: 3px solid #4A90D9; }
        .timestamp { color: #666; font-size: 0.85rem; }
        .keyframe { max-width: 100%; border-radius: 8px; margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .keyframe-caption { font-style: italic; color: #555; font-size: 0.9rem; }
        /* Lightbox per click su immagini */
        .lightbox { display: none; position: fixed; ... }
    </style>
</head>
<body>
    <header>
        <h1>{video_title}</h1>
        <p class="meta">Durata: {duration} | Generato il: {date}</p>
    </header>

    <main>
        <!-- Segmento 1 -->
        <section class="segment" id="seg-1">
            <span class="timestamp">[00:00:15 - 00:01:30]</span>
            <p class="text">
                Oggi parleremo di machine learning e delle sue applicazioni
                nel campo della computer vision. Come potete vedere in questa slide...
            </p>
            <figure>
                <img class="keyframe" src="frames/frame_00_01_12.jpg"
                     alt="Slide: Introduzione al Machine Learning"
                     onclick="openLightbox(this)">
                <figcaption class="keyframe-caption">
                    Slide mostrata durante la spiegazione
                </figcaption>
            </figure>
        </section>

        <!-- Segmento 2 -->
        <section class="segment" id="seg-2">
            <span class="timestamp">[00:01:30 - 00:03:45]</span>
            <p class="text">
                Il primo concetto fondamentale è quello delle reti neurali...
            </p>
            <!-- Nessun keyframe se non rilevante -->
        </section>

        <!-- ... altri segmenti ... -->
    </main>

    <script>
        // Lightbox interattivo
        function openLightbox(img) { ... }
        // Navigazione con timestamp cliccabili
        // Indice laterale (opzionale)
    </script>
</body>
</html>
```

### Funzionalità Interattive

1. **Click su immagine** → Lightbox fullscreen
2. **Click su timestamp** → Scroll al segmento (o link a video originale se disponibile)
3. **Indice laterale** → Navigazione rapida tra sezioni
4. **Ricerca testo** → Trova parole nella trascrizione
5. **Dark mode toggle** → Tema chiaro/scuro

---

## 🔄 Pipeline di Processing - Dettaglio

### Step 1: Estrazione Audio
```python
def extract_audio(video_path: str) -> str:
    """Estrae traccia audio in WAV per WhisperX"""
    # FFmpeg: video → audio.wav (16kHz mono)
    return audio_path
```

### Step 2: Trascrizione con WhisperX
```python
def transcribe(audio_path: str, language: str = "auto") -> TranscriptionResult:
    """
    Returns:
        segments: [
            {
                "start": 0.0,
                "end": 5.2,
                "text": "Oggi parleremo di...",
                "words": [
                    {"word": "Oggi", "start": 0.0, "end": 0.3},
                    {"word": "parleremo", "start": 0.4, "end": 0.9},
                    ...
                ]
            },
            ...
        ]
    """
```

### Step 3: Scene Detection + Frame Sampling
```python
def get_candidate_frames(video_path: str, segment: dict) -> List[Frame]:
    """
    Per ogni segmento di trascrizione:
    1. Rileva scene changes nel range temporale
    2. Campiona frame aggiuntivi (es. 1 fps)
    3. Filtra frame troppo simili (CLIP similarity > 0.95)
    4. Ritorna max N candidati
    """
```

### Step 4: VLM Frame Selection
```python
def select_best_frame(
    text: str,
    candidate_frames: List[Frame],
    vlm_provider: VLMProvider
) -> Optional[Frame]:
    """
    Prompt al VLM:
    "Dato il testo: '{text}'
     Quale frame (1-N) rappresenta meglio visivamente questo contenuto?
     Rispondi con il numero o 'NESSUNO' se nessun frame è rilevante."
    """
```

### Step 5: HTML Generation
```python
def generate_html(
    segments: List[Segment],
    selected_frames: Dict[int, Frame],
    output_path: str
) -> str:
    """Genera documento HTML con Jinja2 template"""
```

---

## 📋 Dipendenze

```txt
# Core
ffmpeg-python>=0.2.0
opencv-python>=4.8.0
numpy>=1.24.0

# Transcription
whisperx>=3.1.0
torch>=2.0.0
torchaudio>=2.0.0

# Scene Detection
scenedetect>=0.6.2

# Embeddings
transformers>=4.35.0
sentence-transformers>=2.2.0
# oppure: open-clip-torch>=2.20.0

# VLM Providers
google-generativeai>=0.3.0     # Gemini
anthropic>=0.18.0              # Claude
openai>=1.12.0                 # GPT-4

# GUI
gradio>=4.19.0

# Output
jinja2>=3.1.0
Pillow>=10.0.0

# Utils
tqdm>=4.65.0
python-dotenv>=1.0.0
```

---

## ⚡ Ottimizzazioni Previste

1. **Caching embeddings**: Salva CLIP embeddings per evitare ricalcolo
2. **Batch processing**: Processa più frame insieme con CLIP
3. **Lazy loading**: Carica frame solo quando necessario
4. **Compressione immagini**: WebP per output più leggero
5. **Parallel processing**: Trascrizione e frame extraction in parallelo

---

## 🚀 Roadmap Implementazione

### Fase 1: Core Pipeline (MVP)
- [ ] Audio extraction + WhisperX transcription
- [ ] Basic frame sampling (uniform)
- [ ] Single VLM provider (Gemini)
- [ ] Basic HTML output

### Fase 2: Intelligent Selection
- [ ] PySceneDetect integration
- [ ] CLIP pre-filtering
- [ ] Multiple VLM providers

### Fase 3: GUI Completa
- [ ] Gradio interface
- [ ] Progress tracking
- [ ] Preview e download

### Fase 4: Polish
- [ ] HTML interattivo avanzato
- [ ] Ottimizzazioni performance
- [ ] Error handling robusto

---

## ❓ Decisioni Aperte

1. **Segmentazione testo**: Per frase? Per N secondi? Per "paragrafo semantico"?
2. **Soglia "nessun frame"**: Quando il VLM decide che nessun frame è rilevante?
3. **Immagini inline vs link**: Base64 embedded o file separati?
4. **Gestione video lunghi**: Streaming o processing completo?
