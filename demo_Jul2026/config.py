# config.py

import os

# =========================================================
# Reachy
# =========================================================

ROBOT_HOST = "192.168.0.120"  # Change if Reachy's IP changes
USE_REACHY = True

# Keep this True for first testing.
# After everything is safe, you can set False for smoother continuous demo.
SAFETY_PAUSES = False

# Duration used for every arm pose transition in robot/motion.py.
ARM_MOVE_DURATION_SECONDS = 2.0
DAB_HOLD_SECONDS = 10.0


# =========================================================
# OpenRouter
# =========================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = "openai/gpt-4o-mini"


# =========================================================
# Kokoro TTS
# =========================================================

KOKORO_LANG_CODE = "a"
KOKORO_VOICE = "am_adam"  # senior male style
OUTPUT_FILE = "reachy_response.wav"
USE_SPEECH_OUTPUT = True


# =========================================================
# Metallic / cute-robot voice FX (post OpenAI PCM)
# =========================================================
# Applied in voice.py to OpenAI TTS and realtime speech output.
# Set WET = 0 to disable FX entirely.

PITCH_SEMITONES = 2.0       # 0=none, 1–2=cute lift, 5+=chipmunk
RING_HZ = 35.0              # 20–30 soft wobble, 60–120 harsher
RING_MIX = 0.14             # 0=off, 0.1–0.2 subtle, 0.3+ obviously synthetic
BITCRUSH_BITS = 11.0        # 16=clean, 11–12 grain, 8=harsh
METALLIC_GAIN_DB = 2.5      # 0=off, 1–3 sparkle, 8+=tin-can
WET = 1.0                   # 1=full FX, 0.5=half dry / half processed


# =========================================================
# Microphone / STT
# =========================================================

SAMPLE_RATE = 16000
MIC_INPUT_FILE = "visitor_question.wav"
WHISPER_MODEL_SIZE = "base"

# Silence detection
CHUNK_DURATION = 0.2
SILENCE_THRESHOLD = 0.015
SILENCE_SECONDS_TO_STOP = 1.5
MAX_RECORD_SECONDS = 15
MIN_RECORD_SECONDS = 1.0


# =========================================================
# Demo conversation
# =========================================================

MAX_QUESTIONS_PER_VISITOR = 5
