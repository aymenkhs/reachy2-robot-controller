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
