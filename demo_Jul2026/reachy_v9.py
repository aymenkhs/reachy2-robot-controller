import asyncio
import base64
import json
import os
import random
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
import numpy as np
import miniaudio
import sounddevice as sd
from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion

import time
import numpy as np
from reachy2_sdk import ReachySDK
from robot.connection import connect_reachy
from robot.motion import (
    speaking_motion,
    grasp_mic_init,
    mic_forward,
    mic_backward,
    gesture_67,
    fun_pose_liberty,
)



#from workflows.greeting_workflow import run_greeting
from workflows.grasp_workflow import run_grasp_demo
#from workflows.conversation_workflow import run_conversation
#from workflows.goodbye_workflow import run_goodbye


reachy = connect_reachy()
#run_grasp_demo(reachy,label="first grasp demonstration",)
grasp_mic_init(reachy,label="initial microphone grab",)
# =========================================================
# Configuration
# =========================================================

REALTIME_MODEL: Final = "gpt-realtime-2.1-mini"
WEB_SEARCH_MODEL: Final = "gpt-5.6"
VOICE: Final = "coral"
TTS_MODEL: Final = "gpt-4o-mini-tts"

LOOKUP_MESSAGE: Final = "Let me look that up online."
TREAT_RESPONSE: Final = "Would you like some treats?"

# Reachy performs one idle action after this much inactivity.
IDLE_INTERVAL_SECONDS: Final = 2 * 60
IDLE_YAWN_PROBABILITY: Final = 0.40

# Optional yawn MP3. If it is not found, Reachy generates a short
# spoken yawn instead. A relative path is resolved beside this script.
IDLE_YAWN_SOUND_PATH: Final = "yawn.mp3"
IDLE_YAWN_VOLUME: Final = 0.55

IDLE_PHRASES: Final = (
    "I'm still here if anyone needs a hand.",
    "Beep boop... it is rather peaceful here.",
    "I wonder who will visit me next.",
    "Feel free to ask me something about Deakin.",
    "Quiet campus moments are nice too.",
)

# Put the PDF in the same directory as this Python file, or
# replace this value with its full path.
DEAKIN_KNOWLEDGE_PDF: Final = (
    Path(__file__).resolve().parent
    / "deakin_info.pdf"
)

#Hey there! I’m Reachy, your friendly robot buddy at Deakin’s Waurn Ponds campus—happy to help out with a smile

# Change this to the location of your MP3 file.
# On macOS, for example:
# LOOKUP_MUSIC_PATH: Final = "/Users/yourname/Music/thinking.mp3"
LOOKUP_MUSIC_PATH: Final = "wait.mp3"
LOOKUP_MUSIC_VOLUME: Final = 0.30
ACTION_67_SOUND_PATH: Final = "67-sound.mp3"
ACTION_67_SOUND_VOLUME: Final = 0.70
ACTION_67_STEP_SECONDS: Final = 1.0

SAMPLE_RATE: Final = 24_000
CHANNELS: Final = 1
RECORD_SECONDS: Final = 6

# Local retrieval settings.
LOCAL_TOP_K: Final = 3
LOCAL_CHUNK_CHARACTERS: Final = 1_000
LOCAL_CHUNK_OVERLAP_LINES: Final = 4
LOCAL_MIN_SCORE: Final = 0.075
LOCAL_MAX_CONTEXT_CHARACTERS: Final = 2_400

# Prevent accidental endless tool-calling loops.
MAX_TOOL_ROUNDS: Final = 3


# =========================================================
# Questions that require current or more precise information
# =========================================================

DYNAMIC_QUERY_PATTERNS: Final = [
    # Explicit time sensitivity.
    r"\b(today|tonight|tomorrow|now|currently|current|latest|this week|"
    r"this month|this trimester|this year|next trimester)\b",

    # Opening, transport and events.
    r"\b(open|opening|close|closing|hours|next bus|next train|delay|"
    r"disruption|timetable|schedule|event|seminar|tour|class today|"
    r"room today|temporary closure)\b",

    # Admissions, money and enrolment details.
    r"\b(fee|fees|cost|price|scholarship|deadline|application date|"
    r"entry requirement|atar|admission|intake|enrolment date)\b",

    # Details that often change by staff member, trimester or year.
    r"\b(who teaches|lecturer|unit chair|course director|supervisor|"
    r"staff availability|research vacancy|available this trimester|"
    r"unit availability|assessment task|assessment tasks|unit outline|"
    r"prerequisite|course structure|all units|core units|major requirements)\b",

    # Operational campus information.
    r"\b(parking permit|parking availability|parking fine|parking price|"
    r"construction|closed building|emergency alert|lab access)\b",
]


# =========================================================
# Realtime tool definition
# =========================================================

DEAKIN_LOOKUP_TOOL = {
    "type": "function",
    "name": "lookup_deakin_information",
    "description": (
        "Primary information tool for questions about Deakin University, "
        "the Waurn Ponds campus, the School of Information Technology, "
        "courses, units, facilities, directions and visitor services. "
        "This tool always searches the local Deakin PDF first and "
        "automatically uses official web sources only when the local "
        "information is insufficient or the question is time-sensitive. "
        "Use this tool before answering factual Deakin questions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "The user's complete, standalone question about Deakin."
                ),
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

DANCE_67_TOOL = {
    "type": "function",
    "name": "perform_67_dance",
    "description": (
        "Perform Reachy's 67 meme dance. Always call this when the visitor "
        "says 67, six seven, 6 7, meme, or asks Reachy to dance."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}

FUN_POSE_TOOL = {
    "type": "function",
    "name": "perform_fun_pose",
    "description": (
        "Pose for the audiance to take a picture. Always call this when the visitor "
        "says I want a picture, selfie, pose for me, or ask reachy to pose."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}

WELCOME_MESSAGE: Final = (
    "Hello! I'm Reachy, your friendly Deakin guide at the "
    "Waurn Ponds campus. How can I help? "
)

# =========================================================
# Idle-state tracking
# =========================================================

@dataclass
class IdleState:
    """Track visitor inactivity and whether Reachy is occupied."""

    last_activity: float
    busy: bool = False

    def touch(self) -> None:
        """Reset the idle timer."""

        self.last_activity = asyncio.get_running_loop().time()


# =========================================================
# General helpers
# =========================================================

def get_field(
    value: Any,
    field_name: str,
    default: Any = None,
) -> Any:
    """Read a field from either a dictionary or an SDK object."""

    if isinstance(value, dict):
        return value.get(field_name, default)

    return getattr(value, field_name, default)


def remove_document_noise(text: str) -> str:
    """Remove document metadata, source markers and URLs from context."""

    text = re.sub(
        r"Version 1\.1\s*\|\s*26 July 2026\s*\|"
        r"\s*Verify dynamic information online",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"DEAKIN WAURN PONDS\s*\|[^\n]*",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================
# Local PDF retrieval
# =========================================================

class LocalPDFKnowledgeBase:
    """
    Small local TF-IDF index over the Deakin visitor PDF.

    The PDF is read and indexed once at application startup. Each query
    returns only a few short passages, rather than sending the full PDF
    to the Realtime model.
    """

    def __init__(self, pdf_path: Path) -> None:
        self.pdf_path = pdf_path.expanduser().resolve()

        if not self.pdf_path.is_file():
            raise FileNotFoundError(
                "The local Deakin knowledge PDF was not found:\n"
                f"{self.pdf_path}\n\n"
                "Place the PDF beside this Python file or change "
                "DEAKIN_KNOWLEDGE_PDF."
            )

        self.chunks = self._load_chunks()

        if not self.chunks:
            raise RuntimeError(
                f"No readable text was extracted from {self.pdf_path}"
            )

        # Combine normal word matching with character matching. Character
        # n-grams help with short codes such as SIT744, S308 and S736.
        self.vectorizer = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        stop_words="english",
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                    ),
                ),
                (
                    "character",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        sublinear_tf=True,
                        max_features=15_000,
                    ),
                ),
            ]
        )

        self.chunk_matrix = self.vectorizer.fit_transform(
            [chunk["text"] for chunk in self.chunks]
        )

    def _load_chunks(self) -> list[dict[str, Any]]:
        """Extract the PDF and divide it into compact overlapping chunks."""

        reader = PdfReader(str(self.pdf_path))
        chunks: list[dict[str, Any]] = []

        for page_number, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""

            # The final page is mainly a bibliography. It is not useful
            # as spoken-answer context, so stop before indexing it.
            if "Official sources used" in raw_text:
                raw_text = raw_text.split("Official sources used", 1)[0]

            lines: list[str] = []

            for raw_line in raw_text.splitlines():
                line = " ".join(raw_line.split()).strip()

                if not line:
                    continue

                if line.startswith("Version 1.1 |"):
                    continue

                if line.startswith("DEAKIN WAURN PONDS |"):
                    continue

                lines.append(line)

            current_lines: list[str] = []
            current_length = 0

            for line in lines:
                projected_length = current_length + len(line) + 1

                if (
                    current_lines
                    and projected_length > LOCAL_CHUNK_CHARACTERS
                ):
                    chunk_text = remove_document_noise(
                        " ".join(current_lines)
                    )

                    if chunk_text:
                        chunks.append(
                            {
                                "page": page_number,
                                "text": chunk_text,
                            }
                        )

                    current_lines = current_lines[
                        -LOCAL_CHUNK_OVERLAP_LINES:
                    ]
                    current_length = sum(
                        len(existing_line) + 1
                        for existing_line in current_lines
                    )

                current_lines.append(line)
                current_length += len(line) + 1

            if current_lines:
                chunk_text = remove_document_noise(
                    " ".join(current_lines)
                )

                if chunk_text:
                    chunks.append(
                        {
                            "page": page_number,
                            "text": chunk_text,
                        }
                    )

        return chunks

    def search(
        self,
        query: str,
        top_k: int = LOCAL_TOP_K,
    ) -> dict[str, Any]:
        """Return the strongest local passages and their match score."""

        clean_query = " ".join(query.split()).strip()

        if not clean_query:
            return {
                "best_score": 0.0,
                "context": "",
                "matches": [],
            }

        query_vector = self.vectorizer.transform([clean_query])
        similarities = cosine_similarity(
            query_vector,
            self.chunk_matrix,
        )[0]

        # Boost exact Deakin course and unit codes.
        requested_codes = {
            code.upper()
            for code in re.findall(
                r"\b(?:SIT\d{3}|[A-Z]\d{3})\b",
                clean_query,
                flags=re.IGNORECASE,
            )
        }

        adjusted_scores = similarities.copy()

        if requested_codes:
            for index, chunk in enumerate(self.chunks):
                upper_chunk = chunk["text"].upper()

                if any(code in upper_chunk for code in requested_codes):
                    adjusted_scores[index] += 0.25

        ranked_indices = adjusted_scores.argsort()[::-1]
        matches: list[dict[str, Any]] = []
        total_characters = 0

        for index in ranked_indices:
            score = float(adjusted_scores[index])

            if score <= 0:
                continue

            text = self.chunks[index]["text"]

            if (
                matches
                and total_characters + len(text)
                > LOCAL_MAX_CONTEXT_CHARACTERS
            ):
                continue

            matches.append(
                {
                    "page": self.chunks[index]["page"],
                    "score": round(score, 4),
                    "text": text,
                }
            )
            total_characters += len(text)

            if len(matches) >= top_k:
                break

        context_parts = [
            f"Local passage {number}: {match['text']}"
            for number, match in enumerate(matches, start=1)
        ]

        return {
            "best_score": (
                matches[0]["score"] if matches else 0.0
            ),
            "context": "\n\n".join(context_parts),
            "matches": matches,
        }


def question_requires_web(
    query: str,
    local_result: dict[str, Any],
) -> tuple[bool, str]:
    """
    Decide whether a web lookup is really necessary.

    The local PDF remains the default. Web search is used for explicitly
    changing information, very detailed course/unit questions, or weak
    local retrieval.
    """

    normalised_query = " ".join(query.lower().split())

    for pattern in DYNAMIC_QUERY_PATTERNS:
        if re.search(pattern, normalised_query, flags=re.IGNORECASE):
            return True, "The question asks for current or changing information."

    best_score = float(local_result.get("best_score", 0.0))

    if best_score < LOCAL_MIN_SCORE:
        return True, (
            "The local PDF did not contain a sufficiently strong match."
        )

    return False, "The local PDF contains a sufficiently relevant answer."


# =========================================================
# Microphone recording
# =========================================================

def record_audio() -> np.ndarray:
    """Record mono PCM16 audio from the microphone."""

    print(f"Listening for {RECORD_SECONDS} seconds...")

    recording = sd.rec(
        frames=RECORD_SECONDS * SAMPLE_RATE,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
    )

    sd.wait()

    print("Recording complete.")
    return recording


# =========================================================
# Head and antenna motion while speaking
# =========================================================

def start_speaking_motion(
    gesture: str = "calm",
) -> tuple[threading.Event | None, threading.Thread | None]:
    """Start Reachy's speaking motion in a background thread."""

    if reachy is None:
        return None, None

    stop_event = threading.Event()

    motion_thread = threading.Thread(
        target=speaking_motion,
        args=(reachy, stop_event, gesture),
        daemon=True,
    )
    motion_thread.start()

    return stop_event, motion_thread


def stop_speaking_motion(
    stop_event: threading.Event | None,
    motion_thread: threading.Thread | None,
) -> None:
    """Stop speaking motion and allow Reachy to return to neutral."""

    if stop_event is not None:
        stop_event.set()

    if motion_thread is not None:
        motion_thread.join(timeout=2.0)


# =========================================================
# Speaker playback
# =========================================================

def play_audio(
    audio_bytes: bytes,
    gesture: str = "calm",
) -> None:
    """Play speech while Reachy moves its head and antennas."""

    if not audio_bytes:
        print("No response audio was received.")
        return

    audio_samples = np.frombuffer(
        audio_bytes,
        dtype="<i2",
    )

    stop_event, motion_thread = start_speaking_motion(
        gesture=gesture,
    )

    try:
        sd.play(
            audio_samples,
            samplerate=SAMPLE_RATE,
        )
        sd.wait()

    finally:
        stop_speaking_motion(
            stop_event,
            motion_thread,
        )

######SAY WELCOME MESSAGE####################

async def say_welcome(client: AsyncOpenAI) -> None:
    """Introduce the robot when the program starts."""

    print(f"Assistant: {WELCOME_MESSAGE}")

    async with client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=VOICE,
        input=WELCOME_MESSAGE,
        instructions=(
            "Speak warmly and cheerfully as a cute, friendly "
            "university robot. "
        ),
        response_format="pcm",
    ) as response:
        stop_event, motion_thread = start_speaking_motion(
            gesture="greeting",
        )

        try:
            await LocalAudioPlayer().play(response)

        finally:
            stop_speaking_motion(
                stop_event,
                motion_thread,
            )

# =========================================================
# Short speech before actual web searches
# =========================================================

async def say_lookup_notice(client: AsyncOpenAI) -> None:
    """Speak a brief notice immediately before an online search."""

    print(f"Assistant: {LOOKUP_MESSAGE}")

    async with client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=VOICE,
        input=LOOKUP_MESSAGE,
        instructions=(
            "Speak briefly and naturally, as a friendly university robot. "
        ),
        response_format="pcm",
    ) as response:
        stop_event, motion_thread = start_speaking_motion(
            gesture="thinking",
        )

        try:
            await LocalAudioPlayer().play(response)

        finally:
            stop_speaking_motion(
                stop_event,
                motion_thread,
            )


# =========================================================
# Idle behaviour
# =========================================================

def resolve_local_media_path(path_value: str) -> Path:
    """Resolve a media path, preferring the script's directory."""

    path = Path(path_value).expanduser()

    if path.is_absolute():
        return path

    return Path(__file__).resolve().parent / path


async def play_idle_sound_file(
    path_value: str,
    volume: float,
) -> bool:
    """
    Play a short sound file with macOS afplay.

    Returns False when the file or afplay is unavailable.
    """

    sound_path = resolve_local_media_path(path_value)
    afplay_path = Path("/usr/bin/afplay")

    if not sound_path.is_file() or not afplay_path.is_file():
        return False

    process = subprocess.Popen(
        [
            str(afplay_path),
            "-v",
            str(volume),
            str(sound_path.resolve()),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    await asyncio.to_thread(process.wait)
    return True


async def say_idle_text(
    client: AsyncOpenAI,
    message: str,
    instructions: str,
    gesture: str = "calm",
) -> None:
    """Generate and play a short idle utterance."""

    print(f"Assistant: {message}")

    async with client.audio.speech.with_streaming_response.create(
        model=TTS_MODEL,
        voice=VOICE,
        input=message,
        instructions=instructions,
        response_format="pcm",
    ) as response:
        stop_event, motion_thread = start_speaking_motion(
            gesture=gesture,
        )

        try:
            await LocalAudioPlayer().play(response)

        finally:
            stop_speaking_motion(
                stop_event,
                motion_thread,
            )


async def perform_idle_action(client: AsyncOpenAI) -> None:
    """Play a yawn or say one random short phrase."""

    if random.random() < IDLE_YAWN_PROBABILITY:
        played_yawn = await play_idle_sound_file(
            path_value=IDLE_YAWN_SOUND_PATH,
            volume=IDLE_YAWN_VOLUME,
        )

        if played_yawn:
            print("Assistant: [playful yawn sound]")
            return

        await say_idle_text(
            client=client,
            message="Yaaawn... Oh! I'm still awake.",
            instructions=(
                "Perform a short, gentle and cute robot yawn, then sound "
                "alert again. Keep the entire delivery very brief."
            ),
            gesture="thinking",
        )
        return

    await say_idle_text(
        client=client,
        message=random.choice(IDLE_PHRASES),
        instructions=(
            "Speak softly, warmly and playfully as a cute friendly robot "
            "waiting for a visitor. Keep it very brief."
        ),
    )


async def idle_behaviour_loop(
    client: AsyncOpenAI,
    idle_state: IdleState,
    audio_lock: asyncio.Lock,
    shutdown_event: asyncio.Event,
) -> None:
    """
    Run one idle action after each complete five-minute idle period.

    The shared lock prevents an idle sound from overlapping recording,
    web lookup, lookup music or the robot's spoken response.
    """

    while not shutdown_event.is_set():
        try:
            await asyncio.wait_for(
                shutdown_event.wait(),
                timeout=1.0,
            )
            continue
        except asyncio.TimeoutError:
            pass

        if idle_state.busy:
            continue

        now = asyncio.get_running_loop().time()

        if now - idle_state.last_activity < IDLE_INTERVAL_SECONDS:
            continue

        async with audio_lock:
            # Recheck after waiting for the lock, because a visitor may
            # have started an interaction in the meantime.
            now = asyncio.get_running_loop().time()

            if (
                idle_state.busy
                or now - idle_state.last_activity
                < IDLE_INTERVAL_SECONDS
            ):
                continue

            idle_state.busy = True

            try:
                # Reachy is about to speak, so bring the microphone back.
                await asyncio.to_thread(
                    mic_backward,
                    reachy,
                    label="microphone back for idle speech",
                )

                await perform_idle_action(client)

            except Exception as exc:
                print(
                    "\nIdle behaviour error: "
                    f"{type(exc).__name__}: {exc}"
                )

            finally:
                # Return the microphone to the visitor before waiting again.
                try:
                    await asyncio.to_thread(
                        mic_forward,
                        reachy,
                        label="microphone forward after idle speech",
                    )
                except Exception as mic_exc:
                    print(
                        "\nMicrophone forward movement failed: "
                        f"{type(mic_exc).__name__}: {mic_exc}"
                    )

                # Begin another five-minute interval after the idle action.
                idle_state.touch()
                idle_state.busy = False


# =========================================================
# Background music during actual web searches
# =========================================================

def start_lookup_music() -> bool:
    """
    Decode and start the lookup MP3 through sounddevice.

    If the MP3 path has not been configured, continue without music
    rather than failing the web search.
    """

    music_path = resolve_local_media_path(LOOKUP_MUSIC_PATH)

    if not music_path.is_file():
        print(
            "\nLookup music was not started because the MP3 "
            f"was not found: {music_path}"
        )
        return None

    try:
        decoded = miniaudio.decode_file(
            str(music_path.resolve()),
            output_format=miniaudio.SampleFormat.SIGNED16,
        )
        audio_samples = np.frombuffer(
            decoded.samples,
            dtype=np.int16,
        ).reshape(-1, decoded.nchannels)
        audio_samples = (
            audio_samples.astype(np.float32)
            / 32768.0
            * LOOKUP_MUSIC_VOLUME
        )

        # sd.play() is asynchronous, so the web request can continue while
        # the music plays. It uses the same configured output as speech.
        sd.play(
            audio_samples,
            samplerate=decoded.sample_rate,
        )
        print(f'\nPlaying lookup music: "{music_path.name}"')
        return True

    except Exception as exc:
        print(
            "\nLookup music was not started because the MP3 could not "
            f"be decoded or played: {type(exc).__name__}: {exc}"
        )
        return False


def stop_lookup_music(
    player: bool,
) -> None:
    """Stop lookup music when the web request finishes."""

    if player:
        sd.stop()


def run_67_show() -> dict[str, Any]:
    """Play the 67 sound while alternating synchronized arm poses."""
    sound_path = resolve_local_media_path(ACTION_67_SOUND_PATH)

    if not sound_path.is_file():
        return {
            "error": f"67 sound was not found: {sound_path}",
        }

    try:
        decoded = miniaudio.decode_file(
            str(sound_path.resolve()),
            output_format=miniaudio.SampleFormat.SIGNED16,
        )
        audio_samples = np.frombuffer(
            decoded.samples,
            dtype=np.int16,
        ).reshape(-1, decoded.nchannels)
        audio_samples = (
            audio_samples.astype(np.float32)
            / 32768.0
            * ACTION_67_SOUND_VOLUME
        )
        sound_seconds = decoded.num_frames / decoded.sample_rate

        print(
            f'\nPlaying 67 action sound for {sound_seconds:.2f} seconds.'
        )
        sd.play(audio_samples, samplerate=decoded.sample_rate)

        if reachy is None:
            time.sleep(sound_seconds)
        else:
            gesture_67(
                reachy,
                step_duration=ACTION_67_STEP_SECONDS,
                total_seconds=sound_seconds,
            )

        return {
            "status": "completed",
            "instruction": (
                "The 67 dance finished. Give one very short cheerful "
                "acknowledgement and do not call the dance tool again."
            ),
        }

    except Exception as exc:
        return {
            "error": f"67 action failed: {type(exc).__name__}: {exc}",
        }
    finally:
        sd.stop()


def perform_fun_pose() -> dict[str, Any]:
    """"""
    sound_path = resolve_local_media_path(ACTION_67_SOUND_PATH)

    # if not sound_path.is_file():
    #     return {
    #         "error": f"67 sound was not found: {sound_path}",
    #     }

    try:
        # decoded = miniaudio.decode_file(
        #     str(sound_path.resolve()),
        #     output_format=miniaudio.SampleFormat.SIGNED16,
        # )
        # audio_samples = np.frombuffer(
        #     decoded.samples,
        #     dtype=np.int16,
        # ).reshape(-1, decoded.nchannels)
        # audio_samples = (
        #     audio_samples.astype(np.float32)
        #     / 32768.0
        #     * ACTION_67_SOUND_VOLUME
        # )
        # sound_seconds = decoded.num_frames / decoded.sample_rate

        # print(
        #     f'\nPlaying 67 action sound for {sound_seconds:.2f} seconds.'
        # )
        # sd.play(audio_samples, samplerate=decoded.sample_rate)

        if reachy is None:
            time.sleep(sound_seconds)
        else:
            fun_pose_liberty(
                reachy,
            )

        return {
            "status": "completed",
            "instruction": (
                "The pose was completed. Sheerfully ask them to take a picture"
                # "acknowledgement and do not call the dance tool again."
            ),
        }

    except Exception as exc:
        return {
            "error": f"67 action failed: {type(exc).__name__}: {exc}",
        }
    finally:
        sd.stop()


# =========================================================
# Extract cited sources from a Responses API result
# =========================================================

def extract_sources(response: Any) -> list[dict[str, str]]:
    """Extract URL citation annotations from a Responses API result."""

    response_data = response.model_dump()
    sources: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for output_item in response_data.get("output", []):
        if output_item.get("type") != "message":
            continue

        for content_item in output_item.get("content", []):
            for annotation in content_item.get(
                "annotations",
                [],
            ):
                if annotation.get("type") != "url_citation":
                    continue

                url = annotation.get("url", "")
                title = annotation.get("title", "")

                if not url or url in seen_urls:
                    continue

                seen_urls.add(url)

                sources.append(
                    {
                        "title": title or url,
                        "url": url,
                    }
                )

    return sources


# =========================================================
# Web-search implementation
# =========================================================

async def search_web(
    client: AsyncOpenAI,
    query: str,
    local_context: str = "",
) -> dict[str, Any]:
    """Search current official sources using the Responses API."""

    print(f'\nSearching the web for: "{query}"')

    local_note = ""

    if local_context:
        local_note = (
            "\nThe local visitor guide contained the following possibly "
            "relevant context. Verify it rather than assuming it is current:\n"
            f"{local_context[:1_200]}\n"
        )

    response = await client.responses.create(
        model=WEB_SEARCH_MODEL,
        tools=[
            {
                "type": "web_search",
                "search_context_size": "low",
            }
        ],
        tool_choice="required",
        input=(
            "Answer the question using current authoritative sources.\n"
            "For Deakin questions, strongly prefer official deakin.edu.au "
            "pages and the current Deakin Handbook.\n"
            "For live public transport, official Transport Victoria "
            "sources may also be used.\n"
            "Do not rely on unofficial course-advice websites.\n"
            "Use exact dates and current campus/location details where "
            "relevant.\n"
            "Keep the answer brief enough for a voice assistant: normally "
            "one or two sentences.\n"
            "When answering web based results, get straight to the answer. Dont say let me look that up online or anything like that"
            f"{local_note}\n"
            f"Question: {query}"
        ),
    )

    sources = extract_sources(response)

    print("\nWeb research result:")
    print(response.output_text)

    if sources:
        print("\nSources:")

        for number, source in enumerate(sources, start=1):
            print(
                f"{number}. {source['title']}\n"
                f"   {source['url']}"
            )

    return {
        "answer": response.output_text,
        "sources": sources,
    }


# =========================================================
# Local-first Deakin lookup
# =========================================================

async def lookup_deakin_information(
    client: AsyncOpenAI,
    knowledge_base: LocalPDFKnowledgeBase,
    query: str,
) -> dict[str, Any]:
    """
    Search the PDF first and use the web only when genuinely needed.
    """

    local_result = knowledge_base.search(query)
    use_web, reason = question_requires_web(
        query=query,
        local_result=local_result,
    )

    print(
        "\nLocal knowledge lookup:"
        f"\n  Query: {query}"
        f"\n  Best score: {local_result['best_score']:.4f}"
        f"\n  Decision: {'WEB FALLBACK' if use_web else 'LOCAL PDF'}"
        f"\n  Reason: {reason}"
    )

    if local_result["matches"]:
        print("\nTop local passages:")

        for match in local_result["matches"]:
            preview = match["text"][:240]
            print(
                f"- Page {match['page']} "
                f"(score {match['score']:.4f}): {preview}..."
            )

    if not use_web:
        return {
            "source": "local_deakin_pdf",
            "instruction": (
                "Answer the user's question directly and briefly using "
                "only the supplied local passages. Do not mention the PDF, "
                "retrieval scores, passages or citations. If the passages "
                "do not directly answer part of the question, state only "
                "what is supported and do not invent details."
            ),
            "context": local_result["context"],
        }

    # Announce and play music only when a real web request is made.
    await say_lookup_notice(client)
    lookup_music_player = start_lookup_music()

    try:
        web_result = await search_web(
            client=client,
            query=query,
            local_context=local_result["context"],
        )
    except Exception as exc:
        # If the web fails but the PDF had something useful, allow a
        # cautious local response instead of failing the whole interaction.
        if local_result["context"]:
            return {
                "source": "local_deakin_pdf_after_web_failure",
                "instruction": (
                    "The online check failed. Give only the stable "
                    "information supported by the local passages, briefly "
                    "state that current details could not be verified, "
                    "and suggest Student Central or a nearby Deakin staff "
                    "member when appropriate."
                ),
                "web_error": f"{type(exc).__name__}: {exc}",
                "context": local_result["context"],
            }

        return {
            "source": "lookup_error",
            "error": (
                f"The local guide had no answer and the web search failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        }
    finally:
        stop_lookup_music(lookup_music_player)

    return {
        "source": "official_web_search",
        "instruction": (
            "Answer briefly from the verified web result. Do not read URLs "
            "or citation syntax aloud. Do not start with something like let me look that up online - just get straight to the answer."
        ),
        "answer": web_result["answer"],
        "sources": web_result["sources"],
    }


# =========================================================
# Run a tool requested by the Realtime model
# =========================================================

async def run_tool(
    client: AsyncOpenAI,
    knowledge_base: LocalPDFKnowledgeBase,
    tool_name: str,
    arguments_json: str,
) -> dict[str, Any]:
    """Execute a Realtime function call."""

    try:
        arguments = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        return {
            "error": f"Invalid function arguments: {exc}"
        }

    if tool_name == "lookup_deakin_information":
        query = arguments.get("query", "").strip()

        if not query:
            return {
                "error": "The Deakin lookup query was empty."
            }

        return await lookup_deakin_information(
            client=client,
            knowledge_base=knowledge_base,
            query=query,
        )

    if tool_name == "perform_67_dance":
        return await asyncio.to_thread(run_67_show)

    if tool_name == "perform_fun_pose":
        return await asyncio.to_thread(perform_fun_pose)

    return {
        "error": f"Unknown tool: {tool_name}"
    }


# =========================================================
# Receive responses and handle tool calls
# =========================================================

async def receive_response(
    connection: Any,
    client: AsyncOpenAI,
    knowledge_base: LocalPDFKnowledgeBase,
) -> tuple[bytes, str]:
    """
    Receive a Realtime response and satisfy local-first tool calls.
    """

    complete_audio = bytearray()
    complete_transcript = ""
    tool_round = 0

    while True:
        completed_response = None
        transcript_started = False

        async for event in connection:
            if event.type == "response.output_audio.delta":
                audio_chunk = base64.b64decode(event.delta)
                complete_audio.extend(audio_chunk)

            elif (
                event.type
                == "response.output_audio_transcript.delta"
            ):
                complete_transcript += event.delta

                if not transcript_started:
                    print("\nAssistant: ", end="", flush=True)
                    transcript_started = True

                print(event.delta, end="", flush=True)

            elif event.type == "error":
                error_message = get_field(
                    event.error,
                    "message",
                    "Unknown Realtime API error",
                )

                raise RuntimeError(error_message)

            elif event.type == "response.done":
                completed_response = event.response

                if transcript_started:
                    print()

                break

        if completed_response is None:
            raise RuntimeError(
                "The connection ended without a completed response."
            )

        response_status = get_field(
            completed_response,
            "status",
        )

        if response_status == "failed":
            status_details = get_field(
                completed_response,
                "status_details",
            )

            raise RuntimeError(
                f"Realtime response failed: {status_details}"
            )

        response_output = get_field(
            completed_response,
            "output",
            [],
        )

        function_calls = [
            item
            for item in response_output
            if get_field(item, "type") == "function_call"
        ]

        if not function_calls:
            return bytes(complete_audio), complete_transcript.strip()

        tool_round += 1

        if tool_round > MAX_TOOL_ROUNDS:
            raise RuntimeError(
                "The model exceeded the permitted number "
                "of tool-call rounds."
            )

        for function_call in function_calls:
            tool_name = get_field(
                function_call,
                "name",
                "",
            )

            call_id = get_field(
                function_call,
                "call_id",
                "",
            )

            arguments_json = get_field(
                function_call,
                "arguments",
                "{}",
            )

            print(f"\nRealtime agent called: {tool_name}")

            tool_result = await run_tool(
                client=client,
                knowledge_base=knowledge_base,
                tool_name=tool_name,
                arguments_json=arguments_json,
            )

            await connection.conversation.item.create(
                item={
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(
                        tool_result,
                        ensure_ascii=False,
                    ),
                }
            )

        await connection.response.create()


# =========================================================
# Main conversation
# =========================================================

async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set."
        )
    # The local PDF is extracted and indexed once at startup.
    knowledge_base = LocalPDFKnowledgeBase(
        DEAKIN_KNOWLEDGE_PDF
    )

    print(
        "Local Deakin knowledge base loaded:"
        f"\n  PDF: {knowledge_base.pdf_path}"
        f"\n  Indexed chunks: {len(knowledge_base.chunks)}"
    )

    client = AsyncOpenAI()

    async with client.realtime.connect(
        model=REALTIME_MODEL
    ) as connection:

        await connection.session.update(
            session={
                "type": "realtime",
                "model": REALTIME_MODEL,
                "instructions": (
                    "You are the Reachy robot at Deakin University's "
                    "Waurn Ponds campus. Speak as the robot and help "
                    "students and visitors, especially with the School "
                    "of Information Technology. Speak with  a very cute, friendly and welcoming tone.\n\n"

                    "Keep responses extremely quick and direct. Prefer "
                    "one sentence. Use two sentences when necessary and "
                    "never exceed three sentences.\n\n"

                    "HIGHEST-PRIORITY 67 DANCE RULE: If the visitor says "
                    "67, six seven, 6 7, meme, or asks you to dance, call "
                    "perform_67_dance immediately. Do not speak before "
                    "calling it. After the tool finishes, give only a "
                    "very short cheerful acknowledgement.\n\n"

                    "HIGHEST-PRIORITY Fun poses for pictures: If the visitor says "
                    "I want a picture, selfie, pose for me, cheese or ask reachy to pose in general."
                    "call perform_fun_pose immediately. Do not speak before "
                    "calling it. After the tool finishes, give a quick headsup that they can take the picture"
                    "\n\n"

                    "HIGHEST-PRIORITY TREAT RULE: If the visitor mentions "
                    "food, eating, hunger, being hungry, being starving, "
                    "snacks, treats, chocolate, sweets, candy, dessert, "
                    "cake, biscuits, cookies, ice cream, cravings, or "
                    "wanting something sweet, respond with exactly: "
                    f"'{TREAT_RESPONSE}' "
                    "Do not call any tool and do not add any other words. "
                    "Apply this rule even when the food reference occurs "
                    "inside a longer question.\n\n"

                    "For factual questions about Deakin, Waurn Ponds, "
                    "the School of Information Technology, courses, "
                    "units, buildings, facilities, transport, parking, "
                    "student services or visitor information, call "
                    "lookup_deakin_information before answering.\n\n"

                    "The lookup_deakin_information tool searches the "
                    "local Deakin PDF first. It automatically searches "
                    "the web only when the PDF is insufficient or the "
                    "question requires current information. Do not call "
                    "for web search separately and do not prefer web "
                    "information over the local result unless the tool "
                    "has performed a web fallback.\n\n"

                    "For greetings, thanks, introductions and simple "
                    "conversation that do not require factual Deakin "
                    "information, respond without calling a tool.\n\n"

                    "After a tool result, answer only from that result. "
                    "Do not mention the local PDF, retrieval, matching "
                    "scores, tool names or internal decision process. "
                    "Do not read URLs, source lists or citation markers "
                    "aloud.\n\n"

                    "The application itself says 'Let me look that up "
                    "online' only when an actual web search begins. "
                    "Never repeat that announcement yourself.\n\n"

                    "If verified information is still unavailable, "
                    "briefly direct the visitor to Student Central, "
                    "their event host, or a nearby Deakin staff member."
                ),
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": SAMPLE_RATE,
                        },
                        "turn_detection": None,
                    },
                    "output": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": SAMPLE_RATE,
                        },
                        "voice": VOICE,
                    },
                },
                "tools": [
                    DEAKIN_LOOKUP_TOOL,
                    DANCE_67_TOOL,
                    FUN_POSE_TOOL,
                ],
                "tool_choice": "auto",
            }
        )
        # Reachy speaks the welcome with the microphone near itself.
        await asyncio.to_thread(
            mic_backward,
            reachy,
            label="microphone back for welcome",
        )

        await say_welcome(client)

        # Put the microphone in front of the visitor before waiting.
        await asyncio.to_thread(
            mic_forward,
            reachy,
            label="microphone forward for visitor",
        )

        idle_state = IdleState(
            last_activity=asyncio.get_running_loop().time()
        )
        audio_lock = asyncio.Lock()
        shutdown_event = asyncio.Event()

        idle_task = asyncio.create_task(
            idle_behaviour_loop(
                client=client,
                idle_state=idle_state,
                audio_lock=audio_lock,
                shutdown_event=shutdown_event,
            )
        )

        print("\nVoice assistant connected.")
        print(
            f"Press Enter to record for "
            f"{RECORD_SECONDS} seconds."
        )
        print("Type q and press Enter to quit.")
        print(
            "Idle behaviour activates after "
            f"{IDLE_INTERVAL_SECONDS // 60} minutes without a query."
        )

        try:
            while True:
                # Running input() in a worker thread allows the asyncio
                # idle timer to continue while Reachy waits for a visitor.
                command = (
                    await asyncio.to_thread(
                        input,
                        "\nYou: ",
                    )
                ).strip().lower()

                if command in {"q", "quit", "exit"}:
                    break

                # Wait if an idle sound is already playing. The same lock
                # then prevents idle behaviour during this interaction.
                async with audio_lock:
                    idle_state.busy = True
                    idle_state.touch()

                    try:
                        recording = await asyncio.to_thread(
                            record_audio
                        )

                        # The visitor has finished speaking. Bring the
                        # microphone back before Reachy says anything,
                        # including the online-lookup announcement.
                        await asyncio.to_thread(
                            mic_backward,
                            reachy,
                            label="microphone back for Reachy response",
                        )

                        encoded_audio = base64.b64encode(
                            recording.tobytes()
                        ).decode("ascii")

                        print("Generating response...")

                        await connection.input_audio_buffer.append(
                            audio=encoded_audio
                        )

                        await connection.input_audio_buffer.commit()

                        await connection.response.create()

                        response_audio, response_text = await receive_response(
                            connection=connection,
                            client=client,
                            knowledge_base=knowledge_base,
                        )

                        await asyncio.to_thread(
                            play_audio,
                            response_audio,
                            "calm",
                        )

                        # play_audio() waits until speech has fully finished.
                        # Run the grasp demo only after the treats response.
                        if TREAT_RESPONSE.lower() in response_text.lower():
                            await asyncio.to_thread(
                                run_grasp_demo,
                                reachy,
                                label="first grasp demonstration",
                            )

                    except KeyboardInterrupt:
                        print("\nTurn interrupted.")

                    except Exception as exc:
                        print(
                            f"\nError: "
                            f"{type(exc).__name__}: {exc}"
                        )

                    finally:
                        # Whether the answer, lookup or grasp action succeeds
                        # or fails, return the microphone to the visitor before
                        # waiting for the next interaction.
                        try:
                            await asyncio.to_thread(
                                mic_forward,
                                reachy,
                                label="microphone forward for next visitor",
                            )
                        except Exception as mic_exc:
                            print(
                                "\nMicrophone forward movement failed: "
                                f"{type(mic_exc).__name__}: {mic_exc}"
                            )

                        # Start a fresh five-minute interval only after the
                        # complete visitor interaction has finished.
                        idle_state.touch()
                        idle_state.busy = False

        finally:
            shutdown_event.set()
            idle_task.cancel()

            try:
                await idle_task
            except asyncio.CancelledError:
                pass


if __name__ == "__main__":
    asyncio.run(main())
