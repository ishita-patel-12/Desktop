"""
Entry point.

  python assistant.py             -> text-mode CLI (type commands, no mic needed)
  python assistant.py --voice     -> continuous voice conversation (VAD-based, no press-to-talk)
  python assistant.py --gesture   -> gesture control with webcam
  python assistant.py --all       -> voice + gesture simultaneously
  python assistant.py --teach-gesture -> capture samples of a new hand pose,
                                          train it, and bind it to a labeled
                                          function (Phase 3, see gestures/teachable.py)
  python assistant.py --list-gestures -> print every taught gesture across
                                          every app, with its human-readable
                                          function label and bound action

Text mode exists so the Phase 1 pipeline (rules -> LLM fallback -> executor)
is demoable and testable without any audio setup at all - useful for
development, CI, and a quick "does the logic work" sanity check before
wiring up the microphone.
"""

from __future__ import annotations
import re
import sys
from pathlib import Path
from loguru import logger

from intent import rules, llm_fallback
from intent.schema import Intent, IntentType
from executor.dispatch import dispatch
from config.loader import known_app_names, load_settings

Path("logs").mkdir(exist_ok=True)
logger.remove()
logger.add("logs/assistant.log", rotation="1 MB", level="DEBUG")
logger.add(sys.stderr, level="INFO", format="<level>{message}</level>")


# ---------------------------------------------------------------------------
# Richer spoken responses: maps intent types to natural verbal templates
# ---------------------------------------------------------------------------
_SUCCESS_PHRASES: dict[IntentType, str] = {
    IntentType.CREATE_FILE:   "Done! I've created the file for you.",
    IntentType.CREATE_FOLDER: "Done! I've created the folder.",
    IntentType.DELETE_FILE:   "Done! The file has been moved to the trash.",
    IntentType.RENAME_FILE:   "Done! I've renamed that for you.",
    IntentType.COPY_FILE:     "Done! The file has been copied.",
    IntentType.OPEN_APP:      "Sure! Opening that for you now.",
    IntentType.CLOSE_APP:     "Okay, I've closed that.",
    IntentType.OPEN_WEBSITE:  "Sure! Opening that in your browser.",
    IntentType.WEB_SEARCH:    "On it! Searching the web now.",
    IntentType.UNDO:          "Done! Last action undone.",
}

_FAIL_PHRASES: dict[IntentType, str] = {
    IntentType.OPEN_APP:    "Sorry, I couldn't open that. It might not be installed, or I don't know about it yet.",
    IntentType.CLOSE_APP:   "Sorry, I couldn't find that running application to close.",
    IntentType.DELETE_FILE: "Sorry, I couldn't find that file to delete.",
    IntentType.RENAME_FILE: "Sorry, I had trouble renaming that. Make sure the file exists.",
    IntentType.COPY_FILE:   "Sorry, I couldn't copy that file.",
}


def _rich_response(intent: Intent, result) -> str:
    """Build a natural spoken response from the dispatch result."""
    if result.message:
        return result.message
    if result.success:
        return _SUCCESS_PHRASES.get(intent.type, "Done!")
    return _FAIL_PHRASES.get(intent.type, "Sorry, something went wrong with that.")


def handle_text(text: str, voice_confirm_fn=None) -> None:
    """Full pipeline: rules -> LLM fallback -> clarification -> dispatch.

    voice_confirm_fn: optional callable(prompt) -> bool for voice-mode
    confirmation (speaks the question, listens for yes/no). Falls back to
    terminal input when not provided.
    """
    apps_known = known_app_names()
    intent: Intent = rules.parse(text, known_apps=apps_known)

    if not intent.is_confident():
        logger.debug(f"Rule parser unconfident for {text!r}, trying LLM fallback...")
        llm_intent = llm_fallback.classify(text)
        if llm_intent.type.value != "unknown":
            intent = llm_intent
        elif intent.type.value == "unknown":
            _speak("Sorry, I didn't understand that. Could you try again with different words?")
            return
        elif intent.type.value != "open_app":
            _speak("I'm not quite sure what you meant. Could you rephrase that?")
            return

    if intent.needs_clarification:
        missing = " and ".join(intent.missing_slots)
        _speak(f"I'm missing some details — I need the {missing}. Could you say that again with those included?")
        return

    result = dispatch(intent)

    if result.needs_confirmation:
        prompt = result.confirmation_prompt or "Are you sure?"
        if voice_confirm_fn:
            confirmed = voice_confirm_fn(prompt)
        else:
            _speak(prompt)
            answer = input("(y/n) > ").strip().lower()
            confirmed = answer in ("y", "yes")

        if confirmed:
            result = dispatch(intent, confirmed=True)
            _speak(_rich_response(intent, result))
        else:
            _speak("Okay, I'll leave that as it is.")
        return

    _speak(_rich_response(intent, result))


def _speak(text: str) -> None:
    """Routes through TTS if available, else just prints."""
    try:
        from voice.tts import say
        say(text)
    except Exception:
        print(f"Assistant: {text}")


def run_text_mode() -> None:
    settings = load_settings()
    name = settings.get("assistant_name", "Assistant")
    print(f"{name} is listening (text mode). Type a command, or 'quit' to exit.\n")
    print("Try things like:")
    print("  create a python file called calculator")
    print("  create a folder called Projects")
    print("  open chrome")
    print("  rename Projects to MyProjects")
    print("  delete calculator.py")
    print("  close chrome\n")

    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in ("quit", "exit"):
            break
        if not text:
            continue
        handle_text(text)



# ---------------------------------------------------------------------------
# Session-exit detection for voice mode.
#
# BUG FIXED HERE: this used to be `any(w in text.lower() for w in (...))`,
# i.e. a *substring* check. "exit chrome" contains the substring "exit", so
# saying "exit chrome" (or "stop the music", "close it, exit please", etc.)
# ended the whole voice session instead of being handled as a command - the
# assistant would quit itself rather than closing the app you asked about.
# This now only matches when the ENTIRE (normalized) utterance is a
# stand-alone exit phrase, so "exit chrome" is left alone and falls through
# to the intent parser (-> CLOSE_APP), while a bare "exit" still ends the
# session.
# ---------------------------------------------------------------------------
_SESSION_EXIT_PHRASES = {
    "quit", "exit", "stop", "goodbye", "bye",
    "stop listening", "that's all", "thats all", "that will be all",
    "never mind", "nevermind", "cancel", "end session", "shut down",
}
_LEADING_ADDRESS = re.compile(r"^(?:hey\s+)?assistant[,]?\s*", re.I)


def _is_session_exit(text: str) -> bool:
    """True only if the whole utterance (after stripping punctuation and an
    optional leading address) IS one of the exit phrases - not merely
    contains one as a word inside a longer command."""
    normalized = re.sub(r"[^\w\s]", "", text.lower()).strip()
    normalized = _LEADING_ADDRESS.sub("", normalized).strip()
    return normalized in _SESSION_EXIT_PHRASES


def run_voice_mode(stop_event=None) -> None:
    """Continuous back-and-forth voice conversation using VAD (no press-to-talk).

    The assistant speaks a prompt, then continuously listens. As soon as you start
    speaking it records, stops when you go quiet, transcribes, executes, and
    responds — then goes back to listening. Say 'quit', 'exit', or 'stop'
    to end the session, or press Esc at any time to interrupt the assistant (stop it
    talking, or cut a listening turn short instead of waiting out the full
    timeout/silence window).

    `stop_event`: optional threading.Event; when set (e.g. by the gesture
    window being closed in `--all` mode), the loop exits cleanly at the
    next opportunity instead of being killed mid-sentence when the process
    tears down its daemon threads.
    """
    from voice.stt import listen_until_silence, preload_model
    from voice.hotkeys import start_interrupt_listener
    settings = load_settings()
    name = settings.get("assistant_name", "Assistant")

    print(f"\n--- {name} Voice Conversation Mode ---")
    print("Just speak naturally. Assistant listens automatically.")
    print("Say 'quit' or 'stop' to exit, or press Esc any time to interrupt.\n")

    start_interrupt_listener()
    preload_model()  # pay the one-time model-load cost now, not mid-sentence

    _speak(f"Hi! I'm your {name}. What would you like me to do?")

    def voice_confirm(prompt: str) -> bool:
        """Speak the confirmation question, listen for a yes/no answer."""
        _speak(prompt + " Say yes or no.")
        answer = listen_until_silence(
            silence_duration=0.8,
            pre_speech_timeout=5.0,
            max_duration=4.0,
        ).lower()
        logger.info(f"Confirmation heard: {answer!r}")
        return any(w in answer for w in ("yes", "yeah", "yep", "sure", "do it", "confirm"))

    while True:
        if stop_event is not None and stop_event.is_set():
            break

        print("[Listening... (Esc to interrupt)]")
        try:
            text = listen_until_silence(
                silence_threshold=0.015,
                silence_duration=1.2,
                pre_speech_timeout=10.0,
                max_duration=12.0,
            )
        except KeyboardInterrupt:
            print()
            break

        if stop_event is not None and stop_event.is_set():
            break

        if not text:
            # Nothing heard (or interrupted before speaking) - keep waiting
            continue

        text = text.strip()
        print(f"Heard: {text!r}")

        if _is_session_exit(text):
            _speak("Goodbye! Have a great day.")
            break

        handle_text(text, voice_confirm_fn=voice_confirm)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__.strip())
        sys.exit(0)

    has_voice = "--voice" in sys.argv or "--all" in sys.argv
    has_gesture = "--gesture" in sys.argv or "--all" in sys.argv
    has_teach = "--teach-gesture" in sys.argv
    has_list_gestures = "--list-gestures" in sys.argv

    if has_list_gestures:
        from gestures.teachable import print_all_taught_gestures
        print_all_taught_gestures()
    elif has_teach:
        from gestures.teachable import run_teach_mode
        run_teach_mode()
    elif has_voice and has_gesture:
        import threading
        from gestures.loop import run_gesture_loop
        from voice.stt import preload_model

        print("Starting both Voice Mode and Gesture Control simultaneously...\n")

        # Pay the one-time whisper model load cost up front, on the main
        # thread, before either loop starts. Loading it lazily on the first
        # utterance (the old behavior) briefly holds the GIL during model
        # deserialization, which stalls the gesture loop's frame processing
        # right as you start talking - this way that stall happens once,
        # predictably, before the camera window even opens.
        print("Loading speech-recognition model...")
        preload_model()

        # Lets closing the gesture window ('q') also tell the voice thread
        # to shut down cleanly (say goodbye, stop listening) instead of
        # being killed mid-sentence when the process exits and its daemon
        # threads get torn down.
        stop_event = threading.Event()
        t_voice = threading.Thread(target=run_voice_mode, args=(stop_event,), daemon=True)
        t_voice.start()
        try:
            run_gesture_loop(stop_event=stop_event)
        finally:
            stop_event.set()
    elif has_voice:
        run_voice_mode()
    elif has_gesture:
        from gestures.loop import run_gesture_loop
        run_gesture_loop()
    else:
        run_text_mode()