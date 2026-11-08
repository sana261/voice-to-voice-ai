
import os
import gradio as gr
import soundfile as sf
import tempfile

from groq import Groq
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro


# 1. LOAD MODELS

print("Loading Whisper Model...")

stt_model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

print("Loading Kokoro TTS...")

tts = Kokoro(
    "kokoro-v1.0.onnx",
    "voices-v1.0.bin"
)

print("Connecting Groq...")

api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing!")

client = Groq(api_key=api_key)


# 2. PROCESS VOICE

def process_audio(audio_path):

    print("Received Audio:", repr(audio_path))

    if audio_path is None:
        return "Please record your voice first.", None

    try:

        # Whisper Speech Recognition

        print("Converting voice to text...")

        segments, info = stt_model.transcribe(
            audio_path,
            beam_size=5
        )

        user_text = " ".join(
            segment.text.strip()
            for segment in segments
        )

        if not user_text.strip():
            return "Voice not understood. Try again.", None

        print("User:", user_text)

    except Exception as e:

        return f"Speech Recognition Error: {str(e)}", None


    try:

        # GROQ AI RESPONSE

        print("Getting AI response...")

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": user_text
                }
            ],
            model="openai/gpt-oss-120b"
        )

        ai_text = response.choices[0].message.content

        print("AI:", ai_text)

    except Exception as e:

        return f"Groq Error: {str(e)}", None


    try:

        # KOKORO TEXT TO SPEECH

        print("Generating AI voice...")

        audio, sample_rate = tts.create(
            ai_text,
            voice="af_heart",
            speed=1.0,
            lang="en-us"
        )

        output_file = tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ).name

        sf.write(
            output_file,
            audio,
            sample_rate
        )

    except Exception as e:

        return f"TTS Error: {str(e)}", None


    # RETURN OUTPUT

    conversation = (
        f"You: {user_text}\n\n"
        f"AI: {ai_text}"
    )

    return conversation, output_file


# 3. GRADIO INTERFACE

with gr.Blocks() as demo:

    gr.Markdown(
        "# Voice-to-Voice AI Assistant"
    )

    input_audio = gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="Record Your Voice"
    )

    submit_btn = gr.Button("Submit")

    output_text = gr.Textbox(
        label="Conversation Transcript",
        lines=6
    )

    output_audio = gr.Audio(
        label="AI Voice Response",
        type="filepath"
    )

    submit_btn.click(
        fn=process_audio,
        inputs=input_audio,
        outputs=[output_text, output_audio]
    )


# 4. LAUNCH APP

if __name__ == "__main__":

    demo.queue()

    demo.launch(
        share=True,
        debug=True
    )
