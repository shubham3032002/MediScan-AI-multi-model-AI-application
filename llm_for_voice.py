import os
import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from voice_patient import transcribe_with_groq
from langchain_groq import ChatGroq
from voice_doctor import text_to_speech_with_elevenlabs

system_prompt = """You have to act as a professional doctor, I know you are not but this is for learning purposes. 
What's the patient's day? Do you find anything wrong with it medically? 
If you make a differential, suggest some remedies for them. Do not add any numbers or special characters in 
your response. Your response should be in one long paragraph. Also, always answer as if you are answering a real person.
Do not say 'In the image I see' but say 'With what I see, I think you have ....'
Do not respond as an AI model in markdown, your answer should mimic that of an actual doctor, not an AI bot. 
Keep your answer concise (max 2 sentences). No preamble, start your answer right away, please."""

# Streamlit UI
st.title("🩺 AI Doctor - Real-Time Voice & Vision")

# 🎤 **Record Live Voice**
st.write("🎙️ **Record Your Symptoms (Speak Now)**")
duration = st.slider("Select Recording Duration (seconds)", 3, 10, 5)
record_button = st.button("🔴 Start Recording")

audio_filepath = "recorded_audio.wav"

if record_button:
    st.text("🎤 Recording... Speak now!")
    
    fs = 44100  # Sample rate
    recorded_audio = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    
    write(audio_filepath, fs, recorded_audio)
    st.success("✅ Recording Complete!")

# 🔍 **Analyze Symptoms Button**
if st.button("Analyze Symptoms"):
    if not os.path.exists(audio_filepath):
        st.error("⚠ Please record your voice first!")
    else:
        # 📝 Convert Speech to Text
        st.text("⏳ Transcribing audio...")
        speech_to_text_output = transcribe_with_groq(
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY"), 
            audio_filepath=audio_filepath,
            stt_model="whisper-large-v3"
        )
        st.success("✅ Transcription Complete!")
        st.text_area("📝 Speech to Text", speech_to_text_output, height=100)

        # 🏥 **Generate AI Doctor's Diagnosis**
        llm = ChatGroq(model="llama-3.2-11b-vision-preview")
        doctor_response = llm.invoke(system_prompt + speech_to_text_output)
        
        # Extract only the response text
        if hasattr(doctor_response, "content"):
            doctor_response_text = doctor_response.content
        else:
            doctor_response_text = str(doctor_response)

        # 🏥 **Display AI Doctor's Diagnosis**
        st.success("🩺 AI Doctor's Diagnosis:")
        st.text_area("📋 Doctor's Response", doctor_response_text, height=100)

        # 🎙 **Convert Doctor's Response to Speech**
        st.text("🔊 Generating Doctor's Voice...")
        voice_output_path = "doctor_response.mp3"
        text_to_speech_with_elevenlabs(input_text=doctor_response_text, output_filepath=voice_output_path)

        st.success("✅ AI Doctor's Response is Ready!")

        # 🎵 **Auto-Play AI Doctor's Voice**
        audio_placeholder = st.empty()
        with open(voice_output_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            audio_placeholder.audio(audio_bytes, format="audio/mp3")
