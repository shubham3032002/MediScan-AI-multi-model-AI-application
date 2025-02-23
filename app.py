import os
import time
import streamlit as st 
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from doctor_brain import encode_image, analyze_image_with_query
from voice_patient import transcribe_with_groq
from voice_doctor import text_to_speech_with_elevenlabs
from doctor_brain_for_uery_query import analyze_image_with_user_query

DB_FAISS_PATH = "vectorstore/db_faiss"

# Prompt Templates
RAG_PROMPT_TEMPLATE = """
Use the provided medical context to answer questions. Be professional and precise.
If unsure, say "I don't know". Never make up information.don't any special symbol and 1 in the the result

Context: {context}
Question: {question}

Answer in markdown format:"""

SYSTEM_PROMPT = """You have to act as a professional doctor. What's in this image? don't any special symbol and 1 in the the result
Do you find anything wrong medically? Suggest remedies. Keep response concise (2 sentences) i remainder you answer should be in one or two sentence.
Address the user directly. Example: 'With what I see, I think you have...'"""
VOICE_PROMPT = """ Act as a professional doctor. Analyze the patient’s condition concisely. If something seems wrong, suggest a remedy.
Answer naturally as if speaking to a real person. Start directly with the diagnosis. Keep it under two sentencesdon't any special symbol and 1 in the the result  """

IMAGE_SYSTEM_PROMPT = "Examine this image and provide a precise medical assessment in 1-2 sentences. Keep the response professional, clear, and to the point."  

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def load_llm():
    return ChatGroq(
        model="llama3-70b-8192",
        temperature=0.5,
        max_tokens=100
    )

def get_conversation_memory():
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# Function to Convert Text to Speech
def convert_text_to_speech(text, output_filepath="doctor_response.mp3"):
    """Converts text to speech and returns audio bytes."""
    text_to_speech_with_elevenlabs(input_text=text, output_filepath=output_filepath)
    
    with open(output_filepath, "rb") as audio_file:
        audio_bytes = audio_file.read()
    
    safe_file_delete(output_filepath)  # Cleanup after use
    return audio_bytes

def main():
    st.sidebar.title("🤖 MediScan-AI Assistant")
    st.sidebar.markdown("""
        MediScan-AI is your AI-powered medical assistant.
        ### Features:
        - 🩺 Medical Q&A
        - 🔬 Image Analysis
        - 🎙 Voice Input
    """)
    st.sidebar.button("🧹 Clear History", on_click=lambda: st.session_state.update({
        "messages": [],
        "chat_history": []
    }))

    st.title("🩺 MediScan-AI - Your Medical Assistant")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history with audio toggles
    for i, message in enumerate(st.session_state.messages):
        role = message["role"]
        with st.chat_message(role):
            st.markdown(message["content"])
            if "audio" in message:
                cols = st.columns([1, 20])
                with cols[0]:
                    if st.button("🔊", key=f"audio_icon_{i}"):
                        st.session_state[f"audio_visible_{i}"] = not st.session_state.get(f"audio_visible_{i}", False)
                with cols[1]:
                    if st.session_state.get(f"audio_visible_{i}", False):
                        st.audio(message["audio"], format="audio/mp3")

    # Custom CSS
    st.markdown("""
    <style>
    button[data-testid="baseButton-secondary"] {
        padding: 0 !important;
        min-width: 30px !important;
        height: 30px !important;
    }
    div[data-testid="stFileUploader"] {
        width: 40px !important;
        height: 40px !important;
    }
    div[data-testid="stFileUploader"]::before {
        content: '📎';
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        font-size: 1.2em;
    }
    
    /*///////upload image//////////*/
    /* Smaller file uploader */
div[data-testid="stFileUploader"] {
    width: 80px !important;
    height: 40px !important;
    min-height: 40px !important;
}

/* Hide original text */
div[data-testid="stFileUploader"] section {
    padding: 0 !important;
}

div[data-testid="stFileUploader"] section > div > span {
    display: none !important;
}

/* Style the upload box */
div[data-testid="stFileUploader"] dropzone {
    border: 1px dashed #ccc !important;
    border-radius: 4px !important;
    padding: 2px !important;
}

/* Add paperclip icon */
div[data-testid="stFileUploader"]::before {
    content: '📎';
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    font-size: 1.2em;
}

/* Remove hover effects */
div[data-testid="stFileUploader"]:hover dropzone {
    border-color: #ccc !important;
    background: transparent !important;
}

/* Style the 'Browse files' button */
div[data-testid="stFileUploader"] button {
    padding: 0.1rem 0.5rem !important;
    font-size: 0.8em !important;
    position: absolute;
    bottom: -25px;
    left: 50%;
    transform: translateX(-50%);
}

/* Remove uploaded file name */
div[data-testid="stFileUploader"] + div {
    display: none !important;
}
    </style>
    """, unsafe_allow_html=True)

    # Input section
    with st.container():
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_query = st.text_input("🔍 Ask your medical question here:", key="query_input")

        with col2:
            cols = st.columns(2)
            with cols[0]:
                if st.button("🎙", key="mic_button", help="Click and speak for 5 seconds"):
                    with st.spinner("🎤 Listening..."):
                        fs = 44100
                        duration = 5
                        recorded_audio = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype=np.int16)
                        sd.wait()
                        audio_filepath = "recorded_audio.wav"
                        with open(audio_filepath, "wb") as f:
                            write(audio_filepath, fs, recorded_audio)
                        st.session_state.audio_filepath = audio_filepath
                        st.rerun()
            
            with cols[1]:
                image_file = st.file_uploader("", type=["jpg", "png", "jpeg"], 
                                            label_visibility="collapsed",
                                            help="Upload medical images")

        if st.button("Analyze Symptoms", use_container_width=True):
            # Process text query
            if user_query:
                st.session_state.messages.append({"role": "user", "content": user_query})
                try:
                    vectorstore = get_vectorstore()
                    qa_chain = ConversationalRetrievalChain.from_llm(
                        llm=load_llm(),
                        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                        memory=get_conversation_memory(),
                        combine_docs_chain_kwargs={"prompt": PromptTemplate(
                            template=RAG_PROMPT_TEMPLATE,
                            input_variables=["context", "question"]
                        )},
                    )
                    response = qa_chain({"question": user_query})
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"**MediBot:** {response['answer']}"
                    })
                except Exception as e:
                    st.error(f"Query error: {str(e)}")
                    #voice input to produce the output
            
# Process voice only
            elif "audio_filepath" in st.session_state and "image_file" not in st.session_state:
                try:
                   # Speech to Text conversion
                        speech_text = transcribe_with_groq(
                       GROQ_API_KEY=os.environ.get("GROQ_API_KEY"), 
                       audio_filepath=st.session_state.audio_filepath,
                       stt_model="whisper-large-v3"
                   )
                        st.session_state.messages.append({"role": "user", "content": speech_text})

                # Generate response
                        llm = load_llm()
                   
                        doctor_response = llm.invoke(VOICE_PROMPT + speech_text)
                        doctor_response_text = doctor_response.content if hasattr(doctor_response, "content") else str(doctor_response)

                # Convert response to speech using the new function
                        audio_bytes = convert_text_to_speech(doctor_response_text)

                # Add assistant response with audio to chat
                        st.session_state.messages.append({
                              "role": "assistant", 
                              "content": f"**MediBot:** {doctor_response_text}",
                              "audio": audio_bytes
                            })

                  # Cleanup
                        safe_file_delete(st.session_state.audio_filepath)
                        del st.session_state.audio_filepath

                except Exception as e:
                    st.error(f"Voice processing error: {str(e)}")


        # Process voice and image together
            elif "audio_filepath" in st.session_state and "image_file" in st.session_state:
                try:
        # Speech to Text conversion
                    speech_text = transcribe_with_groq(
                    GROQ_API_KEY=os.environ.get("GROQ_API_KEY"), 
                    audio_filepath=st.session_state.audio_filepath,
                    stt_model="whisper-large-v3"
                    )
                
                # Add user's voice input to chat
                    st.session_state.messages.append({"role": "user", "content": speech_text})
        
                # Image Analysis
                    image_path = "uploaded_image.jpg"
                    with open(image_path, "wb") as f:
                       f.write(st.session_state.image_file.read())
        
                # Analyze Image with LLM
               
                    doctor_response = analyze_image_with_query(
                    query=SYSTEM_PROMPT + "\n" + speech_text,
                    encoded_image=encode_image(image_path),
                    model="llama-3.2-11b-vision-preview"
                    )

                # Convert response to speech using the new function
                    audio_bytes = convert_text_to_speech(doctor_response)

                # Add AI Doctor's response
                    st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"**AI Doctor:** {doctor_response}",
                    "audio": audio_bytes
                     })

                    # Cleanup
                    safe_file_delete(st.session_state.audio_filepath)
                    safe_file_delete(image_path)
                    del st.session_state.audio_filepath
                    del st.session_state.image_file

                except Exception as e:
                   st.error(f"Processing error: {str(e)}")
                   
                   
            elif "user_query" in st.session_state and "image_file" in st.session_state:
                try:
                    # Save the uploaded image
                    image_path = "uploaded_image.jpg"
                    with open(image_path, "wb") as f:
                        f.write(st.session_state.image_file.read())

                    # Analyze Image with LLM
                    doctor_response = analyze_image_with_user_query(
                        query=SYSTEM_PROMPT + "\n" + st.session_state.user_query,
                        encoded_image=encode_image(image_path),
                        model="llama-3.2-11b-vision-preview"
                    )

                    # Convert response to speech
                    audio_bytes = convert_text_to_speech(doctor_response)

                    # Add response to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"**AI Doctor:** {doctor_response}",
                        "audio": audio_bytes
                    })

                    # Cleanup
                    safe_file_delete(image_path)
                    del st.session_state.image_file
                    del st.session_state.user_query  # Optional: remove query if it's no longer needed

                except Exception as e:
                    st.error(f"Processing error: {str(e)}")

            # #analysis from the images only no prompt and voice
            # elif st.session_state.get("image_file"):
            #         try:
            #             # Save uploaded image
            #             image_path = "uploaded_image.jpg"
            #             with open(image_path, "wb") as f:
            #                 f.write(st.session_state.image_file.read())

            #             # Encode and analyze image
            #             doctor_response = analyze_image_with_prompt(encode_image(image_path))

            #             # Convert response to speech
            #             audio_bytes = convert_text_to_speech(doctor_response)

            #             # Add response to chat
            #             st.session_state.messages.append({
            #                 "role": "assistant",
            #                 "content": f"**AI Doctor:** {doctor_response}",
            #                 "audio": audio_bytes
            #             })

            #             # Cleanup
            #             safe_file_delete(image_path)
            #             st.session_state.image_file = None  

            #         except Exception as e:
            #             st.error(f"Error analyzing image: {str(e)}")


                    
                     
                     
        
            st.rerun()

def safe_file_delete(path, max_retries=5):
    for _ in range(max_retries):
        try:
            if os.path.exists(path):
                os.remove(path)
                return True
        except:
            time.sleep(0.1)
    return False

if __name__ == "__main__":
    main()