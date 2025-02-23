from gtts import gTTS
import os
import platform
import subprocess
from elevenlabs import ElevenLabs
from playsound import playsound
from pydub import AudioSegment
from pydub.playback import play

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")

# def text_to_speech_with_gtts(input_text, output_filepath):
#     language = "en"

#     # Generate speech using gTTS
#     audioobj = gTTS(
#         text=input_text,
#         lang=language,
#         slow=False
#     )
#     audioobj.save(output_filepath)

#     os_name = platform.system()
#     try:
#         if os_name == "Darwin":  # macOS
#             subprocess.run(['afplay', output_filepath])
#         elif os_name == "Windows":  # Windows
#             audio = AudioSegment.from_file(output_filepath, format="mp3")
#             play(audio)  # More reliable than playsound
#         elif os_name == "Linux":  # Linux
#             subprocess.run(['mpg123', output_filepath])  # Alternative: use 'aplay' or 'ffplay'
#         else:
#             raise OSError("Unsupported operating system")
#     except Exception as e:
#         print(f"An error occurred while trying to play the audio: {e}")

def text_to_speech_with_elevenlabs(input_text, output_filepath):
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    
    audio = client.generate(
        text=input_text,
        voice="Aria",
        output_format="mp3_22050_32",
        model="eleven_turbo_v2"
    )
    
    with open(output_filepath, "wb") as f:
        for chunk in audio:  # Corrected: Iterate over the generator
            f.write(chunk)

    os_name = platform.system()
    try:
        if os_name == "Darwin":  # macOS
            subprocess.run(['afplay', output_filepath])
        elif os_name == "Windows":  # Windows
            playsound(output_filepath)  # Fixed autoplay
        elif os_name == "Linux":  # Linux
            subprocess.run(['mpg123', output_filepath])  # Alternative: use 'ffplay'
        else:
            raise OSError("Unsupported operating system")
    except Exception as e:
        print(f"An error occurred while trying to play the audio: {e}")
