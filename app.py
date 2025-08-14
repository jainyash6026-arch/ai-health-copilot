import streamlit as st
import google.generativeai as genai
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
import io
from pydub import AudioSegment
import os
from gtts import gTTS
from langdetect import detect, LangDetectException

# --- SETUP ---
st.set_page_config(layout="wide")
st.title("️️Voice-Powered Health Insurance Co-pilot")

# Configure Google AI
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
except (KeyError, FileNotFoundError):
    st.error("API key not found. Please create a secrets.toml file.")
    st.stop()

# --- AI PROMPT ---
system_prompt = """
You are "HealthGenie," a smart, empathetic, and expert AI voice assistant for health insurance in India. Your personality is that of a knowledgeable and trustworthy financial friend who demystifies the complexities of insurance. Your primary goal is to conduct a natural, multi-step conversation to deeply understand a user's unique life situation and provide a single, holistic, and hyper-personalized recommendation.

### Core Principles:
- **Language Adaptability: You must automatically detect the language the user is speaking (e.g., English, Hindi, or Hinglish) and conduct the entire conversation in that same language. If the user is speaking Hinglish, you should also respond in a natural, conversational Hinglish. Avoid overly formal or traditional Hindi unless the user's language is also formal.**
- **Be Consultative:** Your goal is to understand and advise, not just to collect data and fill a template.
- **Maintain a Natural Rhythm:** Ask one question at a time to keep the conversation flowing smoothly. Avoid overwhelming the user.
- **Explain the "Why":** Build trust by briefly explaining why you're asking a question (e.g., "To recommend the right coverage amount, it's helpful to know...").
- **Synthesize, Don't Script:** Your final recommendation should be a unique synthesis of all the information the user has shared. It should feel completely tailored to them, referencing their specific needs and situation throughout.

### Key Data Points to Understand:
Your conversation should aim to organically understand the following about the user:
1.  **Beneficiaries:** Is the plan for an individual, a family, or parents?
2.  **Core Need/Persona:** What is their primary motivation? (e.g., first-time buyer, managing a chronic condition, planning for maternity, seeking wellness benefits, etc.).
3.  **Financial Context:** What is their approximate budget? Which city do they live in? (This influences the Sum Insured).
4.  **Professional Context:** What is their line of work? (This helps in identifying risks and suggesting specific riders like Hospicash or Personal Accident cover).
5.  **Existing Coverage:** Do they have a corporate plan? If so, what is the coverage amount? (This is key for recommending a Super Top-up).
6.  **Healthcare Preferences:** Do they have a preferred hospital? A preference for "top hospitals" implies a need for a higher Sum Insured.

### Crafting the Final Recommendation:
When you have gathered a clear picture, deliver a holistic "Health Insurance Blueprint" that includes the following logical components:

1.  **The Ideal Base Plan:** Recommend a specific type of plan (e.g., Standard, Family Floater, Senior Citizen) that directly addresses their core need.
2.  **The Right Sum Insured:** Justify your recommended coverage amount by connecting it to their city of residence and healthcare preferences (e.g., "Since you're in a metro city and prefer top-tier hospitals where costs are higher, a sum insured of at least ₹20 Lakhs is advisable.")
3.  **A Smart Professional Strategy:** Provide a powerful, specific insight based on their job.
    -   For **private sector employees**, you must explain the risk of job changes and recommend an independent base plan as their primary safety net, with a Super Top-up as a secondary layer.
    -   For the **self-employed** or **gig workers**, you must emphasize income protection through riders like Hospicash and Personal Accident cover.
4.  **The Most Valuable Add-on(s):** Recommend specific riders that solve the user's stated problems. If they mentioned "day-to-day wellness," you must recommend an OPD benefit. If they're buying for parents, you must recommend a Critical Illness rider.
5.  **Key Features to Look For:** Conclude by empowering the user with knowledge. Suggest 2-3 specific, high-value features they should look for when comparing plans, such as a high No-Claim Bonus, a Restoration Benefit, or low waiting periods.

Your ultimate goal is to make the user feel truly understood and confident in the advice you've provided.
"""

# --- Initialize Chat History in Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "model", "parts": [system_prompt]}]

# --- FUNCTIONS ---
def transcribe_audio(audio_bytes):
    r = sr.Recognizer()
    try:
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        wav_io = io.BytesIO()
        audio_segment.export(wav_io, format="wav")
        wav_io.seek(0)
        with sr.AudioFile(wav_io) as source:
            audio_data = r.record(source)
        # Recognize speech using Google Speech Recognition, try multiple languages
        try:
            # Try recognizing in English
            text = r.recognize_google(audio_data, language='en-IN')
        except sr.UnknownValueError:
            # If English fails, try Hindi
            text = r.recognize_google(audio_data, language='hi-IN')
        return text
    except Exception:
        return None

# This function now has MULTILINGUAL support
def text_to_speech(text):
    # Detect the language of the text
    try:
        lang = detect(text)
    except LangDetectException:
        # Default to English if detection fails
        lang = 'en'
    
    # Generate the audio using the detected language
    tts = gTTS(text=text, lang=lang, tld='co.in')
    slow_audio_fp = io.BytesIO()
    tts.write_to_fp(slow_audio_fp)
    slow_audio_fp.seek(0)
    
    # Load the audio into pydub to speed it up
    audio_segment = AudioSegment.from_file(slow_audio_fp, format="mp3")
    fast_audio_segment = audio_segment.speedup(playback_speed=1.25)
    
    # Export the fast audio to an in-memory file
    fast_audio_fp = io.BytesIO()
    fast_audio_segment.export(fast_audio_fp, format="mp3")
    fast_audio_fp.seek(0)
    
    return fast_audio_fp.read()

# --- APP INTERFACE ---
st.write("Welcome! Click the 'Start Recording' button to have a conversation about your insurance needs.")

for message in st.session_state.messages:
    if message["parts"][0] != system_prompt:
        with st.chat_message(message["role"]):
            st.write(message["parts"][0])

audio = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop Recording", key='recorder')

if audio is not None and audio['bytes']:
    audio_bytes = audio['bytes']
    
    with st.spinner("Transcribing..."):
        user_text = transcribe_audio(audio_bytes)

    if user_text:
        st.session_state.messages.append({"role": "user", "parts": [user_text]})
        with st.chat_message("user"):
            st.write(user_text)

        with st.spinner("AI is thinking..."):
            model = genai.GenerativeModel('gemini-1.5-flash')
            chat = model.start_chat(history=st.session_state.messages)
            response = chat.send_message(user_text)
            ai_response_text = response.text

        st.session_state.messages.append({"role": "model", "parts": [ai_response_text]})
        with st.chat_message("model"):
            st.write(ai_response_text)
            st.audio(text_to_speech(ai_response_text))