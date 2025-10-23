import os
import json
import requests
import logging
import tempfile
import time
from datetime import datetime, timedelta
from urllib.parse import unquote
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from groq import Groq
import firebase_admin
from firebase_admin import credentials, firestore, storage
import uuid
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
import re
import hashlib
from collections import OrderedDict, defaultdict
from google.cloud.firestore import FieldFilter
import statistics
from openai import AzureOpenAI

# --- Timezone Imports and Configuration ---
from datetime import timezone
import zoneinfo # Standard library for modern Python (3.9+)
import sys

# Define IST Timezone explicitly
try:
    # Use standard zoneinfo name for India Standard Time (UTC+5:30)
    IST_TIMEZONE = zoneinfo.ZoneInfo("Asia/Kolkata")
except zoneinfo.ZoneInfoNotFoundError:
    # Fallback for systems without zoneinfo data or older Python (pre-3.9)
    # Check Python version, use pytz if available and zoneinfo isn't working
    if sys.version_info < (3, 9):
        # NOTE: If this environment is pre-Python 3.9, 'pytz' is required
        try:
            import pytz
            IST_TIMEZONE = pytz.timezone("Asia/Kolkata")
        except ImportError:
            # Final fallback: Fixed offset, which doesn't handle DST but IST has none.
            IST_TIMEZONE = timezone(timedelta(hours=5, minutes=30))
    else:
        # Final fallback for modern python if ZoneInfo fails unexpectedly
        IST_TIMEZONE = timezone(timedelta(hours=5, minutes=30))
except Exception:
    IST_TIMEZONE = timezone(timedelta(hours=5, minutes=30))


def get_now_ist():
    """Returns a timezone-aware datetime object for the current time in IST."""
    return datetime.now(IST_TIMEZONE)
# ------------------------------------------

# === New Imports for Tone Analysis ===
import librosa
import numpy as np
import soundfile as sf
# =====================================

load_dotenv()

# Load environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Azure OpenAI environment variables
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_NAME = os.environ.get("AZURE_DEPLOYMENT_NAME")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION")

# Firebase Storage environment variables
FIREBASE_STORAGE_BUCKET = os.environ.get("FIREBASE_STORAGE_BUCKET")

# -------------------------------------------------
# Logging Configuration (Modified to be neat and explicit)
# -------------------------------------------------
def setup_logging():
    """Sets up neat and IST-aware logging."""
    # Custom format for neat log output: [TYPE] TIMESTAMP | FILENAME:LINE - MESSAGE
    log_format = (
        "[%(levelname)s] %(asctime)s | %(filename)s:%(lineno)d - %(message)s"
    )
    
    # Configure logging with a custom formatter that forces IST (if possible)
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S IST")
    
    # Custom converter function to use the IST timezone when logging
    def ist_time_converter(timestamp):
        dt = datetime.fromtimestamp(timestamp, IST_TIMEZONE)
        return dt.timetuple()

    # Apply the custom converter
    formatter.converter = ist_time_converter

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    # Set the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [] # Clear existing handlers
    root_logger.addHandler(handler)

    # Configure __main__ logger for internal messages
    main_logger = logging.getLogger(__name__)
    main_logger.setLevel(logging.INFO)
    
    # Suppress Flask/Werkzeug default request logs
    werkzeug_logger = logging.getLogger("werkzeug")
    werkzeug_logger.setLevel(logging.ERROR)
    
    return main_logger

logger = setup_logging()

# -------------------------------------------------
# Firebase Initialization
# -------------------------------------------------
try:
    # Create Firebase credentials from environment variables
    firebase_cred_dict = {
        "type": os.environ.get("GOOGLE_SERVICE_ACCOUNT_TYPE"),
        "project_id": os.environ.get("GOOGLE_PROJECT_ID"),
        "private_key_id": os.environ.get("GOOGLE_PRIVATE_KEY_ID"),
        "private_key": os.environ.get("GOOGLE_PRIVATE_KEY", "").replace('\\n', '\n'),
        "client_email": os.environ.get("GOOGLE_CLIENT_EMAIL"),
        "client_id": os.environ.get("GOOGLE_CLIENT_ID"),
        "auth_uri": os.environ.get("GOOGLE_AUTH_URI"),
        "token_uri": os.environ.get("GOOGLE_TOKEN_URI"),
        "auth_provider_x509_cert_url": os.environ.get("GOOGLE_AUTH_PROVIDER_CERT_URL"),
        "client_x509_cert_url": os.environ.get("GOOGLE_CLIENT_CERT_URL")
    }
    
    cred = credentials.Certificate(firebase_cred_dict)
    firebase_admin.initialize_app(cred, {
        'storageBucket': FIREBASE_STORAGE_BUCKET
    })
    db = firestore.client()
    bucket = storage.bucket()
    logger.info("Firebase initialized successfully with Storage")
except Exception as e:
    logger.error(f"Failed to initialize Firebase: {str(e)}")
    db = None
    bucket = None

# -------------------------------------------------
# Azure OpenAI Initialization
# -------------------------------------------------
azure_openai_client = None
if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
    try:
        azure_openai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        logger.info("Azure OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI: {str(e)}")
        azure_openai_client = None
else:
    logger.warning("Azure OpenAI credentials not found")

# -------------------------------------------------
# Flask App
# -------------------------------------------------
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 # 50MB max file size

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Processed calls cache with LRU eviction (max 1000 entries)
processed_calls = OrderedDict()
MAX_PROCESSED_CALLS = 1000

# -------------------------------------------------
# Utility Functions with Retry Logic
# -------------------------------------------------
def decode_url_encoded_values(data: dict) -> dict:
    """Decode URL-encoded values from Kaleyra webhook."""
    decoded = {}
    for key, value in data.items():
        decoded[key] = unquote(value) if isinstance(value, str) else value
    return decoded

def generate_call_id(call_data):
    """Generate a unique call ID based on call metadata to prevent duplicates"""
    call_id = call_data.get('id', '')
    if not call_id:
        # Create a hash from call metadata if no ID provided
        call_string = f"{call_data.get('caller', '')}-{call_data.get('called', '')}-{call_data.get('starttime', '')}"
        call_id = hashlib.md5(call_string.encode()).hexdigest()[:12]
    return call_id

def is_call_processed(call_id):
    """Check if call has already been processed to prevent duplicates"""
    # Check in-memory cache first
    if call_id in processed_calls:
        return True
    
    # Check Firestore for existing call with this ID
    if db:
        try:
            calls_ref = db.collection('call_analysis')
            query = calls_ref.where(filter=firestore.FieldFilter('callId', '==', call_id)).limit(1)
            docs = query.stream()
            if any(True for _ in docs):
                # Add to cache to avoid future Firestore queries
                add_to_processed_cache(call_id)
                return True
        except Exception as e:
            logger.error(f"Error checking Firestore for call ID {call_id}: {str(e)}")
    
    return False

def add_to_processed_cache(call_id):
    """Add call ID to processed cache with LRU eviction"""
    if call_id in processed_calls:
        # Move to end (most recently used)
        processed_calls.move_to_end(call_id)
    else:
        processed_calls[call_id] = True
        # Remove oldest item if cache is full
        if len(processed_calls) > MAX_PROCESSED_CALLS:
            processed_calls.popitem(last=False)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def download_audio(url):
    """Download audio from URL and return temporary file path with retry logic"""
    try:
        response = requests.get(url, stream=True, timeout=360)
        response.raise_for_status()
        
        # Create a temporary file
        # Use .mp3 suffix
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        temp_file.close()
        
        return temp_file.name, None
    except requests.exceptions.RequestException as e:
        return None, f"Failed to download audio: {str(e)}"
    except IOError as e:
        return None, f"Failed to write audio file: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error downloading audio: {str(e)}"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def transcribe_audio(file_path):
    """Transcribe audio using Groq API with Whisper (with retry for large files)"""
    if not groq_client:
        return None, "Groq client not initialized"
    
    try:
        with open(file_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(file_path, file.read()),
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
            )
            return transcription, None
    except Exception as e:
        return None, f"Transcription failed: {str(e)}"

def detect_language(transcription):
    """Detect language from Whisper transcription"""
    if not transcription or not hasattr(transcription, 'language'):
        return "Unknown"
    
    language_map = {
        'en': 'English',
        'hi': 'Hindi',
        'ta': 'Tamil',
        'te': 'Telugu',
        'kn': 'Kannada',
        'ml': 'Malayalam'
    }
    
    return language_map.get(transcription.language, transcription.language)

# -------------------------------------------------
# Acoustic Feature and Tone Analysis Functions
# -------------------------------------------------

def extract_features(audio_path):
    """Extract acoustic features (pitch, energy, tempo, ZCR) from the audio."""
    try:
        # Load audio data. Librosa handles various formats (like mp3)
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        logger.error(f"librosa load error for {audio_path}: {e}")
        return None

    # Pitch
    pitches, _ = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[pitches > 0]
    avg_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0.0
    pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0.0

    # Energy (RMS)
    # Using np.mean(librosa.feature.rms(y=y)[0]) for single value
    rms = librosa.feature.rms(y=y)[0]
    avg_energy = np.mean(rms)
    energy_std = np.std(rms)

    # Tempo (Speech Rate Proxy)
    try:
        # tempo returns a single float or array; we take the first element
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr, start_bpm=120)  
        tempo_value = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
    except Exception:
        tempo_value = 0.0
        
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    avg_zcr = np.mean(zcr)

    return {
        "avg_pitch": round(float(avg_pitch), 2),
        "pitch_std": round(float(pitch_std), 2),
        "avg_energy": round(float(avg_energy), 4),
        "energy_std": round(float(energy_std), 4),
        "tempo": round(tempo_value, 2),
        "avg_zcr": round(float(avg_zcr), 4),
        "duration_sec": round(librosa.get_duration(y=y, sr=sr), 2),
    }

def split_audio_channels(audio_path):
    """Split stereo audio into two mono temporary files (agent and customer)."""
    try:
        # Load as stereo (mono=False)
        y, sr = librosa.load(audio_path, sr=None, mono=False)
    except Exception as e:
        logger.error(f"librosa load error for splitting {audio_path}: {e}")
        return audio_path, audio_path, None # Treat as mono if load fails

    if y.ndim == 1:
        # Mono file, cannot split
        return audio_path, audio_path, sr

    # Stereo file: y[0] is typically left (Agent), y[1] is right (Customer)
    agent_temp = tempfile.NamedTemporaryFile(delete=False, suffix='_agent.wav')
    cust_temp = tempfile.NamedTemporaryFile(delete=False, suffix='_cust.wav')
    agent_temp.close()
    cust_temp.close()

    # Write each channel to a separate temporary WAV file
    sf.write(agent_temp.name, y[0], sr)
    sf.write(cust_temp.name, y[1], sr)
    
    logger.info(f"Split stereo audio into {agent_temp.name} and {cust_temp.name}")

    return agent_temp.name, cust_temp.name, sr

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def analyze_tone_with_azure(agent_features, customer_features):
    """Analyze tone and emotion with Azure OpenAI and return mood and tone score."""
    if not azure_openai_client:
        return None, "Azure OpenAI client not initialized"

    prompt = f"""
You are an expert in customer service call tone analysis for a saree company named Prashanti.

You are given extracted acoustic emotion features (pitch, energy, tempo, ZCR) from a customer service call.
Your task is to interpret these features and provide a summarized tone analysis in a specific JSON format.

**Agent Features:**
{agent_features}

**Customer Features:**
{customer_features}

**Interpretation Guidance:**
- **Pitch:** Indicates excitement, confidence, or nervousness. Higher pitch can mean excitement; low, steady pitch can mean composure.
- **Energy:** Reflects enthusiasm, loudness, or fatigue. High energy means engagement; low energy means boredom or calmness.
- **Tempo:** Shows speech speed, engagement, or impatience. Faster tempo can be urgency or excitement; slower can be measured or fatigued.
- **Agent Mood:** Use 1-2 primary terms: Calm, Energetic, Empathetic, Confident, Stressed, Monotone, Hurried, Engaging.
- **Customer Mood:** Use 1-2 primary terms: Cooperative, Neutral, Confused, Frustrated, Angry, Pleased, Thankful, Impatient.
- **Tone Mark (0–10):** Score the agent's tone quality (empathy, enthusiasm, and composure). This score is CRITICAL and represents the overall **acoustic quality** of the agent's performance.

Provide ONLY a valid JSON object with the following exact structure:
{{
    "agent_mood": "The agent's primary mood (e.g., Confident and Engaging)", 
    "customer_mood": "The customer's primary mood (e.g., Frustrated and Impatient)", 
    "tone_mark": 8, 
    "reasoning": "A brief, 1-2 sentence interpretation of the features leading to the scores."
}}
"""

    try:
        response = azure_openai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"}
        )

        response_text = response.choices[0].message.content
        if not response_text:
            return None, "Tone analysis failed: Empty response"

        tone_analysis = json.loads(response_text)
        
        # Basic validation and type coercion
        if 'tone_mark' in tone_analysis:
            try:
                # Ensure tone_mark is an integer between 0 and 10
                tone_analysis['tone_mark'] = max(0, min(10, int(tone_analysis.get('tone_mark', 5))))
            except:
                tone_analysis['tone_mark'] = 5 # Default on failure
        else:
            tone_analysis['tone_mark'] = 5

        # Add default for required mood fields if missing
        tone_analysis['agent_mood'] = tone_analysis.get('agent_mood', 'Unknown')
        tone_analysis['customer_mood'] = tone_analysis.get('customer_mood', 'Unknown')
        
        logger.info(f"Tone analysis successful: Agent Mood={tone_analysis['agent_mood']}, Tone Mark={tone_analysis['tone_mark']}")
        return tone_analysis, None

    except Exception as e:
        logger.error(f"Azure OpenAI tone analysis error: {str(e)}")
        return None, f"Tone analysis failed: {str(e)}"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def analyze_call_with_azure_openai(transcript_text, language, tone_mark):
    """Generate comprehensive call analysis using Azure OpenAI GPT-4 with optimized sentiment detection and call type classification"""
    if not azure_openai_client:
        return None, "Azure OpenAI client not initialized"
    
    # Inject Tone Mark into the prompt for holistic scoring guidance
    system_prompt_addition = f"""
    The agent's acoustic tone quality score (Tone Mark) is: {tone_mark}/10. 
    Use this score to strongly influence your scoring of 'clarity', 'confidence', 'sympathy', and 'intro' to ensure acoustic performance is a primary factor.
    """

    try:
        analysis_prompt = f"""
        You are a call analysis system for a saree company named Prashanti. Your task is to evaluate the customer's sentiment and the call's overall outcome from a customer service perspective. This is a call transcript in {language}.
        
        {system_prompt_addition}

        **CALL TYPE CLASSIFICATION CATEGORIES:**
        
        1. **Product-related Queries**
             - Questions about saree collections, fabrics, designs, colors
             - Size and measurement inquiries
             - Product availability and stock checks
             - Pricing and discount questions
             - Customization requests
        
        2. **Service Queries**
             - Store locations and timings
             - Tailoring services
             - Styling advice and recommendations
             - Exchange policies
             - Delivery options
        
        3. **Loyalty & Membership Queries**
             - Loyalty program benefits
             - Membership registration
             - Points and rewards inquiries
             - Special member discounts
             - Membership tier questions
        
        4. **Technical Queries (Online Platforms)**
             - Website/Shopify login issues
             - Online ordering problems
             - Payment gateway errors
             - Account management
             - Digital catalog access
        
        5. **Complaint & Feedback Queries**
             - Product quality complaints
             - Delivery delays
             - Wrong items received
             - Customer service feedback
             - Return requests
        
        6. **Order Management**
             - Order status tracking
             - Order modification requests
             - Cancellation requests
             - Bulk order inquiries
             - Shipping updates
        
        7. **Sales & Promotion Inquiries**
             - Current offers and promotions
             - Festival discounts
             - Seasonal sales
             - Corporate bulk discounts
             - Wedding collection offers
        
        **CLASSIFICATION RULES:**
        - Choose the PRIMARY category that best represents the main purpose of the call
        - If multiple categories apply, select the most dominant one
        - For complaint-related calls, use "Complaint & Feedback Queries" even if it involves products or services
        - For technical website issues, use "Technical Queries" regardless of context
        
        Provide ONLY a valid JSON object with the following exact structure:
        
        {{
            "summary": "2-line summary of the call",
            "call_type": {{
                "primary_category": "exact_category_name_from_above_list",
                "sub_category": "specific_sub_topic_based_on_call_content",
                "confidence_score": 0.95,
                "secondary_categories": ["list_of_other_relevant_categories"]
            }},
            "objections": ["list of top 3 customer objections with context"],
            "competitors": ["list of competitors mentioned with context about what was said"],
            "scores": {{
                "structure": score_out_of_10_based_on_call_flow,
                "clarity": score_out_of_10_based_on_communication_clearness,
                "confidence": score_out_of_10_based_on_agent_confidence,
                "closing": score_out_of_10_based_on_closure_effectiveness,
                "intro": 10_if_prashanti_mentioned_in_first_3_seconds_else_0,
                "call_summary": score_out_of_10_based_on_whether_agent_provided_summary_at_end,
                "end_call": score_out_of_10_based_on_whether_agent_asked_for_additional_queries,
                "upselling": score_out_of_10_based_on_upselling_attempts,
                "sympathy": score_out_of_10_based_on_use_of_polite_language
            }},
            "coaching": ["3 specific coaching tips based on actual transcript"],
            "filler_words_count": number_of_filler_words_used,
            "talk_ratio": "customer:agent ratio (e.g., 40:60)",
            "key_topics": ["detailed main topics discussed with context"],
            "call_purpose": "Primary purpose of the call (e.g., Order Status, Refund, Query, Sales Inquiry, Complaint)",
            "sentiment": "overall customer satisfaction sentiment of the call (POSITIVE/NEGATIVE/NEUTRAL)",
            "hold_time": total_seconds_agent_asked_caller_to_hold,
            "call_analysis": {{
                "company_intro_early": boolean_if_agent_introduced_company_in_first_3_seconds,
                "provided_summary": boolean_if_agent_summarized_conversation_at_end,
                "asked_for_more_queries": boolean_if_agent_asked_about_additional_queries,
                "upselling_attempted": boolean_if_agent_tried_upselling,
                "polite_language_used": boolean_if_agent_used_polite_words
            }},
            "call_sections": {{
                "intro": {{"summary": "brief summary", "present": true/false}},
                "discovery": {{"summary": "brief summary", "present": true/false}},
                "demo": {{"summary": "brief summary", "present": true/false}},
                "objection": {{"summary": "brief summary", "present": true/false}},
                "closure": {{"summary": "brief summary", "present": true/false}}
            }},
            "intro_check": "Yes" if the agent introduced themselves as being from Prashanti Sarees in their greeting, otherwise "No"
        }}
        
        For filler words, look for: um, uh, like, you know, actually, basically, sort of, kind of, well, so, right, okay.
        
        **SENTIMENT GUIDELINES FOR CUSTOMER SERVICE CONTEXT:**
        
        **POSITIVE:**
        - The customer's primary query or issue was successfully resolved.
        - The customer expresses thanks, happiness, or satisfaction with the service.
        - The agent effectively de-escalated a difficult situation and the customer was receptive.
        - The call ends with the customer in a positive or appreciative mood.
        
        **NEGATIVE:**
        - The customer's issue was not resolved.
        - The customer expresses frustration, anger, or dissatisfaction.
        - The customer requests to speak with a manager or threatens to leave.
        - The call ends abruptly, or the customer uses rude/angry language.
        - The agent fails to provide a clear path to resolution.
        
        **NEUTRAL:**
        - The call is purely informational without clear emotional markers.
        - The customer's query is answered, but no positive or negative emotion is expressed.
        - The call ends without a clear resolution, but also without frustration.
        - The conversation is short and transactional, such as a quick status check.
        
        CRITICAL: First, determine the 'call_purpose'. Then, classify 'sentiment' based on whether the agent successfully handled that purpose to the customer's satisfaction. A resolved complaint is a POSITIVE outcome, not a NEUTRAL one. A sales call with buying signals is also a POSITIVE outcome.
        
        Transcript: {transcript_text[:15000]}
        """
        
        response = azure_openai_client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": analysis_prompt}],
            temperature=0.3,
            max_tokens=1500,
            top_p=0.95,
            response_format={"type": "json_object"}
        )
        
        response_text = response.choices[0].message.content
        
        # Validate response is not empty
        if not response_text or response_text.strip() == "":
            return None, "Analysis failed: Empty response from Azure OpenAI"
        
        # Parse JSON response with better error handling
        try:
            analysis = json.loads(response_text)
            
            call_type = analysis.get('call_type', {})
            if not isinstance(call_type, dict):
                analysis['call_type'] = {
                    'primary_category': 'Unknown',
                    'sub_category': 'Unknown',
                    'confidence_score': 0.0,
                    'secondary_categories': []
                }
            else:
                required_fields = ['primary_category', 'sub_category', 'confidence_score', 'secondary_categories']
                for field in required_fields:
                    if field not in call_type:
                        if field == 'primary_category':
                            call_type[field] = 'Unknown'
                        elif field == 'sub_category':
                            call_type[field] = 'Unknown'
                        elif field == 'confidence_score':
                            call_type[field] = 0.0
                        elif field == 'secondary_categories':
                            call_type[field] = []
                
            talk_ratio = analysis.get('talk_ratio', '')
            if talk_ratio and not re.match(r'^\d+:\d+$', talk_ratio):
                ratio_match = re.search(r'(\d+:\d+)', talk_ratio)
                if ratio_match:
                    analysis['talk_ratio'] = ratio_match.group(1)
                else:
                    analysis['talk_ratio'] = "50:50"
                    logger.warning(f"Invalid talk_ratio format: {talk_ratio}. Defaulting to 50:50")
                    
            sentiment = analysis.get('sentiment', '').upper()
            valid_sentiments = ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
            if sentiment not in valid_sentiments:
                logger.warning(f"Invalid sentiment value: {sentiment}. Defaulting to NEUTRAL")
                analysis['sentiment'] = 'NEUTRAL'
            else:
                analysis['sentiment'] = sentiment
                
            primary_category = analysis.get('call_type', {}).get('primary_category', 'Unknown')
            logger.info(f"Call type classification: {primary_category}, Sentiment: {sentiment}")
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed. Response was: {response_text}")
            return None, f"Analysis failed: Invalid JSON response - {str(e)}"
        
        return analysis, None
        
    except Exception as e:
        logger.error(f"Azure OpenAI analysis error: {str(e)}")
        return None, f"Analysis failed: {str(e)}"

def calculate_call_score(analysis, tone_mark):
    """Calculate overall call score from individual metrics, giving high priority to the Tone Mark."""
    scores = analysis.get("scores", {})
    
    # Apply a high weight to the Tone Mark (e.g., 20% of the total score)
    
    # Recalculate component weights to sum to 1.0 (or adjust the denominator)
    if scores:
        # Standardize scores (total weight = 0.80)
        weighted_score_components = (
            scores.get("structure", 0) * 0.12 +    # Decreased from 0.15
            scores.get("clarity", 0) * 0.12 +      # Decreased from 0.15
            scores.get("confidence", 0) * 0.12 +  # Decreased from 0.15
            scores.get("closing", 0) * 0.12 +      # Decreased from 0.15
            scores.get("intro", 0) * 0.08 +        # Decreased from 0.10
            scores.get("call_summary", 0) * 0.08 + # Decreased from 0.10
            scores.get("end_call", 0) * 0.08 +      # Decreased from 0.10
            scores.get("upselling", 0) * 0.04 +    # Decreased from 0.05
            scores.get("sympathy", 0) * 0.04          # Decreased from 0.05
        ) # Total weight = 0.80

        # High-priority Tone Mark (20% weight)
        tone_weight = 0.20
        tone_score = tone_mark * tone_weight

        overall_score = round(weighted_score_components + tone_score, 1)
        return overall_score
        
    # If no scores are present, use Tone Mark if available
    return round(tone_mark * 1.0, 1) if tone_mark is not None else 0

def get_agent_by_phone_number(phone_number, call_source="INCOMING"):
    """Retrieve agent from Firestore by phone number, handling both INCOMING and C2C calls"""
    if not db:
        return None
    
    try:
        # Clean phone number (remove any non-digit characters)
        clean_phone = ''.join(filter(str.isdigit, phone_number))
        
        if not clean_phone:
            return None
            
        # Query agents collection with proper filter syntax
        agents_ref = db.collection('agents')
        query = agents_ref.where(filter=firestore.FieldFilter('phone', '==', clean_phone)).limit(1)
        docs = query.stream()
        
        for doc in docs:
            return {**doc.to_dict(), 'id': doc.id}
        
        logger.warning(f"No agent found for phone: {clean_phone} (source: {call_source})")
        return None
    except Exception as e:
        logger.error(f"Error fetching agent for phone {phone_number}: {str(e)}")
        return None

def get_next_call_document_name():
    try:
        # Find the highest existing call number
        calls_ref = db.collection('call_analysis')
        # Using a fixed arbitrary string like 'Call_' is bad practice for scale. 
        # For simplicity and to match the original code, we keep the original slow logic.
        docs = calls_ref.stream()
        
        max_num = 0
        for doc in docs:
            doc_id = doc.id
            if doc_id.startswith('Call_'):
                try:
                    num = int(doc_id.split('_')[1])
                    max_num = max(max_num, num)
                except (ValueError, IndexError):
                    continue
        
        return f"Call_{max_num + 1:02d}"
    except Exception as e:
        logger.error(f"Error finding max call number: {str(e)}")
        return f"Call_{int(time.time())}"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def upload_to_firebase_storage(file_path, agent_email, call_id):
    if not bucket:
        return None, "Firebase Storage not initialized"
    
    try:
        # Generate unique filename with IST timestamp
        timestamp = get_now_ist().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_recordings/{agent_email}/{call_id}_{timestamp}.mp3"
        
        # Create blob with 30-day expiration
        blob = bucket.blob(filename)
        
        # Calculate expiration time (IST aware, but stored in Firestore Timestamp compatible format)
        expiration_time = get_now_ist() + timedelta(days=30)
        
        # Set metadata for automatic deletion after 30 days
        blob.metadata = {
            'firebaseStorageDownloadTokens': str(uuid.uuid4()),
            # Store expiration time in ISO format (which is implicitly UTC if not specified, but fine for calculation)
            'deleteAfter': expiration_time.isoformat() 
        }
        
        # Upload file
        blob.upload_from_filename(file_path, content_type='audio/mpeg')
        
        # Make the file publicly accessible and get download URL
        blob.make_public()
        download_url = blob.public_url
        
        # Add the token to the URL for authentication
        token = blob.metadata.get('firebaseStorageDownloadTokens', '')
        if token:
            download_url = f"{download_url}?alt=media&token={token}"
        
        logger.info(f"Audio file uploaded to Firebase Storage: {filename}")
        return download_url, None
        
    except Exception as e:
        return None, f"Failed to upload to Firebase Storage: {str(e)}"

# ====================================================================
# NEW FUNCTION: Update C2C Specific Stats
# ====================================================================
def update_c2c_stats(call_data, call_type="answered"):
    """
    Update Click2Call volume statistics. This is a dedicated counter
    separate from Inbound volume.
    """
    if not db:
        return
        
    try:
        c2c_ref = db.collection('c2c_stats').document('overall')
        
        # Get current document
        c2c_doc = c2c_ref.get()
        
        if c2c_doc.exists:
            c2c_data = c2c_doc.to_dict()
            # Increment total attempts
            c2c_data['totalCallsReceived'] = c2c_data.get('totalCallsReceived', 0) + 1
            
            # Increment answered calls only if successful
            if call_type == "answered":
                c2c_data['totalCallsAnswered'] = c2c_data.get('totalCallsAnswered', 0) + 1
                
            c2c_data['lastUpdated'] = get_now_ist() # Use IST here
            
            # Update the document
            c2c_ref.set(c2c_data)
        else:
            # Create new document
            c2c_data = {
                'totalCallsReceived': 1,
                'totalCallsAnswered': 1 if call_type == "answered" else 0,
                'lastUpdated': get_now_ist() # Use IST here
            }
            c2c_ref.set(c2c_data)
            
        logger.info(f"Updated C2C stats for {call_type} call. Received: {c2c_data['totalCallsReceived']}, Answered: {c2c_data.get('totalCallsAnswered', 0)}")

    except Exception as e:
        logger.error(f"Failed to update C2C stats: {str(e)}")

# MODIFIED FUNCTIONS: Store Analysis & Volume Stats
# ====================================================================

# MODIFIED: Takes 'type_of_call' (INCOMING or C2C) as a new argument
def store_call_analysis(agent_data, call_data, analysis, tone_analysis, storage_url, language, type_of_call):
    """Store call analysis in Firestore including new tone analysis data."""
    if not db:
        logger.warning("Firestore not available - skipping storage")
        return None
    
    try:
        # Get tone score for overall score calculation
        tone_mark = tone_analysis.get('tone_mark', 0)
        
        # Generate the next call document name
        call_doc_name = get_next_call_document_name()
        timestamp = get_now_ist() # Use IST here
        
        # Calculate overall score using the updated logic
        overall_score = calculate_call_score(analysis, tone_mark)
        
        # Extract call type information
        call_type_data = analysis.get('call_type', {
            'primary_category': 'Unknown',
            'sub_category': 'Unknown',
            'confidence_score': 0.0,
            'secondary_categories': []
        })
        
        # Extract call sections from the analysis
        call_sections = analysis.get('call_sections', {
            "intro": {"summary": "", "present": False},
            "discovery": {"summary": "", "present": False},
            "demo": {"summary": "", "present": False},
            "objection": {"summary": "", "present": False},
            "closure": {"summary": "", "present": False}
        })
        
        # Extract talk ratio (customer:agent format)
        talk_ratio = analysis.get('talk_ratio', '50:50')
        
        # Prepare call document with enhanced data including call type and tone
        call_doc = {
            'callId': call_data.get('id', ''),
            'agentId': agent_data['id'],
            'agentName': agent_data.get('name', ''),
            'agentEmail': agent_data.get('email', ''),
            'timestamp': timestamp, # Stored as Firestore Timestamp
            'called': call_data.get('called', ''),
            'caller': call_data.get('caller', ''),
            'dialed': call_data.get('dialed', ''),
            'duration': int(call_data.get('duration', 0)),
            'recordingUrl': storage_url, # Firebase Storage URL
            'summary': analysis.get('summary', ''),
            
            # --- NEW FIELD: Call Type (INCOMING/C2C) ---
            'type_of_call': type_of_call,
            # -------------------------------------------
            
            # Call type classification
            'callType': {
                'primary': call_type_data.get('primary_category', 'Unknown'),
                'subCategory': call_type_data.get('sub_category', 'Unknown'),
                'confidence': call_type_data.get('confidence_score', 0.0),
                'secondary': call_type_data.get('secondary_categories', [])
            },
            
            # Tone Analysis Fields
            'toneAnalysis': {
                'agentMood': tone_analysis.get('agent_mood', 'Unknown'),
                'customerMood': tone_analysis.get('customer_mood', 'Unknown'),
                'toneMark': tone_mark,
                'reasoning': tone_analysis.get('reasoning', '')
            },
            
            'objections': analysis.get('objections', []),
            'competitors': analysis.get('competitors', []),
            'scores': analysis.get('scores', {}),
            'overallScore': overall_score,
            'coachingTips': analysis.get('coaching', []),
            'language': language,
            'fillerWords': analysis.get('filler_words_count', 0),
            'talkRatio': talk_ratio,
            'keyTopics': analysis.get('key_topics', []),
            'sentiment': analysis.get('sentiment', 'neutral'),
            'holdTime': analysis.get('hold_time', 0),
            'callAnalysis': analysis.get('call_analysis', {}),
            'callSections': call_sections,
            'metadata': {
                'circle': call_data.get('circle', ''),
                'network': call_data.get('network', ''),
                'ringtime': call_data.get('ringtime', ''),
                'starttime': call_data.get('starttime', ''),
                'endtime': call_data.get('endtime', ''),
                'processedAt': timestamp,
                # Store expiration time, replacing timezone info for Firebase compatibility but using the IST calculated time
                'audioExpiresAt': (timestamp + timedelta(days=30)).replace(tzinfo=None).isoformat() 
            }
        }
        
        # Store in calls collection with the generated document name
        db.collection('call_analysis').document(call_doc_name).set(call_doc)
        
        # Mark as processed to prevent duplicates
        add_to_processed_cache(call_data.get('id', ''))
        
        # Update agent's stats
        update_agent_stats(agent_data['id'], call_doc, call_doc_name)
        
        # Update INBOUND call volume statistics ONLY IF it was an incoming call
        if type_of_call == "INCOMING":
            update_call_volume_stats(call_doc, "answered")
        # NOTE: C2C stats are updated directly in the webhook function.
        
        logger.info(f"Call analysis stored for agent {agent_data['name']} as {call_doc_name} - Score: {overall_score} - Type: {type_of_call}")
        return call_doc_name
        
    except Exception as e:
        logger.error(f"Failed to store call analysis: {str(e)}")
        return None
        
def update_agent_stats(agent_id, call_data, call_doc_name):
    if not db:
        return
    
    try:
        agent_ref = db.collection('agents').document(agent_id)
        agent = agent_ref.get()
        
        if not agent.exists:
            return
        
        agent_data = agent.to_dict()
        agent_name = agent_data.get('name', 'Unknown')
        
        # Current date for daily tracking (using IST for string formatting)
        current_time_ist = get_now_ist()
        current_date = current_time_ist.strftime("%Y-%m-%d")
        current_week = current_time_ist.strftime("%Y-%U")
        current_month = current_time_ist.strftime("%Y-%m")
        
        # Update daily stats using agent ID as document ID
        daily_ref = db.collection('agent_stats').document(agent_id).collection('daily_stats').document(current_date)
        daily_doc = daily_ref.get()
        
        if daily_doc.exists:
            daily_data = daily_doc.to_dict()
            daily_data['callCount'] += 1
            daily_data['totalDuration'] += call_data['duration']
            daily_data['totalScore'] += call_data['overallScore']
            daily_data['avgScore'] = daily_data['totalScore'] / daily_data['callCount']
            daily_ref.set(daily_data)
        else:
            daily_ref.set({
                'callCount': 1,
                'totalDuration': call_data['duration'],
                'totalScore': call_data['overallScore'],
                'avgScore': call_data['overallScore'],
                'agentName': agent_name,
                'agentId': agent_id,
                'date': current_date
            })
        
        # Update weekly stats (unchanged logic)
        weekly_ref = db.collection('agent_stats').document(agent_id).collection('weekly_stats').document(current_week)
        weekly_doc = weekly_ref.get()
        
        if weekly_doc.exists:
            weekly_data = weekly_doc.to_dict()
            weekly_data['callCount'] += 1
            weekly_data['totalDuration'] += call_data['duration']
            weekly_data['totalScore'] += call_data['overallScore']
            weekly_data['avgScore'] = weekly_data['totalScore'] / weekly_data['callCount']
            weekly_ref.set(weekly_data)
        else:
            weekly_ref.set({
                'callCount': 1,
                'totalDuration': call_data['duration'],
                'totalScore': call_data['overallScore'],
                'avgScore': call_data['overallScore'],
                'agentName': agent_name,
                'agentId': agent_id,
                'week': current_week
            })
        
        # Update monthly stats (unchanged logic)
        monthly_ref = db.collection('agent_stats').document(agent_id).collection('monthly_stats').document(current_month)
        monthly_doc = monthly_ref.get()
        
        if monthly_doc.exists:
            monthly_data = monthly_doc.to_dict()
            monthly_data['callCount'] += 1
            monthly_data['totalDuration'] += call_data['duration']
            monthly_data['totalScore'] += call_data['overallScore']
            monthly_data['avgScore'] = monthly_data['totalScore'] / monthly_data['callCount']
            monthly_ref.set(monthly_data)
        else:
            monthly_ref.set({
                'callCount': 1,
                'totalDuration': call_data['duration'],
                'totalScore': call_data['overallScore'],
                'avgScore': call_data['overallScore'],
                'agentName': agent_name,
                'agentId': agent_id,
                'month': current_month
            })
        
        # Update agent summary stats in main document
        current_total_calls = agent_data.get('stats', {}).get('totalCalls', 0) + 1
        
        # Calculate weighted overall score (recent calls weighted more heavily)
        weight = min(0.7, 0.3 + (current_total_calls * 0.01)) # Dynamic weighting
        current_score = agent_data.get('stats', {}).get('overallScore', 0)
        new_overall = (current_score * (1 - weight) + call_data['overallScore'] * weight)
        
        # Update agent document with summary stats
        agent_ref.update({
            'stats.totalCalls': current_total_calls,
            'stats.overallScore': new_overall,
            'stats.lastCallDate': current_time_ist, # Use IST here
            'updatedAt': current_time_ist # Use IST here
        })
        
        # Also store a reference to this call in the agent's call history
        call_ref = db.collection('agent_stats').document(agent_id).collection('call_history').document(call_data['callId'])
        call_ref.set({
            'callId': call_data['callId'],
            'callDocName': call_doc_name,
            'timestamp': call_data.get('timestamp', current_time_ist), # Use IST here
            'score': call_data['overallScore'],
            'duration': call_data['duration'],
            'agentName': agent_name,
            'agentId': agent_id
        })
        
    except Exception as e:
        logger.error(f"Failed to update agent stats: {str(e)}")

def process_call_recording(recording_url, call_data, call_type):
    """Process call recording - download, transcribe, analyze, and upload to Firebase Storage"""
    # Generate unique call ID
    call_id = generate_call_id(call_data)
    call_data['id'] = call_id

    # Check if call already processed
    if is_call_processed(call_id):
        logger.warning(f"Call {call_id} already processed - skipping")
        return None, "Call already processed"
    
    # Mark as processed early to avoid race conditions
    add_to_processed_cache(call_id)

    # --- 1. Download audio ---
    audio_path, error = download_audio(recording_url)
    if error:
        # NOTE: Remove from cache if download fails early
        if call_id in processed_calls: processed_calls.pop(call_id)
        return None, error

    # Temporary files list for cleanup
    temp_files_to_clean = [audio_path]
    agent_audio_path = None
    cust_audio_path = None
    tone_analysis = None
    analysis = None

    try:
        # --- 2. Tone Analysis Prep: Split audio and extract features ---
        agent_audio_path, cust_audio_path, _ = split_audio_channels(audio_path)
        
        # Add new temporary files to cleanup list
        if agent_audio_path and agent_audio_path != audio_path:
            temp_files_to_clean.append(agent_audio_path)
        if cust_audio_path and cust_audio_path != audio_path:
            temp_files_to_clean.append(cust_audio_path)
        
        agent_features = extract_features(agent_audio_path)
        customer_features = extract_features(cust_audio_path)
        
        if not agent_features or not customer_features:
            logger.warning("Acoustic feature extraction failed. Proceeding with transcription only.")
            tone_mark = 0 # Default score
            tone_analysis = {'tone_mark': 0, 'agent_mood': 'Unknown', 'customer_mood': 'Unknown'} # Initialize tone_analysis
        else:
            # --- 3. Tone Analysis (LLM) ---
            tone_analysis, tone_error = analyze_tone_with_azure(agent_features, customer_features)
            if tone_error:
                logger.warning(f"Tone analysis failed: {tone_error}. Defaulting score.")
                tone_analysis = {'tone_mark': 0, 'agent_mood': 'Unknown', 'customer_mood': 'Unknown'}

            tone_mark = tone_analysis.get('tone_mark', 0)
            
        # --- 4. Transcription ---
        transcription, error = transcribe_audio(audio_path)
        if error:
            return None, error

        transcript_text = transcription.text if transcription else ""
        
        # Check if transcription is valid
        if not transcript_text or len(transcript_text.strip()) < 10:
            return None, "Transcription failed: Empty or too short transcript"

        # Detect language from Whisper
        language = detect_language(transcription)

        # --- 5. Content Analysis (LLM) ---
        analysis, error = analyze_call_with_azure_openai(transcript_text, language, tone_mark)
        if error:
            return None, error
        
        # Validate analysis response
        if not analysis:
            return None, "Analysis failed: No content analysis data returned"
        
        # Update intro score based on intro_check
        if analysis.get('intro_check') == 'Yes':
            analysis['scores']['intro'] = 10
            analysis['call_analysis']['company_intro_early'] = True
        else:
            analysis['scores']['intro'] = 0
            analysis['call_analysis']['company_intro_early'] = False

        # --- MODIFIED: Get agent based on call type ---
        agent = None
        if call_type == "INCOMING":
            # For incoming calls, agent is identified by dialed number
            agent = get_agent_by_phone_number(call_data.get('dialed', ''), "INCOMING")
        else:  # C2C calls
            # For C2C calls, agent is identified by caller number (the agent making the call)
            agent = get_agent_by_phone_number(call_data.get('caller', ''), "C2C")
        
        if not agent:
            logger.warning(f"Agent not found for {call_type} call - dialed: {call_data.get('dialed', '')}, caller: {call_data.get('caller', '')}")
            return None, "Agent not found"

        # --- 6. Upload to Firebase Storage ---
        storage_url, error = upload_to_firebase_storage(audio_path, agent.get('email', ''), call_id)
        if error:
            logger.warning(f"Firebase Storage upload failed: {error}. Storing without audio link.")
            storage_url = None

        # --- 7. Store analysis ---
        call_doc_name = store_call_analysis(agent, call_data, analysis, tone_analysis, storage_url, language, call_type) 
        if not call_doc_name:
            return None, "Failed to store analysis"

        return {
            'agent': agent,
            'analysis': analysis,
            'callDocName': call_doc_name,
            'language': language,
            'storageUrl': storage_url,
            'toneMark': tone_mark
        }, None

    except Exception as e:
        logger.error(f"Unexpected error in process_call_recording: {str(e)}")
        # Remove from cache if processing failed due to unexpected error
        if call_id in processed_calls:
            processed_calls.pop(call_id) 
        return None, f"Processing failed: {str(e)}"
    finally:
        # Clean up temporary files
        for f in temp_files_to_clean:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except Exception as e:
                    logger.error(f"Failed to delete temp file {f}: {str(e)}")
                    
def update_call_volume_stats(call_data, call_type="answered"):
    if not db:
        return
    
    try:
        current_time = get_now_ist() # Use IST here
        current_date = current_time.strftime("%Y-%m-%d")
        current_week = current_time.strftime("%Y-%U")
        current_month = current_time.strftime("%Y-%m")
        current_hour = current_time.strftime("%H:00")
        
        # Determine if this is an off-hours call (before 10am or after 7pm)
        hour = current_time.hour
        is_off_hours = hour < 10 or hour >= 19
        
        # Get or create call volume stats document
        volume_ref = db.collection('call_volume_stats').document('overall')
        volume_doc = volume_ref.get()
        
        if volume_doc.exists:
            volume_data = volume_doc.to_dict()
        else:
            # Initialize with default values
            volume_data = {
                'totalCallsReceived': 0,
                'totalCallsAnswered': 0,
                'totalOffHoursCalls': 0,  # NEW: Track off-hours calls
                'dailyStats': {},
                'weeklyStats': {},
                'monthlyStats': {},
                'hourlyDistribution': {},
                'offHoursDistribution': {  # NEW: Track off-hours by time periods
                    'early_morning': 0,    # 12am - 10am
                    'evening_night': 0      # 7pm - 12am
                },
                'peakHours': {
                    'daily': {},
                    'weekly': {},
                    'monthly': {}
                },
                'lastUpdated': get_now_ist() # Use IST here
            }
        
        # Update overall stats
        volume_data['totalCallsReceived'] += 1
        
        # FIX: Update off-hours calls count for ALL calls (answered + unanswered)
        if is_off_hours:
            volume_data['totalOffHoursCalls'] += 1
            
            # Update off-hours distribution for ALL off-hours calls
            if hour < 10:
                volume_data['offHoursDistribution']['early_morning'] += 1
            else:  # hour >= 19
                volume_data['offHoursDistribution']['evening_night'] += 1
        
        if call_type == "answered":
            volume_data['totalCallsAnswered'] += 1
        
        # Update daily stats (unchanged logic, but using IST strings)
        if current_date not in volume_data['dailyStats']:
            volume_data['dailyStats'][current_date] = {
                'callsReceived': 0,
                'callsAnswered': 0,
                'offHoursCalls': 0,  # NEW: Track off-hours per day
                'hourlyBreakdown': {}
            }
        
        daily_stats = volume_data['dailyStats'][current_date]
        daily_stats['callsReceived'] += 1
        
        # FIX: Update daily off-hours calls for ALL calls
        if is_off_hours:
            daily_stats['offHoursCalls'] += 1
        
        if call_type == "answered":
            daily_stats['callsAnswered'] += 1
        
        # Update hourly breakdown for the day
        if current_hour not in daily_stats['hourlyBreakdown']:
            daily_stats['hourlyBreakdown'][current_hour] = {
                'received': 0,
                'answered': 0,
                'offHours': 0  # NEW: Track off-hours per hour
            }
        
        daily_stats['hourlyBreakdown'][current_hour]['received'] += 1
        
        # FIX: Update hourly off-hours for ALL calls
        if is_off_hours:
            daily_stats['hourlyBreakdown'][current_hour]['offHours'] += 1
        
        if call_type == "answered":
            daily_stats['hourlyBreakdown'][current_hour]['answered'] += 1
        
        # Update weekly stats (unchanged logic, but using IST strings)
        if current_week not in volume_data['weeklyStats']:
            volume_data['weeklyStats'][current_week] = {
                'callsReceived': 0,
                'callsAnswered': 0,
                'offHoursCalls': 0,  # NEW: Track off-hours per week
                'dailyBreakdown': {}
            }
        
        weekly_stats = volume_data['weeklyStats'][current_week]
        weekly_stats['callsReceived'] += 1
        
        # FIX: Update weekly off-hours calls for ALL calls
        if is_off_hours:
            weekly_stats['offHoursCalls'] += 1
        
        if call_type == "answered":
            weekly_stats['callsAnswered'] += 1
        
        # Update daily breakdown for the week
        if current_date not in weekly_stats['dailyBreakdown']:
            weekly_stats['dailyBreakdown'][current_date] = {
                'received': 0,
                'answered': 0,
                'offHours': 0  # NEW: Track off-hours per day in week
            }
        
        weekly_stats['dailyBreakdown'][current_date]['received'] += 1
        
        # FIX: Update daily off-hours in weekly breakdown for ALL calls
        if is_off_hours:
            weekly_stats['dailyBreakdown'][current_date]['offHours'] += 1
        
        if call_type == "answered":
            weekly_stats['dailyBreakdown'][current_date]['answered'] += 1
        
        # Update monthly stats (unchanged logic, but using IST strings)
        if current_month not in volume_data['monthlyStats']:
            volume_data['monthlyStats'][current_month] = {
                'callsReceived': 0,
                'callsAnswered': 0,
                'offHoursCalls': 0,  # NEW: Track off-hours per month
                'weeklyBreakdown': {}
            }
        
        monthly_stats = volume_data['monthlyStats'][current_month]
        monthly_stats['callsReceived'] += 1
        
        # FIX: Update monthly off-hours calls for ALL calls
        if is_off_hours:
            monthly_stats['offHoursCalls'] += 1
        
        if call_type == "answered":
            monthly_stats['callsAnswered'] += 1
        
        # Update weekly breakdown for the month
        if current_week not in monthly_stats['weeklyBreakdown']:
            monthly_stats['weeklyBreakdown'][current_week] = {
                'received': 0,
                'answered': 0,
                'offHours': 0  # NEW: Track off-hours per week in month
            }
        
        monthly_stats['weeklyBreakdown'][current_week]['received'] += 1
        
        # FIX: Update weekly off-hours in monthly breakdown for ALL calls
        if is_off_hours:
            monthly_stats['weeklyBreakdown'][current_week]['offHours'] += 1
        
        if call_type == "answered":
            monthly_stats['weeklyBreakdown'][current_week]['answered'] += 1
        
        # Update hourly distribution (across all time)
        if current_hour not in volume_data['hourlyDistribution']:
            volume_data['hourlyDistribution'][current_hour] = {
                'received': 0,
                'answered': 0,
                'offHours': 0  # NEW: Track off-hours in hourly distribution
            }
        
        volume_data['hourlyDistribution'][current_hour]['received'] += 1
        
        # FIX: Update off-hours in hourly distribution for ALL calls
        if is_off_hours:
            volume_data['hourlyDistribution'][current_hour]['offHours'] += 1
        
        if call_type == "answered":
            volume_data['hourlyDistribution'][current_hour]['answered'] += 1
        
        # Update peak hours (this will be calculated on-demand when requested)
        # For now, we'll just mark that we need to recalculate
        volume_data['peakHours']['needsRecalculation'] = True
        
        # Update last updated timestamp
        volume_data['lastUpdated'] = get_now_ist() # Use IST here
        
        # Save back to Firestore
        volume_ref.set(volume_data)
        
        logger.info(f"Updated call volume stats for {call_type} call - Off-hours: {is_off_hours}")
        
    except Exception as e:
        logger.error(f"Failed to update call volume stats: {str(e)}")

def calculate_peak_hours(volume_data):
    try:
        peak_hours = {
            'daily': {},
            'weekly': {},
            'monthly': {}
        }
        
        # Calculate daily peak hours (for the last 7 days)
        recent_days = list(volume_data.get('dailyStats', {}).items())[-7:]
        for date, day_data in recent_days:
            hourly_data = day_data.get('hourlyBreakdown', {})
            if hourly_data:
                # Find hour with most calls received
                peak_hour = max(hourly_data.items(), 
                                key=lambda x: x[1].get('received', 0))
                peak_hours['daily'][date] = {
                    'peakHour': peak_hour[0],
                    'callsReceived': peak_hour[1].get('received', 0),
                    'callsAnswered': peak_hour[1].get('answered', 0)
                }
        
        # Calculate weekly peak days (for the last 4 weeks)
        recent_weeks = list(volume_data.get('weeklyStats', {}).items())[-4:]
        for week, week_data in recent_weeks:
            daily_data = week_data.get('dailyBreakdown', {})
            if daily_data:
                # Find day with most calls received
                peak_day = max(daily_data.items(), 
                              key=lambda x: x[1].get('received', 0))
                peak_hours['weekly'][week] = {
                    'peakDay': peak_day[0],
                    'callsReceived': peak_day[1].get('received', 0),
                    'callsAnswered': peak_day[1].get('answered', 0)
                }
        
        # Calculate monthly peak weeks (for the last 3 months)
        recent_months = list(volume_data.get('monthlyStats', {}).items())[-3:]
        for month, month_data in recent_months:
            weekly_data = month_data.get('weeklyBreakdown', {})
            if weekly_data:
                # Find week with most calls received
                peak_week = max(weekly_data.items(), 
                                key=lambda x: x[1].get('received', 0))
                peak_hours['monthly'][month] = {
                    'peakWeek': peak_week[0],
                    'callsReceived': peak_week[1].get('received', 0),
                    'callsAnswered': peak_week[1].get('answered', 0)
                }
        
        return peak_hours
        
    except Exception as e:
        logger.error(f"Failed to calculate peak hours: {str(e)}")
        return {
            'daily': {},
            'weekly': {},
            'monthly': {}
        }

def get_call_volume_stats():
    if not db:
        return None
    
    try:
        volume_ref = db.collection('call_volume_stats').document('overall')
        volume_doc = volume_ref.get()
        
        if not volume_doc.exists:
            return None
        
        volume_data = volume_doc.to_dict()
        
        # Ensure off-hours fields exist for backward compatibility
        if 'totalOffHoursCalls' not in volume_data:
            volume_data['totalOffHoursCalls'] = 0
        
        if 'offHoursDistribution' not in volume_data:
            volume_data['offHoursDistribution'] = {
                'early_morning': 0,
                'evening_night': 0
            }
        
        # Calculate peak hours if needed
        if volume_data.get('peakHours', {}).get('needsRecalculation', False):
            peak_hours = calculate_peak_hours(volume_data)
            volume_data['peakHours'] = peak_hours
            
            # Remove the recalculation flag and update
            if 'needsRecalculation' in volume_data['peakHours']:
                del volume_data['peakHours']['needsRecalculation']
            
            volume_ref.set(volume_data)
        
        return volume_data
        
    except Exception as e:
        logger.error(f"Failed to get call volume stats: {str(e)}")
        return None

def calculate_volume_trend(data_points):
    if not data_points or len(data_points) < 2:
        return "stable"
    
    # Extract call received counts
    calls = [point.get('callsReceived', 0) for point in data_points]
    
    # Simple trend calculation
    if len(calls) >= 2:
        recent_avg = statistics.mean(calls[-2:])
        previous_avg = statistics.mean(calls[:-2]) if len(calls) > 2 else calls[0]
        
        if recent_avg > previous_avg * 1.1: # 10% increase
            return "up"
        elif recent_avg < previous_avg * 0.9: # 10% decrease
            return "down"
    
    return "stable"

def generate_peak_recommendations(peak_hours):
    recommendations = []
    
    # Analyze daily peak patterns
    daily_peaks = peak_hours.get('daily', {})
    if daily_peaks:
        common_hours = defaultdict(int)
        for date, data in daily_peaks.items():
            common_hours[data.get('peakHour', '')] += 1
        
        if common_hours:
            most_common_hour = max(common_hours.items(), key=lambda x: x[1])
            recommendations.append(f"Most common peak hour: {most_common_hour[0]} ({most_common_hour[1]} days)")
    
    # Analyze weekly patterns
    weekly_peaks = peak_hours.get('weekly', {})
    if weekly_peaks:
        week_days = defaultdict(int)
        for week, data in weekly_peaks.items():
            peak_day = data.get('peakDay', '')
            if peak_day:
                # Extract day of week from date (YYYY-MM-DD)
                try:
                    day_of_week = datetime.strptime(peak_day, "%Y-%m-%d").strftime("%A")
                    week_days[day_of_week] += 1
                except:
                    pass
        
        if week_days:
            busiest_day = max(week_days.items(), key=lambda x: x[1])
            recommendations.append(f"Busiest day of week: {busiest_day[0]} ({busiest_day[1]} weeks)")
    
    # Analyze monthly patterns
    monthly_peaks = peak_hours.get('monthly', {})
    if monthly_peaks:
        month_weeks = defaultdict(int)
        for month, data in monthly_peaks.items():
            peak_week = data.get('peakWeek', '')
            if peak_week:
                month_weeks[peak_week] += 1
        
        if month_weeks:
            busiest_week = max(month_weeks.items(), key=lambda x: x[1])
            recommendations.append(f"Busiest week of month: Week {busiest_week[0].split('-')[-1]} ({busiest_week[1]} months)")
    
    # Add general staffing recommendations
    volume_data = get_call_volume_stats()
    if volume_data:
        hourly_dist = volume_data.get('hourlyDistribution', {})
        if hourly_dist:
            # Find the 3 busiest hours
            busiest_hours = sorted(
                [(hour, data.get('received', 0)) for hour, data in hourly_dist.items()],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            if busiest_hours:
                rec_text = "Ensure adequate staffing during peak hours: "
                rec_text += ", ".join([f"{hour} ({calls} calls)" for hour, calls in busiest_hours])
                recommendations.append(rec_text)
    
    return recommendations if recommendations else ["Insufficient data for specific recommendations"]


# -------------------------------------------------
# MODIFIED: Unified Webhook Route
# -------------------------------------------------
@app.route("/webhook", methods=["GET", "POST"])
def webhook():
    """
    Unified Webhook endpoint for Kaleyra calls (Inbound and Click2Call).
    Differentiates source using the 'call_source=C2C' query parameter.
    """
    try:
        # Collect request data
        data = {}
        if request.method == "GET":
            data.update(request.args.to_dict())
        else: # POST
            if request.args:
                data.update(request.args.to_dict())
            if request.form:
                data.update(request.form.to_dict())
            if request.is_json:
                try:
                    data.update(request.get_json() or {})
                except Exception as json_error:
                    logger.warning(f"Failed to parse JSON data: {json_error}")

        # Decode URL-encoded values
        data = decode_url_encoded_values(data)

        # === CORE LOGIC: Determine Call Source ===
        # Default is INCOMING (for Inbound Settings URL), C2C is passed via the C2C Callback URL
        call_source = data.get("call_source", "INCOMING").upper() 
        
        log_data = {k: v for k, v in data.items() if 'password' not in k.lower() and 'token' not in k.lower()}
        logger.info(f"Webhook received ({call_source}): {log_data}")

        # Check for answered call
        dial_status = data.get("dialstatus", "").upper()
        
        if dial_status == "ANSWER":
            
            if not data.get('recording'):
                logger.warning(f"No recording URL provided for answered {call_source} call")
                return jsonify({"status": "skipped", "message": "No recording URL"}), 200
            
            # --- Answered Call Processing (Same for both INCOMING and C2C) ---
            # NOTE: Pass the determined call_source to the processing function
            result, error = process_call_recording(data.get('recording', ''), data, call_source)
            
            if error:
                if "already processed" in error.lower():
                    return jsonify({"status": "skipped", "message": error}), 200
                logger.error(f"{call_source} call processing failed: {error}")
                return jsonify({"status": "error", "message": error}), 500
            
            # --- Source-Specific Stat Update (Answered) ---
            if call_source == "C2C":
                # Update C2C volume stats only (Inbound stats handled in store_call_analysis)
                update_c2c_stats(data, call_type="answered")
            
            logger.info(f"{call_source} call processed successfully - Doc: {result['callDocName']}")
            return jsonify({
                "status": "success", 
                "message": "Call processed and stored",
                "source": call_source,
                "callDocName": result['callDocName'],
                "toneMark": result.get('toneMark', 'N/A')
            }), 200
            
        else:
            # --- Source-Specific Stat Update for Non-Answered ---
            call_data_for_stats = {
                'id': generate_call_id(data),
                'duration': int(data.get('duration', 0)),
                'overallScore': 0
            }

            if call_source == "C2C":
                # Only track total C2C attempts (answered + unanswered)
                update_c2c_stats(data, call_type="unanswered")
            else: # INCOMING
                # Track unanswered incoming calls for volume stats
                update_call_volume_stats(call_data_for_stats, "unanswered")
            
            logger.info(f"Skipping non-answered {call_source} call with status: {dial_status}")
            return jsonify({"status": "skipped", "message": f"Not an ANSWERed call (source: {call_source}, status: {dial_status})"}), 200

    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500
        
@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint."""
    services = {
        "flask": "healthy",
        "groq": "healthy" if groq_client else "unavailable",
        "azure_openai": "healthy" if azure_openai_client else "unavailable",
        "firebase": "healthy" if db else "unavailable",
        "firebase_storage": "healthy" if bucket else "unavailable"
    }
    
    return jsonify({
        "status": "healthy", 
        "timestamp": get_now_ist().isoformat(), # Use IST here
        "services": services,
        "processedCalls": len(processed_calls)
    }), 200

@app.route("/agent/<email>", methods=["GET"])
def get_agent_stats(email):
    """Get agent statistics and call history by email"""
    if not db:
        return jsonify({"error": "Database not available"}), 500
    
    try:
        # Query agents collection by email
        agents_ref = db.collection('agents')
        query = agents_ref.where(filter=firestore.FieldFilter('email', '==', email)).limit(1)
        docs = query.stream()
        
        agent = None
        for doc in docs:
            agent = {**doc.to_dict(), 'id': doc.id}
        
        if not agent:
            return jsonify({"error": "Agent not found"}), 404
        
        # Get recent calls for this agent
        calls_ref = db.collection('call_analysis')
        query = calls_ref.where(filter=firestore.FieldFilter('agentEmail', '==', email)).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(10)
        calls = [doc.to_dict() for doc in query.stream()]
        
        # Get daily stats for the last 7 days
        daily_stats = {}
        daily_docs = db.collection('agent_stats').document(agent['id']).collection('daily_stats').order_by('date', direction=firestore.Query.DESCENDING).limit(7).stream()
        
        for doc in daily_docs:
            daily_stats[doc.id] = doc.to_dict()
        
        # Calculate ranking based on overall score
        all_agents = db.collection('agents').stream()
        agent_scores = []
        for a in all_agents:
            a_data = a.to_dict()
            if 'stats' in a_data and 'overallScore' in a_data['stats']:
                agent_scores.append({
                    'id': a.id,
                    'name': a_data.get('name', 'Unknown'),
                    'email': a_data.get('email', ''),
                    'score': a_data['stats']['overallScore']
                })
        
        # Sort by score descending
        agent_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Find current agent's rank
        rank = next((i+1 for i, a in enumerate(agent_scores) if a['email'] == email), None)
        
        return jsonify({
            "agent": agent,
            "recentCalls": calls,
            "dailyStats": daily_stats,
            "ranking": {
                "position": rank,
                "totalAgents": len(agent_scores),
                "topPerformers": agent_scores[:3] if len(agent_scores) >= 3 else agent_scores
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching agent stats: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/stats/volume", methods=["GET"])
def get_volume_stats():
    """Get call volume statistics with peak hours"""
    if not db:
        return jsonify({"error": "Database not available"}), 500
    
    try:
        volume_data = get_call_volume_stats()
        c2c_ref = db.collection('c2c_stats').document('overall')
        c2c_doc = c2c_ref.get()
        c2c_data = c2c_doc.to_dict() if c2c_doc.exists else {'totalCallsReceived': 0, 'totalCallsAnswered': 0}
        
        if not volume_data:
            volume_data = {} # Initialize empty to prevent error
            
        # Calculate some additional metrics for INCOMING
        total_received = volume_data.get('totalCallsReceived', 0)
        total_answered = volume_data.get('totalCallsAnswered', 0)
        answer_ratio = round((total_answered / total_received * 100), 2) if total_received > 0 else 0
        
        # Get recent data for trends
        daily_stats = volume_data.get('dailyStats', {})
        weekly_stats = volume_data.get('weeklyStats', {})
        monthly_stats = volume_data.get('monthlyStats', {})
        
        # Calculate trends
        daily_trend = calculate_volume_trend(list(daily_stats.values())[-7:]) if daily_stats else "stable"
        weekly_trend = calculate_volume_trend(list(weekly_stats.values())[-4:]) if weekly_stats else "stable"
        monthly_trend = calculate_volume_trend(list(monthly_stats.values())[-3:]) if monthly_stats else "stable"
        
        return jsonify({
            "overview": {
                "totalIncomingReceived": total_received,
                "totalIncomingAnswered": total_answered,
                "incomingAnswerRatio": answer_ratio,
                "totalC2CAttempts": c2c_data.get('totalCallsReceived', 0),
                "totalC2CAnswered": c2c_data.get('totalCallsAnswered', 0),
            },
            "timeBased": {
                "daily": daily_stats,
                "weekly": weekly_stats,
                "monthly": monthly_stats
            },
            "hourlyDistribution": volume_data.get('hourlyDistribution', {}),
            "peakHours": volume_data.get('peakHours', {}),
            "trends": {
                "daily": daily_trend,
                "weekly": weekly_trend,
                "monthly": monthly_trend
            },
            "lastUpdated": volume_data.get('lastUpdated', '')
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching volume stats: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/stats/peak-hours", methods=["GET"])
def get_peak_hours():
    """Get current peak hours analysis"""
    if not db:
        return jsonify({"error": "Database not available"}), 500
    
    try:
        volume_data = get_call_volume_stats()
        
        if not volume_data:
            return jsonify({"error": "Volume stats not found"}), 404
        
        peak_hours = volume_data.get('peakHours', {})
        
        # Get current recommendations
        recommendations = generate_peak_recommendations(peak_hours)
        
        return jsonify({
            "peakHours": peak_hours,
            "recommendations": recommendations,
            "lastUpdated": volume_data.get('lastUpdated', '')
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching peak hours: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.route("/", methods=["GET"])
def root():
    """Root endpoint with service information"""
    return jsonify({
        "service": "Prashanti Customer Support Call Analysis API",
        "status": "running", 
        "timestamp": get_now_ist().isoformat(),
        "endpoints": {
            "health": "/health",
            "webhook": "/webhook (POST)",
            "agent_stats": "/agent/<email> (GET)",
            "volume_stats": "/stats/volume (GET)", 
            "peak_hours": "/stats/peak-hours (GET)"
        },
        "version": "1.0"
    }), 200

# -------------------------------------------------
# Main Runner  
# -------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    # Production-ready setting: No debugger, no reloader
    app.run(host="0.0.0.0", port=port, debug=False)
