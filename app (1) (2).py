
# ============================================================
# MULTIMODAL SENTIMENT & EMOTION ANALYZER
# ============================================================

from pathlib import Path
from collections import deque
import threading
import time
import re
import os
import shutil
import textwrap

import av
import cv2
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_webrtc import webrtc_streamer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Multimodal Sentiment & Emotion Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# GLOBAL COLOURS
# ============================================================

POSITIVE_COLOR = "#22C55E"
NEUTRAL_COLOR = "#8B5CF6"
NEGATIVE_COLOR = "#EF4444"

HAPPY_COLOR = "#22C55E"
SAD_COLOR = "#3B82F6"
ANGRY_COLOR = "#EF4444"
FEAR_COLOR = "#A855F7"
SURPRISE_COLOR = "#F59E0B"
DISGUST_COLOR = "#84CC16"
FACE_NEUTRAL_COLOR = "#64748B"

TEXT_COLORS = {
    "positive": POSITIVE_COLOR,
    "neutral": NEUTRAL_COLOR,
    "negative": NEGATIVE_COLOR
}

FACE_COLORS = {
    "happy": HAPPY_COLOR,
    "sad": SAD_COLOR,
    "angry": ANGRY_COLOR,
    "fear": FEAR_COLOR,
    "surprise": SURPRISE_COLOR,
    "disgust": DISGUST_COLOR,
    "neutral": FACE_NEUTRAL_COLOR
}

TEXT_EMOJIS = {
    "positive": "😊",
    "neutral": "😐",
    "negative": "😞"
}

FACE_EMOJIS = {
    "happy": "😄",
    "sad": "😢",
    "angry": "😠",
    "fear": "😨",
    "surprise": "😲",
    "disgust": "🤢",
    "neutral": "😐"
}


# ============================================================
# CUSTOM UI
# ============================================================

st.html(
    textwrap.dedent("""
    <style>

    /* ---------- Main background ---------- */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(99,102,241,0.14),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(236,72,153,0.12),
                transparent 25%
            ),
            radial-gradient(
                circle at 50% 90%,
                rgba(34,197,94,0.10),
                transparent 30%
            );
    }

    .block-container {
        max-width: 1250px;
        padding-top: 4.2rem !important;
        padding-bottom: 3rem;
    }


    /* ---------- Header ---------- */

    .hero {
        margin-top: 12px;
        padding: 34px 30px 28px 30px;
        border-radius: 25px;
        margin-bottom: 20px;

        background:
            linear-gradient(
                125deg,
                #312E81,
                #7C3AED,
                #DB2777
            );

        box-shadow:
            0 18px 45px
            rgba(79,70,229,0.25);

        color: white;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 850;
        margin: 0;
        line-height: 1.2;
    }

    .hero-subtitle {
        font-size: 1.02rem;
        opacity: 0.92;
        margin-top: 10px;
    }

    .badge {
        display: inline-block;
        padding: 6px 11px;
        margin-right: 7px;
        margin-top: 13px;
        border-radius: 999px;
        background: rgba(255,255,255,0.16);
        border: 1px solid rgba(255,255,255,0.24);
        font-size: 0.82rem;
        font-weight: 650;
    }


    /* ---------- Metrics ---------- */

    [data-testid="stMetric"] {
        background:
            linear-gradient(
                135deg,
                rgba(255,255,255,0.94),
                rgba(248,250,252,0.90)
            );

        border: 1px solid
            rgba(148,163,184,0.24);

        padding: 17px 18px;
        border-radius: 18px;

        box-shadow:
            0 8px 24px
            rgba(15,23,42,0.07);
    }

    [data-testid="stMetricValue"] {
        font-weight: 800;
    }


    /* ---------- Tabs ---------- */

    button[data-baseweb="tab"] {
        border-radius: 13px;
        padding-left: 18px;
        padding-right: 18px;
        font-weight: 700;
    }


    /* ---------- Buttons ---------- */

    .stButton > button {
        border-radius: 13px;
        font-weight: 700;
        min-height: 45px;
        transition: all 0.20s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow:
            0 8px 20px
            rgba(99,102,241,0.18);
    }


    /* ---------- Result cards ---------- */

    .result-card {
        padding: 20px;
        border-radius: 20px;
        border: 1px solid
            rgba(148,163,184,0.25);

        background:
            rgba(255,255,255,0.82);

        box-shadow:
            0 10px 28px
            rgba(15,23,42,0.07);

        margin-bottom: 10px;
    }

    .status-box {
        text-align: center;
        padding: 22px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 1.25rem;
        margin-top: 8px;
    }

    .small-note {
        opacity: 0.74;
        font-size: 0.86rem;
    }


    /* ---------- Dividers ---------- */

    hr {
        border: none;
        height: 1px;
        background:
            linear-gradient(
                90deg,
                transparent,
                rgba(99,102,241,0.4),
                transparent
            );
    }

    </style>
    """)
)


# ============================================================
# PATHS
# ============================================================

BASE_FOLDER = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_FOLDER
    / "sentiment_model_3class.pkl"
)

TFIDF_PATH = (
    BASE_FOLDER
    / "tfidf_vectorizer_3class.pkl"
)


# ============================================================
# LOAD TEXT MODEL
# ============================================================

@st.cache_resource
def load_text_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "sentiment_model_3class.pkl was not found."
        )

    if not TFIDF_PATH.exists():
        raise FileNotFoundError(
            "tfidf_vectorizer_3class.pkl was not found."
        )

    model = joblib.load(
        MODEL_PATH
    )

    tfidf = joblib.load(
        TFIDF_PATH
    )

    return model, tfidf


text_model, tfidf = load_text_model()




# ============================================================
# LAZY DEEPFACE LOADER
# ============================================================

_deepface_class = None
deepface_import_lock = threading.Lock()

def get_deepface():

    global _deepface_class

    if _deepface_class is None:

        with deepface_import_lock:

            if _deepface_class is None:

                # --------------------------------------------
                # Prepare bundled DeepFace emotion weights
                # --------------------------------------------

                bundled_weight = (
                    BASE_FOLDER
                    / "weights"
                    / "facial_expression_model_weights.h5"
                )

                deepface_weights = (
                    Path.home()
                    / ".deepface"
                    / "weights"
                )

                deepface_weights.mkdir(
                    parents=True,
                    exist_ok=True
                )

                destination_weight = (
                    deepface_weights
                    / "facial_expression_model_weights.h5"
                )

                if (
                    bundled_weight.exists()
                    and
                    not destination_weight.exists()
                ):

                    shutil.copy2(
                        bundled_weight,
                        destination_weight
                    )

                # Load DeepFace only when camera analysis
                # actually reaches its first analysis frame.
                from deepface import DeepFace as _DeepFace

                _deepface_class = _DeepFace

    return _deepface_class


# DeepFace/TensorFlow inference is protected by a lock.
deepface_lock = threading.Lock()


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    text = re.sub(
        r"http\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# TEXT PREDICTION
# ============================================================

def predict_text(text):

    cleaned = clean_text(
        text
    )

    vector = tfidf.transform(
        [cleaned]
    )

    prediction = str(
        text_model.predict(
            vector
        )[0]
    ).lower()

    probabilities = (
        text_model.predict_proba(
            vector
        )[0]
    )

    probability_map = {

        str(label).lower():
        float(probability)

        for label, probability
        in zip(
            text_model.classes_,
            probabilities
        )
    }

    confidence = float(
        max(probabilities)
    )

    return (
        prediction,
        confidence,
        probability_map
    )


# ============================================================
# CLAUSE SEGMENTATION
# ============================================================

CONTRAST_WORDS = [

    "but",
    "however",
    "although",
    "though",
    "yet",
    "whereas",
    "while",
    "nevertheless"
]


CONTRAST_REGEX = (
    "|".join(
        CONTRAST_WORDS
    )
)


def split_clauses(text):

    text = re.sub(
        r"\s+",
        " ",
        str(text)
    ).strip()

    if not text:
        return []


    # Put a marker before contrast words.
    marked = re.sub(

        rf"\s+({CONTRAST_REGEX})\s+",

        r" ||| \1 ",

        text,

        flags=re.I
    )


    # Also split on strong punctuation.
    marked = re.sub(
        r"[;.!?]+",
        " ||| ",
        marked
    )


    pieces = marked.split(
        "|||"
    )


    clauses = []

    for item in pieces:

        item = item.strip(
            " ,;:"
        )

        if len(
            item.split()
        ) >= 2:

            clauses.append(
                item
            )


    return clauses


def remove_connector(clause):

    cleaned = re.sub(

        rf"^({CONTRAST_REGEX})\b[:,]?\s*",

        "",

        clause,

        flags=re.I
    ).strip()

    return (
        cleaned
        if cleaned
        else clause
    )


def begins_with_contrast(
    clause
):

    return bool(
        re.search(
            rf"^({CONTRAST_REGEX})\b",
            clause.strip(),
            flags=re.I
        )
    )


# ============================================================
# TEXT CONFLICT SCORE
# ============================================================

SENTIMENT_VALUE = {

    "negative": -1,

    "neutral": 0,

    "positive": 1
}


def conflict_description(
    score
):

    if score is None:
        return "Not Applicable"

    if score <= 19:
        return "Little / No Conflict"

    if score <= 49:
        return "Mild Mixed Sentiment"

    if score <= 74:
        return "Moderate Conflict"

    return "Strong Conflict"


def calculate_text_conflict(
    text
):

    raw_clauses = split_clauses(
        text
    )


    if len(
        raw_clauses
    ) < 2:

        return {

            "score": None,

            "level":
                "Not Applicable",

            "clauses": [],

            "strongest_pair": None
        }


    clause_results = []


    # --------------------------------------------------------
    # Predict each clause separately
    # --------------------------------------------------------

    for index, raw_clause in enumerate(
        raw_clauses,
        start=1
    ):

        model_clause = (
            remove_connector(
                raw_clause
            )
        )

        (
            sentiment,
            confidence,
            probabilities

        ) = predict_text(
            model_clause
        )


        clause_results.append(
            {

                "number":
                    index,

                "text":
                    raw_clause,

                "model_text":
                    model_clause,

                "sentiment":
                    sentiment,

                "confidence":
                    confidence,

                "probabilities":
                    probabilities
            }
        )


    best_conflict = 0.0

    best_pair = None


    # --------------------------------------------------------
    # Compare all clause pairs
    # --------------------------------------------------------

    for i in range(
        len(
            clause_results
        )
    ):

        for j in range(
            i + 1,
            len(
                clause_results
            )
        ):

            first = (
                clause_results[i]
            )

            second = (
                clause_results[j]
            )


            value_a = (
                SENTIMENT_VALUE[
                    first[
                        "sentiment"
                    ]
                ]
            )

            value_b = (
                SENTIMENT_VALUE[
                    second[
                        "sentiment"
                    ]
                ]
            )


            polarity_distance = (

                abs(
                    value_a
                    -
                    value_b
                )

                / 2.0
            )


            minimum_confidence = min(

                first[
                    "confidence"
                ],

                second[
                    "confidence"
                ]
            )


            # Explicit contrast is applied
            # when adjacent clauses are linked
            # by but/however/etc.
            explicit_contrast = (

                j == i + 1

                and

                begins_with_contrast(
                    raw_clauses[j]
                )
            )


            contrast_weight = (

                1.15

                if explicit_contrast

                else 1.0
            )


            pair_conflict = min(

                1.0,

                polarity_distance

                *

                minimum_confidence

                *

                contrast_weight
            )


            if (
                pair_conflict
                >
                best_conflict
            ):

                best_conflict = (
                    pair_conflict
                )

                best_pair = {

                    "first":
                        first,

                    "second":
                        second,

                    "polarity_distance":
                        polarity_distance,

                    "contrast_weight":
                        contrast_weight,

                    "explicit_contrast":
                        explicit_contrast
                }


    conflict_score = int(
        round(
            best_conflict
            *
            100
        )
    )


    return {

        "score":
            conflict_score,

        "level":
            conflict_description(
                conflict_score
            ),

        "clauses":
            clause_results,

        "strongest_pair":
            best_pair
    }


# ============================================================
# PLOTLY HELPERS
# ============================================================

def text_probability_donut(
    probabilities
):

    labels = [
        "Positive",
        "Neutral",
        "Negative"
    ]

    values = [

        probabilities.get(
            "positive",
            0
        )
        *
        100,

        probabilities.get(
            "neutral",
            0
        )
        *
        100,

        probabilities.get(
            "negative",
            0
        )
        *
        100
    ]


    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.58,
                marker=dict(
                    colors=[
                        POSITIVE_COLOR,
                        NEUTRAL_COLOR,
                        NEGATIVE_COLOR
                    ]
                ),
                textinfo=
                    "label+percent",
                hovertemplate=
                    "%{label}: %{value:.2f}%<extra></extra>"
            )
        ]
    )


    fig.update_layout(

        title=dict(
            text=
                "Sentiment Probability",
            x=0.5
        ),

        height=360,

        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        ),

        legend=dict(
            orientation="h",
            y=-0.1,
            x=0.5,
            xanchor="center"
        )
    )

    return fig


def percentage_gauge(
    value,
    title,
    color
):

    value = float(
        value
    )


    fig = go.Figure(
        go.Indicator(
            mode=
                "gauge+number",

            value=
                value,

            number={
                "suffix": "%"
            },

            title={
                "text": title
            },

            gauge={

                "axis": {
                    "range": [
                        0,
                        100
                    ]
                },

                "bar": {
                    "color": color
                },

                "steps": [

                    {
                        "range": [
                            0,
                            40
                        ],
                        "color":
                            "#F1F5F9"
                    },

                    {
                        "range": [
                            40,
                            70
                        ],
                        "color":
                            "#E2E8F0"
                    },

                    {
                        "range": [
                            70,
                            100
                        ],
                        "color":
                            "#CBD5E1"
                    }
                ]
            }
        )
    )


    fig.update_layout(

        height=300,

        margin=dict(
            l=25,
            r=25,
            t=55,
            b=20
        )
    )

    return fig


def face_probability_bar(
    probabilities
):

    order = [

        "happy",
        "neutral",
        "surprise",
        "sad",
        "angry",
        "fear",
        "disgust"
    ]


    data = pd.DataFrame(
        {

            "Emotion":
                [
                    item.title()
                    for item
                    in order
                ],

            "Percentage":
                [
                    probabilities.get(
                        item,
                        0
                    )
                    for item
                    in order
                ],

            "Color":
                [
                    FACE_COLORS[
                        item
                    ]
                    for item
                    in order
                ]
        }
    )


    fig = go.Figure()


    for _, row in (
        data.iterrows()
    ):

        fig.add_trace(
            go.Bar(

                x=[
                    row[
                        "Percentage"
                    ]
                ],

                y=[
                    row[
                        "Emotion"
                    ]
                ],

                orientation="h",

                marker_color=
                    row[
                        "Color"
                    ],

                name=
                    row[
                        "Emotion"
                    ],

                text=[
                    f'{row["Percentage"]:.1f}%'
                ],

                textposition=
                    "outside",

                hovertemplate=
                    f'{row["Emotion"]}: '
                    '%{x:.2f}%'
                    '<extra></extra>'
            )
        )


    fig.update_layout(

        title=
            "Live Emotion Distribution",

        xaxis=dict(
            range=[
                0,
                100
            ],
            title="Confidence (%)"
        ),

        yaxis=dict(
            title=""
        ),

        showlegend=False,

        height=410,

        margin=dict(
            l=30,
            r=40,
            t=55,
            b=35
        ),

        bargap=0.25
    )


    return fig


# ============================================================
# FACE STATE
# ============================================================

class FaceState:

    def __init__(self):

        self.lock = (
            threading.Lock()
        )

        self.frame_number = 0

        self.emotion = None

        self.confidence = 0.0

        self.probabilities = {

            "angry": 0.0,
            "disgust": 0.0,
            "fear": 0.0,
            "happy": 0.0,
            "sad": 0.0,
            "surprise": 0.0,
            "neutral": 0.0
        }

        self.status = (
            "Camera inactive"
        )

        self.region = None

        self.last_update = None

        # Latest facial-analysis error, if any. Keeping it in shared
        # state makes callback failures visible in the Streamlit UI
        # instead of silently looking like an inactive camera.
        self.error = None

        self.history = deque(
            maxlen=30
        )


    def increase_frame(self):

        with self.lock:

            self.frame_number += 1

            return (
                self.frame_number
            )


    def update(

        self,
        emotion,
        confidence,
        probabilities,
        region
    ):

        with self.lock:

            self.emotion = (
                emotion
            )

            self.confidence = float(
                confidence
            )

            self.probabilities = {
                str(key).lower():
                float(value)

                for key, value
                in probabilities.items()
            }

            self.region = (
                region
            )

            self.status = (
                "Face detected"
            )

            self.error = None

            self.last_update = (
                time.time()
            )

            self.history.append(
                {

                    "time":
                        time.strftime(
                            "%H:%M:%S"
                        ),

                    "emotion":
                        emotion.title(),

                    "confidence":
                        float(
                            confidence
                        )
                }
            )


    def camera_active(self):

        with self.lock:

            # Do not overwrite a useful face result while frames continue.
            if self.emotion is None:
                self.status = (
                    "Camera active — preparing facial analysis"
                )


    def no_face(self):

        with self.lock:

            self.status = (
                "Camera active — no clear face detected"
            )

            self.region = None
            self.error = None


    def set_error(self, message):

        with self.lock:

            self.status = (
                "Facial analysis error"
            )

            self.error = str(message)
            self.region = None


    def reset(self):

        with self.lock:

            self.emotion = None

            self.confidence = 0.0

            self.frame_number = 0
            self.error = None

            self.status = (
                "Camera inactive"
            )

            self.region = None

            self.history.clear()


    def snapshot(self):

        with self.lock:

            return {

                "emotion":
                    self.emotion,

                "confidence":
                    self.confidence,

                "probabilities":
                    dict(
                        self.probabilities
                    ),

                "status":
                    self.status,

                "region":
                    self.region,

                "last_update":
                    self.last_update,

                "error":
                    self.error,

                "history":
                    list(
                        self.history
                    )
            }


# ============================================================
# SESSION STATE
# ============================================================

if "face_state" not in (
    st.session_state
):

    st.session_state.face_state = (
        FaceState()
    )


face_state = (
    st.session_state.face_state
)


if "text_history" not in (
    st.session_state
):

    st.session_state.text_history = []


# ============================================================
# FACIAL ANALYSIS CALLBACK
# ============================================================

ANALYZE_EVERY_N_FRAMES = 30


def video_frame_callback(
    frame
):

    image = frame.to_ndarray(
        format="bgr24"
    )

    # If this callback is running, WebRTC is successfully delivering
    # browser camera frames to Python.
    face_state.camera_active()


    # Mirror webcam like a normal camera preview.
    image = cv2.flip(
        image,
        1
    )


    # Keep processing lighter.
    height, width = (
        image.shape[:2]
    )


    if width > 480:

        ratio = (
            480
            /
            width
        )

        image = cv2.resize(
            image,
            (
                480,
                int(
                    height
                    *
                    ratio
                )
            )
        )


    frame_number = (
        face_state
        .increase_frame()
    )


    # --------------------------------------------------------
    # Run emotion analysis only every N frames
    # --------------------------------------------------------

    if (
        frame_number
        %
        ANALYZE_EVERY_N_FRAMES
        ==
        0
    ):

        try:

            with deepface_lock:

                DeepFace = get_deepface()

                analysis = (
                    DeepFace.analyze(

                        img_path=image,

                        actions=[
                            "emotion"
                        ],

                        enforce_detection=True,

                        detector_backend=
                            "opencv",

                        align=False,

                        silent=True
                    )
                )


            if isinstance(
                analysis,
                list
            ):

                analysis = (
                    analysis[0]
                )


            dominant = str(
                analysis[
                    "dominant_emotion"
                ]
            ).lower()


            scores = (
                analysis[
                    "emotion"
                ]
            )


            confidence = float(
                scores.get(
                    dominant,
                    0
                )
            )


            region = (
                analysis.get(
                    "region",
                    None
                )
            )


            face_state.update(

                dominant,

                confidence,

                scores,

                region
            )


        except Exception as exc:

            message = str(exc)
            message_lower = message.lower()

            # DeepFace raises an exception when no usable face is found.
            # Treat that as a normal camera state, but surface real model /
            # dependency errors so deployment problems are visible.
            if (
                "face could not be detected" in message_lower
                or "face not detected" in message_lower
                or "no face" in message_lower
            ):
                face_state.no_face()
            else:
                face_state.set_error(
                    f"{type(exc).__name__}: {message}"
                )


    # --------------------------------------------------------
    # Draw latest result onto video
    # --------------------------------------------------------

    snapshot = (
        face_state.snapshot()
    )


    emotion = (
        snapshot[
            "emotion"
        ]
    )


    region = (
        snapshot[
            "region"
        ]
    )


    if (
        emotion
        is not None
    ):

        label = (

            f"{FACE_EMOJIS.get(emotion, '🙂')} "

            f"{emotion.title()} "

            f"{snapshot['confidence']:.1f}%"
        )


        # Draw face box
        if isinstance(
            region,
            dict
        ):

            x = int(
                region.get(
                    "x",
                    0
                )
            )

            y = int(
                region.get(
                    "y",
                    0
                )
            )

            w = int(
                region.get(
                    "w",
                    0
                )
            )

            h = int(
                region.get(
                    "h",
                    0
                )
            )


            if (
                w > 0
                and
                h > 0
            ):

                cv2.rectangle(

                    image,

                    (
                        x,
                        y
                    ),

                    (
                        x + w,
                        y + h
                    ),

                    (
                        46,
                        204,
                        113
                    ),

                    3
                )


        cv2.rectangle(

            image,

            (
                8,
                8
            ),

            (
                min(
                    image.shape[1] - 8,
                    360
                ),
                55
            ),

            (
                17,
                24,
                39
            ),

            -1
        )


        cv2.putText(

            image,

            label,

            (
                18,
                40
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.75,

            (
                255,
                255,
                255
            ),

            2,

            cv2.LINE_AA
        )


    return (
        av.VideoFrame
        .from_ndarray(
            image,
            format="bgr24"
        )
    )


def video_ended():

    face_state.reset()


# ============================================================
# CROSS-MODAL INTERPRETATION
# ============================================================

def cross_modal_interpretation(

    text_sentiment,
    text_confidence,
    facial_emotion,
    facial_confidence
):

    if (
        text_sentiment
        is None
        or
        facial_emotion
        is None
    ):

        return (
            "Ambiguous / Insufficient Evidence",
            "🟡"
        )


    if (
        text_confidence
        <
        0.55
        or
        facial_confidence
        <
        45
    ):

        return (
            "Ambiguous / Insufficient Evidence",
            "🟡"
        )


    text_sentiment = (
        text_sentiment.lower()
    )

    facial_emotion = (
        facial_emotion.lower()
    )


    positive_face = {
        "happy"
    }


    negative_face = {

        "sad",

        "angry",

        "fear",

        "disgust"
    }


    neutral_face = {
        "neutral"
    }


    ambiguous_face = {
        "surprise"
    }


    if (
        facial_emotion
        in
        ambiguous_face
    ):

        return (
            "Ambiguous / Insufficient Evidence",
            "🟡"
        )


    if (
        text_sentiment
        ==
        "positive"
    ):

        if (
            facial_emotion
            in
            positive_face
        ):

            return (
                "Consistent",
                "🟢"
            )

        if (
            facial_emotion
            in
            negative_face
        ):

            return (
                "Emotion-Text Mismatch Detected",
                "🔴"
            )


    elif (
        text_sentiment
        ==
        "negative"
    ):

        if (
            facial_emotion
            in
            negative_face
        ):

            return (
                "Consistent",
                "🟢"
            )

        if (
            facial_emotion
            in
            positive_face
        ):

            return (
                "Emotion-Text Mismatch Detected",
                "🔴"
            )


    elif (
        text_sentiment
        ==
        "neutral"
    ):

        if (
            facial_emotion
            in
            neutral_face
        ):

            return (
                "Consistent",
                "🟢"
            )


    return (
        "Ambiguous / Insufficient Evidence",
        "🟡"
    )


# ============================================================
# HERO
# ============================================================

st.html(
    textwrap.dedent("""
    <div class="hero">

        <div class="hero-title">
            🧠 Multimodal Sentiment
            & Emotion Analyzer
        </div>

        <div class="hero-subtitle">
            Understand what the text says,
            what the face expresses,
            and where the signals agree or differ.
        </div>

        <span class="badge">
            📝 3 Text Classes
        </span>

        <span class="badge">
            📷 7 Facial Expressions
        </span>

        <span class="badge">
            ⚡ Live Analysis
        </span>

        <span class="badge">
            🔀 Text Conflict Detection
        </span>

        <span class="badge">
            📊 Interactive Analytics
        </span>

    </div>
    """)
)


# ============================================================
# TOP STATS
# ============================================================

top_1, top_2, top_3, top_4 = (
    st.columns(4)
)


top_1.metric(
    "Training Texts",
    "118,452",
    "Balanced dataset"
)


top_2.metric(
    "Text Classes",
    "3",
    "Positive • Neutral • Negative"
)


top_3.metric(
    "Facial Classes",
    "7",
    "Real-time"
)


top_4.metric(
    "Text Feature",
    "Conflict Score",
    "Clause-level"
)


# ============================================================
# MAIN TABS
# ============================================================

(
    text_tab,
    face_tab,
    combined_tab,
    analytics_tab

) = st.tabs(
    [

        "📝 Text Intelligence",

        "📷 Live Face Lab",

        "🔗 Combined Insight",

        "📊 Model Analytics"
    ]
)


# ============================================================
# TEXT TAB
# ============================================================

with text_tab:

    st.subheader(
        "📝 Text Sentiment Intelligence"
    )

    st.caption(
        "Three-class sentiment analysis "
        "with probability analytics and "
        "our text-only Clause-Level Conflict Score."
    )


    # --------------------------------------------------------
    # Quick examples
    # --------------------------------------------------------

    ex1, ex2, ex3, ex4 = (
        st.columns(4)
    )


    if ex1.button(
        "😊 Positive",
        use_container_width=True
    ):

        st.session_state.text_input = (
            "The phone is amazing and "
            "the battery lasts all day."
        )


    if ex2.button(
        "😐 Neutral",
        use_container_width=True
    ):

        st.session_state.text_input = (
            "The meeting begins tomorrow "
            "at 10 AM in room 204."
        )


    if ex3.button(
        "😞 Negative",
        use_container_width=True
    ):

        st.session_state.text_input = (
            "The product was terrible "
            "and the service was extremely slow."
        )


    if ex4.button(
        "🔀 Mixed",
        use_container_width=True
    ):

        st.session_state.text_input = (
            "The food was excellent but "
            "the service was extremely slow."
        )


    # --------------------------------------------------------
    # Text input
    # --------------------------------------------------------

    user_text = st.text_area(

        "Enter text for analysis",

        height=165,

        key="text_input",

        placeholder=(
            "Type a review, comment, "
            "feedback message or statement..."
        )
    )


    analyse_col, clear_col = (
        st.columns(
            [3, 1]
        )
    )


    analyse = (
        analyse_col.button(

            "✨ Analyze Text",

            type="primary",

            use_container_width=True
        )
    )


    if clear_col.button(
        "🗑️ Clear",
        use_container_width=True
    ):

        st.session_state.text_input = ""

        st.session_state.pop(
            "text_result",
            None
        )

        st.rerun()


    # --------------------------------------------------------
    # Analysis
    # --------------------------------------------------------

    if analyse:

        if not (
            user_text.strip()
        ):

            st.warning(
                "Enter some text first."
            )

        else:

            (
                sentiment,
                confidence,
                probabilities

            ) = predict_text(
                user_text
            )


            conflict = (
                calculate_text_conflict(
                    user_text
                )
            )


            result = {

                "text":
                    user_text,

                "sentiment":
                    sentiment,

                "confidence":
                    confidence,

                "probabilities":
                    probabilities,

                "conflict":
                    conflict,

                "words":
                    len(
                        user_text.split()
                    ),

                "characters":
                    len(
                        user_text
                    )
            }


            st.session_state[
                "text_result"
            ] = result


            st.session_state[
                "text_history"
            ].insert(
                0,
                {

                    "Text":
                        user_text[:80],

                    "Sentiment":
                        sentiment.title(),

                    "Confidence":
                        f"{confidence*100:.1f}%",

                    "Conflict":
                        (
                            "N/A"

                            if conflict[
                                "score"
                            ]
                            is None

                            else
                            f'{conflict["score"]}/100'
                        )
                }
            )


            st.session_state[
                "text_history"
            ] = (

                st.session_state[
                    "text_history"
                ][:10]
            )


    result = (
        st.session_state.get(
            "text_result"
        )
    )


    # ========================================================
    # TEXT RESULT DASHBOARD
    # ========================================================

    if result:

        sentiment = (
            result[
                "sentiment"
            ]
        )

        confidence_percent = (
            result[
                "confidence"
            ]
            *
            100
        )


        conflict_score = (
            result[
                "conflict"
            ][
                "score"
            ]
        )


        emoji = (
            TEXT_EMOJIS.get(
                sentiment,
                "🧠"
            )
        )


        st.markdown("---")


        # ----------------------------------------------------
        # Hero result
        # ----------------------------------------------------

        st.html(
    textwrap.dedent(f"""
            <div class="result-card">

                <div style="
                    font-size:3.2rem;
                    text-align:center;
                ">
                    {emoji}
                </div>

                <div style="
                    text-align:center;
                    font-size:1.9rem;
                    font-weight:850;
                    color:{TEXT_COLORS[sentiment]};
                ">
                    {sentiment.upper()}
                </div>

                <div style="
                    text-align:center;
                    margin-top:5px;
                    opacity:0.72;
                ">
                    Overall text sentiment
                </div>

            </div>
            """)
)


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        m1, m2, m3, m4 = (
            st.columns(4)
        )


        m1.metric(
            "Sentiment",
            f"{emoji} {sentiment.title()}"
        )


        m2.metric(
            "Confidence",
            f"{confidence_percent:.1f}%"
        )


        m3.metric(
            "Conflict Score",
            (
                "N/A"

                if conflict_score
                is None

                else
                f"{conflict_score}/100"
            )
        )


        m4.metric(
            "Word Count",
            result[
                "words"
            ]
        )


        # ----------------------------------------------------
        # Confidence warning
        # ----------------------------------------------------

        if (
            result[
                "confidence"
            ]
            <
            0.70
        ):

            st.warning(
                "⚠️ Low-confidence prediction. "
                "Low confidence does NOT automatically "
                "mean Neutral."
            )


        # ----------------------------------------------------
        # Charts
        # ----------------------------------------------------

        chart_1, chart_2 = (
            st.columns(2)
        )


        with chart_1:

            st.plotly_chart(

                text_probability_donut(
                    result[
                        "probabilities"
                    ]
                ),

                use_container_width=True
            )


        with chart_2:

            st.plotly_chart(

                percentage_gauge(

                    confidence_percent,

                    "Model Confidence",

                    TEXT_COLORS[
                        sentiment
                    ]
                ),

                use_container_width=True
            )


        # ----------------------------------------------------
        # Conflict section
        # ----------------------------------------------------

        st.subheader(
            "🔀 Clause-Level Conflict Analysis"
        )


        if (
            conflict_score
            is None
        ):

            st.info(
                "This text does not contain enough "
                "meaningful clauses for a clause-to-clause "
                "Conflict Score."
            )


        else:

            conflict_left, conflict_right = (
                st.columns(
                    [1, 1.5]
                )
            )


            with conflict_left:

                st.plotly_chart(

                    percentage_gauge(

                        conflict_score,

                        "Text Conflict Score",

                        (
                            "#EF4444"

                            if conflict_score
                            >=
                            75

                            else
                            "#F59E0B"

                            if conflict_score
                            >=
                            50

                            else
                            "#3B82F6"
                        )
                    ),

                    use_container_width=True
                )


            with conflict_right:

                st.html(
    textwrap.dedent(f"""
                    <div class="result-card">

                    <h3>
                    📌 {result["conflict"]["level"]}
                    </h3>

                    <p>
                    The Conflict Score measures disagreement
                    between different clauses of the text.
                    It is separate from overall model confidence.
                    </p>

                    <b>
                    Conflict Score:
                    {conflict_score}/100
                    </b>

                    </div>
                    """)
)


            # Clause table
            clause_rows = []


            for item in (
                result[
                    "conflict"
                ][
                    "clauses"
                ]
            ):

                clause_rows.append(
                    {

                        "Clause":
                            item[
                                "number"
                            ],

                        "Text":
                            item[
                                "model_text"
                            ],

                        "Sentiment":
                            (
                                TEXT_EMOJIS[
                                    item[
                                        "sentiment"
                                    ]
                                ]
                                +
                                " "
                                +
                                item[
                                    "sentiment"
                                ].title()
                            ),

                        "Confidence":
                            (
                                f'{item["confidence"]*100:.1f}%'
                            )
                    }
                )


            st.markdown(
                "#### 🧩 Clause Breakdown"
            )


            st.dataframe(

                pd.DataFrame(
                    clause_rows
                ),

                use_container_width=True,

                hide_index=True
            )


            strongest_pair = (
                result[
                    "conflict"
                ][
                    "strongest_pair"
                ]
            )


            if strongest_pair:

                first = (
                    strongest_pair[
                        "first"
                    ]
                )

                second = (
                    strongest_pair[
                        "second"
                    ]
                )


                st.caption(
                    "Strongest disagreement: "
                    f'{first["sentiment"].title()} '
                    "vs "
                    f'{second["sentiment"].title()}'
                )


        # ----------------------------------------------------
        # Detailed percentages
        # ----------------------------------------------------

        with st.expander(
            "🔬 View exact class percentages"
        ):

            probabilities_df = (
                pd.DataFrame(
                    {

                        "Sentiment":
                            [
                                "Positive 😊",
                                "Neutral 😐",
                                "Negative 😞"
                            ],

                        "Probability":
                            [

                                result[
                                    "probabilities"
                                ].get(
                                    "positive",
                                    0
                                )
                                *
                                100,

                                result[
                                    "probabilities"
                                ].get(
                                    "neutral",
                                    0
                                )
                                *
                                100,

                                result[
                                    "probabilities"
                                ].get(
                                    "negative",
                                    0
                                )
                                *
                                100
                            ]
                    }
                )
            )


            probabilities_df[
                "Probability"
            ] = (

                probabilities_df[
                    "Probability"
                ]
                .map(
                    lambda x:
                    f"{x:.2f}%"
                )
            )


            st.dataframe(
                probabilities_df,
                use_container_width=True,
                hide_index=True
            )


        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        if (
            st.session_state[
                "text_history"
            ]
        ):

            with st.expander(
                "🕘 Recent Text Analyses"
            ):

                st.dataframe(

                    pd.DataFrame(
                        st.session_state[
                            "text_history"
                        ]
                    ),

                    use_container_width=True,

                    hide_index=True
                )


# ============================================================
# FACIAL TAB
# ============================================================

with face_tab:

    st.subheader(
        "📷 Real-Time Facial Expression Lab"
    )


    st.caption(
        "Live camera analysis with continuously "
        "updated emotion percentages."
    )


    st.info(
        "🔐 Camera frames are processed during "
        "the active session and are not stored "
        "by this application."
    )


    camera_col, live_col = (
        st.columns(
            [1.25, 1]
        )
    )


    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    with camera_col:

        st.markdown(
            "### 🎥 Live Camera"
        )


        camera_context = (
            webrtc_streamer(

                key=
                    "emotion-camera",

                rtc_configuration={
                    "iceServers": [
                        {
                            "urls": [
                                "stun:stun.l.google.com:19302"
                            ]
                        },
                        {
                            "urls": [
                                "stun:stun.cloudflare.com:3478"
                            ]
                        }
                    ]
                },

                # IMPORTANT: without this callback the browser preview works,
                # but no frame ever reaches DeepFace.
                video_frame_callback=video_frame_callback,

                media_stream_constraints={
                    "video": {
                        "width": {
                            "ideal": 640
                        },
                        "height": {
                            "ideal": 480
                        },
                        "frameRate": {
                            "ideal": 12,
                            "max": 15
                        }
                    },
                    "audio": False
                },

                async_processing=True
            )
        )


        st.caption(
            "Press START to enable your camera. "
            "The facial AI loads only after camera frames "
            "begin arriving. The first result can take "
            "a few seconds."
        )


    # --------------------------------------------------------
    # REAL-TIME DASHBOARD
    # --------------------------------------------------------

    with live_col:

        st.markdown(
            "### ⚡ Live Detection"
        )


        @st.fragment(
            run_every="1s"
        )
        def live_face_summary():

            face = (
                face_state.snapshot()
            )


            if (
                face[
                    "emotion"
                ]
                is None
            ):

                st.html(
    textwrap.dedent("""
                    <div class="result-card">

                        <div style="
                            text-align:center;
                            font-size:3rem;
                        ">
                            📷
                        </div>

                        <div style="
                            text-align:center;
                            font-weight:800;
                            font-size:1.25rem;
                        ">
                            Waiting for facial analysis
                        </div>

                    </div>
                    """)
)


                st.caption(
                    face[
                        "status"
                    ]
                )

                if face.get("error"):
                    st.error(
                        "Facial AI could not run: "
                        + face["error"]
                    )


            else:

                emotion = (
                    face[
                        "emotion"
                    ]
                )


                emoji = (
                    FACE_EMOJIS.get(
                        emotion,
                        "🙂"
                    )
                )


                color = (
                    FACE_COLORS.get(
                        emotion,
                        "#6366F1"
                    )
                )


                st.html(
    textwrap.dedent(f"""
                    <div class="result-card">

                        <div style="
                            text-align:center;
                            font-size:3.8rem;
                        ">
                            {emoji}
                        </div>

                        <div style="
                            text-align:center;
                            font-size:1.8rem;
                            font-weight:850;
                            color:{color};
                        ">
                            {emotion.upper()}
                        </div>

                        <div style="
                            text-align:center;
                            margin-top:5px;
                            opacity:0.75;
                        ">
                            Detected Facial Expression/Emotion
                        </div>

                    </div>
                    """)
)


                a, b = (
                    st.columns(2)
                )


                a.metric(
                    "Emotion",
                    (
                        f"{emoji} "
                        f"{emotion.title()}"
                    )
                )


                b.metric(
                    "Confidence",
                    f'{face["confidence"]:.1f}%'
                )


                if (
                    face[
                        "confidence"
                    ]
                    <
                    45
                ):

                    st.warning(
                        "⚠️ Low-confidence facial result."
                    )


                st.caption(
                    "Status: "
                    +
                    face[
                        "status"
                    ]
                )


        live_face_summary()


    # ========================================================
    # LIVE CHARTS
    # ========================================================

    @st.fragment(
        run_every="1s"
    )
    def live_face_charts():

        snapshot = (
            face_state.snapshot()
        )


        if (
            snapshot[
                "emotion"
            ]
            is not None
        ):

            st.markdown("---")

            st.subheader(
                "📊 Live Facial Analytics"
            )


            chart_left, chart_right = (
                st.columns(2)
            )


            with chart_left:

                st.plotly_chart(

                    face_probability_bar(
                        snapshot[
                            "probabilities"
                        ]
                    ),

                    use_container_width=True
                )


            with chart_right:

                labels = [

                    emotion.title()

                    for emotion
                    in snapshot[
                        "probabilities"
                    ].keys()
                ]


                values = list(

                    snapshot[
                        "probabilities"
                    ].values()
                )


                colors = [

                    FACE_COLORS.get(
                        emotion.lower(),
                        "#94A3B8"
                    )

                    for emotion
                    in snapshot[
                        "probabilities"
                    ].keys()
                ]


                donut = go.Figure(
                    data=[
                        go.Pie(

                            labels=
                                labels,

                            values=
                                values,

                            hole=
                                0.58,

                            marker=dict(
                                colors=colors
                            ),

                            textinfo=
                                "label+percent",

                            hovertemplate=
                                "%{label}: "
                                "%{value:.2f}%"
                                "<extra></extra>"
                        )
                    ]
                )


                donut.update_layout(

                    title=dict(
                        text=
                            "Emotion Probability Mix",
                        x=0.5
                    ),

                    height=410,

                    margin=dict(
                        l=20,
                        r=20,
                        t=55,
                        b=20
                    )
                )


                st.plotly_chart(
                    donut,
                    use_container_width=True
                )


            # ------------------------------------------------
            # Emotion trend
            # ------------------------------------------------

            history = (
                snapshot[
                    "history"
                ]
            )


            if len(
                history
            ) >= 2:

                st.markdown(
                    "#### 📈 Recent Confidence Trend"
                )


                history_df = (
                    pd.DataFrame(
                        history
                    )
                )


                trend = (
                    px.line(

                        history_df,

                        x="time",

                        y="confidence",

                        color="emotion",

                        markers=True,

                        labels={
                            "time":
                                "Time",
                            "confidence":
                                "Confidence (%)",
                            "emotion":
                                "Emotion"
                        }
                    )
                )


                trend.update_yaxes(
                    range=[
                        0,
                        100
                    ]
                )


                trend.update_layout(
                    height=330,
                    legend_title_text=""
                )


                st.plotly_chart(
                    trend,
                    use_container_width=True
                )


    live_face_charts()


    st.caption(
        "ℹ️ Supported pretrained facial labels: "
        "Happy, Sad, Angry, Neutral, Surprise, "
        "Fear and Disgust. 'Excited' is not "
        "directly produced by this model."
    )


    st.warning(
        "Facial-expression models estimate "
        "visible expression patterns. They do "
        "not determine a person's true internal "
        "emotional state."
    )


# ============================================================
# COMBINED TAB
# ============================================================

with combined_tab:

    st.subheader(
        "🔗 Combined Text–Face Insight"
    )


    st.caption(
        "The original text and facial results "
        "stay independent. This section only "
        "describes whether they appear aligned."
    )


    st.info(
        "🔀 The numeric Conflict Score is TEXT ONLY. "
        "There is deliberately no numeric "
        "text–face conflict score."
    )


    @st.fragment(
        run_every="1s"
    )
    def combined_dashboard():

        text_result = (
            st.session_state.get(
                "text_result"
            )
        )


        face_result = (
            face_state.snapshot()
        )


        if (
            text_result
            is None
        ):

            st.info(
                "📝 Analyze some text first "
                "in the Text Intelligence tab."
            )

            return


        if (
            face_result[
                "emotion"
            ]
            is None
        ):

            st.info(
                "📷 Start the camera in "
                "Live Face Lab and wait for "
                "a facial result."
            )

            return


        text_sentiment = (
            text_result[
                "sentiment"
            ]
        )


        face_emotion = (
            face_result[
                "emotion"
            ]
        )


        (
            status,
            status_icon

        ) = cross_modal_interpretation(

            text_sentiment,

            text_result[
                "confidence"
            ],

            face_emotion,

            face_result[
                "confidence"
            ]
        )


        # ----------------------------------------------------
        # Side-by-side signals
        # ----------------------------------------------------

        text_side, face_side = (
            st.columns(2)
        )


        with text_side:

            st.html(
    textwrap.dedent(f"""
                <div class="result-card">

                    <div style="
                        text-align:center;
                        font-size:3rem;
                    ">
                        {TEXT_EMOJIS[text_sentiment]}
                    </div>

                    <div style="
                        text-align:center;
                        font-size:1.45rem;
                        font-weight:850;
                        color:{TEXT_COLORS[text_sentiment]};
                    ">
                        {text_sentiment.upper()}
                    </div>

                    <div style="
                        text-align:center;
                        margin-top:7px;
                    ">
                        Text Confidence:
                        {text_result["confidence"]*100:.1f}%
                    </div>

                </div>
                """)
)


        with face_side:

            face_color = (
                FACE_COLORS.get(
                    face_emotion,
                    "#64748B"
                )
            )


            st.html(
    textwrap.dedent(f"""
                <div class="result-card">

                    <div style="
                        text-align:center;
                        font-size:3rem;
                    ">
                        {FACE_EMOJIS.get(face_emotion, "🙂")}
                    </div>

                    <div style="
                        text-align:center;
                        font-size:1.45rem;
                        font-weight:850;
                        color:{face_color};
                    ">
                        {face_emotion.upper()}
                    </div>

                    <div style="
                        text-align:center;
                        margin-top:7px;
                    ">
                        Facial Confidence:
                        {face_result["confidence"]:.1f}%
                    </div>

                </div>
                """)
)


        # ----------------------------------------------------
        # Interpretation
        # ----------------------------------------------------

        if (
            status
            ==
            "Consistent"
        ):

            background = (
                "linear-gradient("
                "135deg,#DCFCE7,#BBF7D0)"
            )

            text_color = (
                "#166534"
            )


        elif (
            status
            ==
            "Emotion-Text Mismatch Detected"
        ):

            background = (
                "linear-gradient("
                "135deg,#FEE2E2,#FECACA)"
            )

            text_color = (
                "#991B1B"
            )


        else:

            background = (
                "linear-gradient("
                "135deg,#FEF3C7,#FDE68A)"
            )

            text_color = (
                "#92400E"
            )


        st.html(
    textwrap.dedent(f"""
            <div class="status-box"
                 style="
                    background:{background};
                    color:{text_color};
                 ">

                {status_icon}
                {status}

            </div>
            """)
)


        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )


        summary_1, summary_2, summary_3 = (
            st.columns(3)
        )


        conflict_score = (

            text_result[
                "conflict"
            ][
                "score"
            ]
        )


        summary_1.metric(

            "Text Confidence",

            f'{text_result["confidence"]*100:.1f}%'
        )


        summary_2.metric(

            "Facial Confidence",

            f'{face_result["confidence"]:.1f}%'
        )


        summary_3.metric(

            "Text Conflict",

            (
                "N/A"

                if conflict_score
                is None

                else
                f"{conflict_score}/100"
            )
        )


        # ----------------------------------------------------
        # Interpretation explanation
        # ----------------------------------------------------

        if (
            status
            ==
            "Emotion-Text Mismatch Detected"
        ):

            st.warning(
                "The text sentiment and visible "
                "facial-expression prediction point "
                "in different directions. This can "
                "indicate a mismatch, but it does NOT "
                "prove sarcasm, deception, or a person's "
                "actual emotional state."
            )


        elif (
            status
            ==
            "Consistent"
        ):

            st.success(
                "The current text and facial-expression "
                "predictions appear broadly consistent."
            )


        else:

            st.info(
                "The available signals are not strong "
                "enough for a clear consistency or "
                "mismatch interpretation."
            )


    combined_dashboard()


# ============================================================
# MODEL ANALYTICS TAB
# ============================================================

with analytics_tab:

    st.subheader(
        "📊 Model & Dataset Analytics"
    )


    st.caption(
        "Transparent evaluation data from "
        "the current trained three-class model."
    )


    # --------------------------------------------------------
    # Core metrics
    # --------------------------------------------------------

    q1, q2, q3, q4 = (
        st.columns(4)
    )


    q1.metric(
        "Balanced Dataset",
        "118,452",
        "Total texts"
    )


    q2.metric(
        "Per Class",
        "39,484",
        "Perfectly balanced"
    )


    q3.metric(
        "Internal Accuracy",
        "73.39%"
    )


    q4.metric(
        "Macro F1",
        "0.7332"
    )


    st.warning(
        "The stricter untouched TweetEval test "
        "produced 58.05% accuracy and Macro F1 "
        "of 0.5581. It is shown separately because "
        "it measures cross-dataset generalization."
    )


    # --------------------------------------------------------
    # Dataset charts
    # --------------------------------------------------------

    st.markdown(
        "### 🗂️ Training Dataset"
    )


    class_df = pd.DataFrame(
        {

            "Sentiment": [
                "Positive",
                "Neutral",
                "Negative"
            ],

            "Count": [
                39484,
                39484,
                39484
            ]
        }
    )


    source_df = pd.DataFrame(
        {

            "Source": [

                "SST3 + DynaSent",
                "TweetEval",
                "Amazon",
                "IMDb",
                "Sentiment140",
                "Restaurant"
            ],

            "Count": [

                36645,
                26763,
                17953,
                17877,
                17768,
                1446
            ]
        }
    )


    d1, d2 = (
        st.columns(2)
    )


    with d1:

        class_chart = (
            px.pie(

                class_df,

                names="Sentiment",

                values="Count",

                hole=0.58,

                color="Sentiment",

                color_discrete_map={
                    "Positive":
                        POSITIVE_COLOR,
                    "Neutral":
                        NEUTRAL_COLOR,
                    "Negative":
                        NEGATIVE_COLOR
                },

                title=
                    "Final Class Distribution"
            )
        )


        class_chart.update_layout(
            height=390
        )


        st.plotly_chart(
            class_chart,
            use_container_width=True
        )


    with d2:

        source_chart = (
            px.bar(

                source_df,

                x="Source",

                y="Count",

                text="Count",

                title=
                    "Training Source Distribution"
            )
        )


        source_chart.update_traces(
            textposition="outside"
        )


        source_chart.update_layout(
            height=390
        )


        st.plotly_chart(
            source_chart,
            use_container_width=True
        )


    # --------------------------------------------------------
    # Internal classification metrics
    # --------------------------------------------------------

    st.markdown(
        "### 🎯 Internal Held-Out Test"
    )


    internal_metrics = (
        pd.DataFrame(
            {

                "Class": [
                    "Negative",
                    "Neutral",
                    "Positive"
                ],

                "Precision": [
                    0.7587,
                    0.6845,
                    0.7753
                ],

                "Recall": [
                    0.6984,
                    0.8221,
                    0.6811
                ],

                "F1": [
                    0.7273,
                    0.7470,
                    0.7252
                ]
            }
        )
    )


    metrics_chart = (
        px.bar(

            internal_metrics.melt(
                id_vars="Class",
                var_name="Metric",
                value_name="Score"
            ),

            x="Class",

            y="Score",

            color="Metric",

            barmode="group",

            text_auto=".2f",

            title=
                "Precision / Recall / F1 by Class"
        )
    )


    metrics_chart.update_yaxes(
        range=[
            0,
            1
        ]
    )


    metrics_chart.update_layout(
        height=420
    )


    st.plotly_chart(
        metrics_chart,
        use_container_width=True
    )


    # --------------------------------------------------------
    # Internal confusion matrix
    # --------------------------------------------------------

    internal_cm = np.array(
        [
            [5515, 1434, 948],
            [794, 6492, 611],
            [960, 1558, 5379]
        ]
    )


    labels = [
        "Negative",
        "Neutral",
        "Positive"
    ]


    internal_heatmap = (
        go.Figure(
            data=
                go.Heatmap(

                    z=
                        internal_cm,

                    x=[
                        "Pred Negative",
                        "Pred Neutral",
                        "Pred Positive"
                    ],

                    y=[
                        "Actual Negative",
                        "Actual Neutral",
                        "Actual Positive"
                    ],

                    text=
                        internal_cm,

                    texttemplate=
                        "%{text}",

                    colorscale=
                        "Purples"
                )
        )
    )


    internal_heatmap.update_layout(

        title=
            "Internal Confusion Matrix",

        height=410
    )


    st.plotly_chart(
        internal_heatmap,
        use_container_width=True
    )


    # --------------------------------------------------------
    # External TweetEval
    # --------------------------------------------------------

    st.markdown(
        "### 🌍 Untouched External TweetEval Test"
    )


    e1, e2 = (
        st.columns(2)
    )


    e1.metric(
        "External Accuracy",
        "58.05%"
    )


    e2.metric(
        "External Macro F1",
        "0.5581"
    )


    external_cm = np.array(
        [
            [1888, 1716, 368],
            [1083, 4014, 840],
            [276, 870, 1229]
        ]
    )


    external_heatmap = (
        go.Figure(
            data=
                go.Heatmap(

                    z=
                        external_cm,

                    x=[
                        "Pred Negative",
                        "Pred Neutral",
                        "Pred Positive"
                    ],

                    y=[
                        "Actual Negative",
                        "Actual Neutral",
                        "Actual Positive"
                    ],

                    text=
                        external_cm,

                    texttemplate=
                        "%{text}",

                    colorscale=
                        "Blues"
                )
        )
    )


    external_heatmap.update_layout(

        title=
            "External TweetEval Confusion Matrix",

        height=410
    )


    st.plotly_chart(
        external_heatmap,
        use_container_width=True
    )


    st.caption(
        "Internal and external evaluation are "
        "shown separately to avoid presenting "
        "cross-dataset performance as if it were "
        "the same evaluation."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")


st.html(
    textwrap.dedent("""
    <div style="
        text-align:center;
        opacity:0.75;
        padding:8px;
    ">

        🧠 Text Sentiment
        &nbsp;•&nbsp;
        🔀 Clause Conflict
        &nbsp;•&nbsp;
        📷 Facial Expression
        &nbsp;•&nbsp;
        📊 Live Analytics

        <br>

        <span style="font-size:0.80rem;">
        Text Conflict Score is numeric and text-only.
        Facial analysis remains independent.
        Combined interpretation is qualitative only.
        </span>

    </div>
    """)
)
