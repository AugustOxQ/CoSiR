"""Exact shared image/caption system instructions."""
PROMPTS = {
    "default": "Represent the user's input.",
    "emotion_focus": "Represent the emotion that this artwork, or this description of an artwork, conveys. Focus only on the emotion.",
    "emotion_word": "Describe the emotion of this artwork in one word.",
    "emotion_neg": "Represent the emotion of this artwork, not its subject and not its artistic style.",
    "genre_focus": "Represent the genre and subject matter of this artwork, or of this description of an artwork. Focus only on the genre and subject matter.",
    "genre_word": "Describe the genre of this artwork in one word.",
    "genre_neg": "Represent the genre and subject matter of this artwork, not its emotion and not its artistic style.",
    "style_focus": "Represent the artistic style of this artwork, or of this description of an artwork. Focus only on the artistic style.",
    "style_word": "Describe the artistic style of this artwork in one word.",
    "style_neg": "Represent the artistic style of this artwork, not its subject and not its emotion.",
}
