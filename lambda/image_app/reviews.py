# Simplified review generation for Art Classifier
from typing import Dict, List
import random

# Simplified review templates by style category
STYLE_REVIEWS: Dict[str, List[str]] = {
    "Impressionism": [
        "Soft light and blurred contours create a dreamy atmosphere.",
        "Bright colors capture fleeting moments of light.",
        "Romantic play of light and shadow evokes emotion."
    ],
    "Post_Impressionism": [
        "Stronger contrasts and expressive colors show artistic evolution.",
        "Saturated colors and symbolic elements create depth.",
        "Bold brushwork moves beyond traditional impressionism."
    ],
    "Expressionism": [
        "Intense colors and distorted forms express inner emotions.",
        "Dramatic composition reveals the artist's psychological state.",
        "Bold contrasts create powerful emotional impact."
    ],
    "Cubism": [
        "Geometric forms fragment reality into multiple perspectives.",
        "Analytical breakdown creates new visual understanding.",
        "Multi-dimensional view challenges traditional representation."
    ],
    "Abstract_Expressionism": [
        "Pure abstraction conveys raw emotional energy.",
        "Spontaneous brush gestures express subconscious creativity.",
        "Non-representational forms create visual poetry."
    ],
    "Pop_Art": [
        "Bright colors and bold contrasts celebrate popular culture.",
        "Commercial imagery transformed into high art.",
        "Vibrant composition reflects modern consumer society."
    ],
    "Baroque": [
        "Theatrical composition with dramatic lighting effects.",
        "Rich details and opulent colors create grandeur.",
        "Dynamic movement and emotional intensity dominate."
    ],
    "Renaissance": [
        "Classical harmony and technical perfection.",
        "Attention to detail and realistic proportions.",
        "Idealized forms reflect humanistic values."
    ],
    "Romanticism": [
        "Emotional intensity overrides rational composition.",
        "Nature's power and human emotion intertwine.",
        "Melancholy atmosphere evokes deep feeling."
    ],
    "Minimalism": [
        "Clean lines and simple forms create powerful impact.",
        "Reduced elements focus attention on essential beauty.",
        "Geometric precision achieves visual harmony."
    ]
}

def generate_review(style: str, confidence: float) -> str:
    """Generate AI review based on predicted style and confidence"""
    # Get reviews for the style, fallback to generic
    reviews = STYLE_REVIEWS.get(style, [
        "Interesting work with unique artistic character.",
        "Distinctive style shows creative expression.",
        "Artistic elements create visual interest."
    ])
    
    # Select review based on confidence level
    if confidence > 0.8:
        # High confidence - use first (most specific) review
        return reviews[0]
    elif confidence > 0.6:
        # Medium confidence - use second review or first if only one
        return reviews[1] if len(reviews) > 1 else reviews[0]
    else:
        # Low confidence - use last review or random if multiple
        if len(reviews) > 2:
            return reviews[-1]
        elif len(reviews) > 1:
            return reviews[1]
        else:
            return reviews[0]

def get_confidence_description(confidence: float) -> str:
    """Get human-readable confidence description"""
    if confidence > 0.9:
        return "Very High"
    elif confidence > 0.8:
        return "High"
    elif confidence > 0.6:
        return "Medium"
    elif confidence > 0.4:
        return "Low"
    else:
        return "Very Low"
