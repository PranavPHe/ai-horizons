"""
RiboReach Flask API Server

Provides REST endpoints for the web frontend to call the AI backend.
"""

import sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from dynamic_classifier import DynamicMisinformationClassifier, MisinfoCategory
from user_profiler import UserProfiler, TrustLevel
from dynamic_responder import DynamicResponder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Initialize components lazily
_classifier = None
_profiler = None
_responder = None


def get_classifier():
    """Lazy-load the classifier."""
    global _classifier
    if _classifier is None:
        logger.info("Initializing DynamicMisinformationClassifier...")
        _classifier = DynamicMisinformationClassifier(use_embeddings=True)
    return _classifier


def get_profiler():
    """Lazy-load the profiler."""
    global _profiler
    if _profiler is None:
        logger.info("Initializing UserProfiler...")
        _profiler = UserProfiler()
    return _profiler


def get_responder():
    """Lazy-load the responder."""
    global _responder
    if _responder is None:
        logger.info("Initializing DynamicResponder...")
        _responder = DynamicResponder(get_classifier())
    return _responder


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "riboreach-api"})


# Patterns to detect legitimate concerns vs misinformation
CONCERN_PATTERNS = [
    r"\b(worried|concern|scared|afraid|nervous|anxious|unsure|hesitant)\b",
    r"\b(should i|is it safe|can i|will it|what if|wondering)\b",
    r"\?(.*$)",  # Questions
    r"\b(heard that|someone told me|read that|saw online)\b",
    r"\b(don'?t know|not sure|confused about)\b",
    r"\b(my (child|children|kids|baby|family))\b",
    r"\b(considering|thinking about|deciding)\b",
]

# Patterns that indicate definitive misinformation claims
MISINFO_CLAIM_PATTERNS = [
    r"vaccines? (cause|causes|caused|causing|will cause)",
    r"(it'?s|they'?re|vaccines? are) (a |)(lie|hoax|scam|fake|fraud)",
    r"(proven|fact|know|truth):?\s*(that\s*)?vaccines?",
    r"vaccines? (don'?t|doesn'?t|never|won'?t) (work|help|protect)",
    r"vaccines? (contain|have|put|inject|are full of)\s*(microchip|chip|5g|graphene|poison|tracker)",
    r"(big pharma|they|government) (is |are )?(hiding|covering|lying|don'?t want)",
    r"wake up|open your eyes|sheeple|do your research",
    r"(kill|murder|genocide|depopulation|sterilize|infertility)",
]

# Patterns for general information-seeking
INFO_SEEKING_PATTERNS = [
    r"^(what|where|how|when|why|which|can you|could you)",
    r"\b(tell me|explain|help me understand|more info|learn about)\b",
    r"\b(schedule|appointment|cost|location|available|eligible)\b",
]

import re

def classify_intent(text: str) -> tuple:
    """
    Classify whether text is misinformation, legitimate concern, or info-seeking.
    Returns: (intent_type, confidence_modifier)
    
    intent_type: 'misinformation', 'legitimate_concern', 'question', 'neutral'
    """
    text_lower = text.lower()
    
    # Count pattern matches
    misinfo_matches = sum(1 for p in MISINFO_CLAIM_PATTERNS if re.search(p, text_lower, re.IGNORECASE))
    concern_matches = sum(1 for p in CONCERN_PATTERNS if re.search(p, text_lower, re.IGNORECASE))
    info_matches = sum(1 for p in INFO_SEEKING_PATTERNS if re.search(p, text_lower, re.IGNORECASE))
    
    # If strong misinformation language regardless of concern framing
    if misinfo_matches >= 2:
        return ('misinformation', 1.0)
    
    # If expressing concern without making false claims
    # Higher match count → higher confidence that this is a genuine concern
    if concern_matches >= 2 and misinfo_matches == 0:
        concern_confidence = min(0.95, 0.7 + concern_matches * 0.05)
        return ('legitimate_concern', concern_confidence)
    
    # If asking questions
    if info_matches >= 1 and misinfo_matches == 0:
        return ('question', 0.75)
    
    # Mixed signals - check ratio
    if concern_matches > misinfo_matches:
        return ('legitimate_concern', 0.65)
    elif misinfo_matches > 0:
        return ('misinformation', 0.8)
    
    return ('neutral', 0.5)


@app.route("/api/analyze", methods=["POST"])
def analyze_text():
    """
    Analyze text for misinformation vs legitimate concerns.
    
    Expected JSON payload:
    {
        "text": "I'm worried about vaccine side effects for my children"
    }
    """
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        # First, classify intent
        intent_type, confidence_mod = classify_intent(text)
        
        classifier = get_classifier()
        result = classifier.classify(text)
        
        # Adjust classification based on intent
        is_misinformation = result.is_misinformation
        adjusted_confidence = result.confidence
        
        if intent_type == 'legitimate_concern':
            # This is a genuine concern, not misinformation
            # Use the intent classifier's confidence directly — it reflects
            # how sure we are this is a legitimate concern
            is_misinformation = False
            adjusted_confidence = confidence_mod
        elif intent_type == 'question':
            # Information seeking
            is_misinformation = False
            adjusted_confidence = confidence_mod
        elif intent_type == 'misinformation':
            # Definitive false claim
            is_misinformation = True
            adjusted_confidence = min(0.95, max(adjusted_confidence, confidence_mod))
        
        # Generate response based on whether it's misinformation or concern
        response_data = None
        responder = get_responder()
        
        if is_misinformation:
            dynamic_response = responder.generate_response(text)
            if dynamic_response:
                response_data = {
                    "headline": dynamic_response.headline,
                    "explanation": dynamic_response.explanation,
                    "key_facts": dynamic_response.key_facts,
                    "myth": dynamic_response.myth_vs_fact[0] if dynamic_response.myth_vs_fact else None,
                    "fact": dynamic_response.myth_vs_fact[1] if dynamic_response.myth_vs_fact else None,
                    "sources": dynamic_response.sources,
                    "follow_up": dynamic_response.follow_up_suggestion
                }
        else:
            # For legitimate concerns, provide supportive educational response
            response_data = generate_concern_response(text, result.primary_category.value, intent_type)
        
        return jsonify({
            "success": True,
            "classification": {
                "is_misinformation": is_misinformation,
                "intent_type": intent_type,
                "category": result.primary_category.value,
                "confidence": round(adjusted_confidence, 3),
                "severity": result.severity if is_misinformation else "none",
                "detected_claims": result.detected_claims,
                "category_scores": {k: round(v, 3) for k, v in result.category_scores.items()}
            },
            "response": response_data
        })
        
    except Exception as e:
        logger.exception("Error in analyze_text")
        return jsonify({"success": False, "error": str(e)}), 500


def generate_concern_response(text: str, category: str, intent_type: str) -> dict:
    """Generate a supportive response for legitimate concerns."""
    
    concern_responses = {
        "vaccine_safety": {
            "headline": "Your Safety Concerns Are Valid",
            "explanation": "It's completely reasonable to want to understand vaccine safety. Many people share these concerns, and asking questions is a healthy part of making informed decisions.",
            "key_facts": [
                "All vaccines go through rigorous clinical trials before approval",
                "Safety monitoring continues long after vaccines are approved",
                "You can discuss your specific concerns with your healthcare provider",
                "Serious side effects are rare and carefully tracked"
            ],
            "sources": ["Your healthcare provider", "CDC.gov/vaccines", "FDA vaccine information"]
        },
        "side_effects": {
            "headline": "Understanding Side Effects",
            "explanation": "Worrying about side effects is natural. Most vaccine side effects are mild and temporary, but it's good to know what to expect.",
            "key_facts": [
                "Common side effects include soreness, fatigue, and mild fever",
                "These usually resolve within 1-2 days",
                "Serious side effects are very rare but are monitored closely",
                "Talk to your doctor about any specific health conditions"
            ],
            "sources": ["CDC Side Effect Information", "Your pharmacist", "Vaccine package inserts"]
        },
        "rushed_development": {
            "headline": "Understanding Development Speed",
            "explanation": "Questions about how quickly vaccines were developed are understandable. Let me explain what actually happened.",
            "key_facts": [
                "mRNA vaccine technology was researched for over 30 years before COVID",
                "Development phases were run in parallel, not skipped",
                "Global funding and coordination accelerated timelines",
                "Full safety trials were completed before authorization"
            ],
            "sources": ["FDA approval documentation", "Scientific publications on mRNA research history"]
        },
        "ingredient_fear": {
            "headline": "Understanding Vaccine Ingredients",
            "explanation": "Wanting to know what's in vaccines is completely fair. Full ingredient lists are publicly available.",
            "key_facts": [
                "Every ingredient has a specific purpose and is tested for safety",
                "Ingredient amounts are carefully measured and well below harmful levels",
                "You can review the full ingredient list with your healthcare provider"
            ],
            "sources": ["Vaccine package inserts", "CDC ingredient information", "Your pharmacist"]
        }
    }
    
    # Default response for other categories or general questions
    default_response = {
        "headline": "Thank You for Your Question",
        "explanation": "Asking questions about vaccines is an important part of making informed health decisions. Here's some information that might help.",
        "key_facts": [
            "Talk to your healthcare provider about your specific situation",
            "Reputable sources include CDC.gov and WHO.int",
            "It's okay to take time to make your decision"
        ],
        "sources": ["Your healthcare provider", "CDC.gov", "WHO.int"]
    }
    
    response = concern_responses.get(category, default_response)
    
    # Add a follow-up based on intent
    if intent_type == 'question':
        response["follow_up"] = "Would you like more specific information about any of these points?"
    else:
        response["follow_up"] = "Would you like to discuss your specific concerns further?"
    
    return response


@app.route("/api/profile", methods=["POST"])
def create_profile():
    """
    Create or update user profile based on their input.
    
    Expected JSON payload:
    {
        "text": "I'm worried about vaccine side effects for my children"
    }
    """
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        profiler = get_profiler()
        profile = profiler.analyze(text)
        
        return jsonify({
            "success": True,
            "profile": {
                "trust_level": profile.trust_level.value,
                "trust_score": round(profile.trust_score, 2),
                "primary_concerns": [c.value for c in profile.primary_concerns],
                "information_style": profile.information_style.value,
                "concerns_breakdown": {
                    c.value: round(s, 3) 
                    for c, s in profile.concern_scores.items()
                },
                "recommended_approach": get_approach_for_trust_level(profile.trust_level),
                "content_recommendations": generate_content_recommendations(profile)
            }
        })
        
    except Exception as e:
        logger.exception("Error in create_profile")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/respond", methods=["POST"])
def generate_response():
    """
    Generate an educational response to a specific concern or claim.
    
    Expected JSON payload:
    {
        "text": "mRNA vaccines modify your DNA",
        "user_profile": { ... optional profile data ... }
    }
    """
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        
        if not text:
            return jsonify({"success": False, "error": "No text provided"}), 400
        
        responder = get_responder()
        response = responder.generate_response(text)
        
        if not response:
            return jsonify({
                "success": True,
                "response": {
                    "headline": "Thank you for your question",
                    "explanation": "This appears to be a legitimate concern or question. We recommend consulting with a healthcare professional for personalized advice.",
                    "key_facts": [],
                    "sources": ["Your local healthcare provider", "CDC.gov", "WHO.int"]
                }
            })
        
        return jsonify({
            "success": True,
            "response": {
                "category": response.detected_category,
                "confidence": round(response.confidence, 3),
                "headline": response.headline,
                "explanation": response.explanation,
                "key_facts": response.key_facts,
                "myth": response.myth_vs_fact[0] if response.myth_vs_fact else None,
                "fact": response.myth_vs_fact[1] if response.myth_vs_fact else None,
                "sources": response.sources,
                "follow_up": response.follow_up_suggestion
            }
        })
        
    except Exception as e:
        logger.exception("Error in generate_response")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/batch-analyze", methods=["POST"])
def batch_analyze():
    """
    Analyze multiple texts at once.
    
    Expected JSON payload:
    {
        "texts": ["text1", "text2", "text3"]
    }
    """
    try:
        data = request.get_json()
        texts = data.get("texts", [])
        
        if not texts:
            return jsonify({"success": False, "error": "No texts provided"}), 400
        
        classifier = get_classifier()
        results = []
        
        misinfo_count = 0
        category_counts = {}
        
        for text in texts[:50]:  # Limit to 50 texts
            result = classifier.classify(text)
            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "is_misinformation": result.is_misinformation,
                "category": result.primary_category.value,
                "confidence": round(result.confidence, 3),
                "severity": result.severity
            })
            
            if result.is_misinformation:
                misinfo_count += 1
                cat = result.primary_category.value
                category_counts[cat] = category_counts.get(cat, 0) + 1
        
        return jsonify({
            "success": True,
            "summary": {
                "total_analyzed": len(results),
                "misinformation_count": misinfo_count,
                "legitimate_count": len(results) - misinfo_count,
                "misinfo_rate": round(misinfo_count / len(results) * 100, 1) if results else 0,
                "category_breakdown": category_counts
            },
            "results": results
        })
        
    except Exception as e:
        logger.exception("Error in batch_analyze")
        return jsonify({"success": False, "error": str(e)}), 500


def get_approach_for_trust_level(trust_level: TrustLevel) -> dict:
    """Get recommended communication approach for a trust level."""
    approaches = {
        TrustLevel.VERY_LOW: {
            "tone": "empathetic_non_confrontational",
            "strategy": "Build rapport first, acknowledge concerns, use stories and testimonials",
            "avoid": "Authoritative language, dismissing concerns, medical jargon"
        },
        TrustLevel.LOW: {
            "tone": "curious_collaborative", 
            "strategy": "Ask questions, explore concerns together, provide balanced information",
            "avoid": "Lecturing, overwhelming with data"
        },
        TrustLevel.MODERATE: {
            "tone": "informative_balanced",
            "strategy": "Provide clear facts with context, address specific questions",
            "avoid": "Condescension, oversimplification"
        },
        TrustLevel.HIGH: {
            "tone": "direct_factual",
            "strategy": "Share detailed information, scientific sources, encourage others",
            "avoid": "Unnecessary repetition"
        }
    }
    return approaches.get(trust_level, approaches[TrustLevel.MODERATE])


def generate_content_recommendations(profile) -> list:
    """Generate content recommendations based on profile."""
    recommendations = []
    
    from user_profiler import ConcernType, InformationStyle
    
    # Based on primary concerns
    concern_content = {
        ConcernType.SAFETY: "Vaccine safety monitoring and clinical trial data",
        ConcernType.EFFICACY: "Real-world effectiveness studies and statistics",
        ConcernType.INGREDIENTS: "Ingredient breakdowns and purpose explanations",
        ConcernType.SPEED: "History of mRNA research and development timeline",
        ConcernType.TRUST: "Independent research and peer review process",
        ConcernType.FREEDOM: "Individual rights and community health balance",
        ConcernType.MEDICAL: "Medical exemption information and alternatives",
        ConcernType.RELIGIOUS: "Religious leader perspectives and exemptions"
    }
    
    for concern in profile.primary_concerns[:3]:
        if concern in concern_content:
            recommendations.append(concern_content[concern])
    
    # Based on information style
    style_additions = {
        InformationStyle.SCIENTIFIC: "Peer-reviewed research papers and clinical data",
        InformationStyle.EMOTIONAL: "Personal stories and community experiences",
        InformationStyle.PRACTICAL: "FAQ-style quick answers and action steps",
        InformationStyle.ANALYTICAL: "Comparative data and risk-benefit analyses"
    }
    
    if profile.information_style in style_additions:
        recommendations.append(style_additions[profile.information_style])
    
    return recommendations if recommendations else ["General vaccine education resources"]


if __name__ == "__main__":
    print("Starting RiboReach API server on port 5003...")
    app.run(host="0.0.0.0", port=5003, debug=True)
