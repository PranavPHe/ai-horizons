"""
User Profiler for RiboReach

Profiles users based on their concerns, information needs, and trust levels
to enable personalized educational content delivery.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime
import logging

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)


class ConcernType(Enum):
    """Types of vaccine-related concerns."""
    SAFETY = "safety"                   # General safety concerns
    EFFICACY = "efficacy"               # Doubts about effectiveness
    TRUST = "trust"                     # Trust in institutions
    INGREDIENTS = "ingredients"         # Specific ingredient concerns
    DEVELOPMENT_SPEED = "speed"         # Rushed development concerns
    PERSONAL_FREEDOM = "freedom"        # Autonomy/mandate concerns
    RELIGIOUS = "religious"             # Religious/ethical concerns
    MEDICAL_HISTORY = "medical"         # Personal health conditions
    INFORMATION_OVERLOAD = "overload"   # Confusion from conflicting info
    COST_ACCESS = "cost_access"         # Cost, affordability, accessibility
    NONE = "none"                       # No significant concerns


class InformationStyle(Enum):
    """Preferred information delivery style."""
    SCIENTIFIC = "scientific"           # Detailed, data-driven
    STORYTELLING = "storytelling"       # Personal stories, narratives
    VISUAL = "visual"                   # Infographics, charts
    CONVERSATIONAL = "conversational"   # Simple, friendly explanation
    AUTHORITATIVE = "authoritative"     # Expert/doctor recommendations
    PEER = "peer"                       # Community voices, testimonials


class TrustLevel(Enum):
    """Trust level in health institutions."""
    HIGH = "high"           # Trusts health authorities
    MODERATE = "moderate"   # Some skepticism but open
    LOW = "low"             # Significant distrust
    VERY_LOW = "very_low"   # Deep distrust, may believe conspiracies


@dataclass
class UserProfile:
    """Profile of a user for personalization."""
    user_id: str
    
    # Core attributes
    primary_concerns: List[ConcernType] = field(default_factory=list)
    trust_level: TrustLevel = TrustLevel.MODERATE
    preferred_style: InformationStyle = InformationStyle.CONVERSATIONAL
    
    # Demographics (optional, for research/analytics)
    age_group: Optional[str] = None  # '18-25', '26-35', etc.
    education_level: Optional[str] = None
    location_type: Optional[str] = None  # 'urban', 'suburban', 'rural'
    
    # Behavioral signals
    engagement_score: float = 0.5  # 0-1, how engaged they are
    openness_to_info: float = 0.5  # 0-1, receptiveness to new info
    
    # Interaction history
    topics_viewed: List[str] = field(default_factory=list)
    content_interactions: Dict[str, int] = field(default_factory=dict)
    misinformation_encountered: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    last_interaction: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "primary_concerns": [c.value for c in self.primary_concerns],
            "trust_level": self.trust_level.value,
            "preferred_style": self.preferred_style.value,
            "engagement_score": self.engagement_score,
            "openness_to_info": self.openness_to_info,
            "topics_viewed_count": len(self.topics_viewed),
            "created_at": self.created_at.isoformat(),
            "last_interaction": self.last_interaction.isoformat()
        }
    
    def update_engagement(self, interaction_type: str, positive: bool = True):
        """Update engagement score based on interaction."""
        delta = 0.05 if positive else -0.02
        self.engagement_score = np.clip(self.engagement_score + delta, 0, 1)
        self.last_interaction = datetime.now()
        
        # Track content interaction
        self.content_interactions[interaction_type] = \
            self.content_interactions.get(interaction_type, 0) + 1


class UserProfiler:
    """
    Profiles users based on their text inputs and interactions.
    
    Example usage:
    ```python
    profiler = UserProfiler()
    
    # Analyze user text to infer concerns
    profile = profiler.analyze_and_create_profile(
        user_id="user123",
        texts=["I'm worried about the rushed development", "Can I trust the CDC?"]
    )
    
    print(f"Primary concerns: {profile.primary_concerns}")
    print(f"Trust level: {profile.trust_level}")
    ```
    """
    
    # Keywords for detecting concerns
    CONCERN_KEYWORDS = {
        ConcernType.SAFETY: [
            "safe", "safety", "dangerous", "risk", "harm", "injury",
            "side effect", "side effects", "side affect", "side affects",
            "adverse", "hurt", "damage", "autism",
            "cause", "causes", "linked", "toxic", "poison",
            "worried", "worry", "worries", "concerned", "concern", "concerns",
            "scared", "afraid", "fear", "nervous", "anxious", "anxiety", "uneasy"
        ],
        ConcernType.EFFICACY: [
            "work", "effective", "efficacy", "protect", "prevent",
            "breakthrough", "still get sick", "useless",
            "doubt", "uncertain", "wonder if"
        ],
        ConcernType.TRUST: [
            "trust", "believe", "lying", "corrupt", "agenda",
            "truth", "honest", "transparent", "hidden",
            "skeptical", "suspicious"
        ],
        ConcernType.INGREDIENTS: [
            "ingredient", "contain", "mercury", "aluminum", "chemical",
            "what's in", "fetal", "graphene", "mrna"
        ],
        ConcernType.DEVELOPMENT_SPEED: [
            "rushed", "fast", "quick", "experimental", "tested",
            "long-term", "years", "trials", "guinea pig"
        ],
        ConcernType.PERSONAL_FREEDOM: [
            "choice", "mandate", "forced", "freedom", "rights",
            "my body", "require", "coerce", "pressure"
        ],
        ConcernType.RELIGIOUS: [
            "religious", "faith", "god", "church", "moral",
            "ethical", "sin", "belief", "spiritual"
        ],
        ConcernType.MEDICAL_HISTORY: [
            "allergic", "allergy", "condition", "immune", "pregnant",
            "autoimmune", "medication", "doctor said", "health issue"
        ],
        ConcernType.COST_ACCESS: [
            "cost", "costly", "expensive", "afford", "affordable", "price",
            "cheap", "free", "pay", "money", "insurance", "coverage",
            "access", "accessible", "available", "availability", "hard to get",
            "where to get", "can't get", "no access", "underserved", "rural"
        ]
    }
    
    # Trust level indicators
    DISTRUST_INDICATORS = [
        "don't trust", "can't trust", "lying", "cover up", "corrupt",
        "agenda", "big pharma", "government control", "they want",
        "wake up", "sheeple", "conspiracy", "hiding"
    ]
    
    # Misinformation phrases that indicate low trust
    MISINFO_PHRASES = [
        "cause autism", "causes autism", "autism", "microchip", "chip",
        "5g", "infertility", "dna change", "alter dna", "magnetic",
        "depopulation", "kill people", "poison", "toxic", "bioweapon",
        "plandemic", "scamdemic", "hoax", "fake", "not real",
        "immune system damage", "more harm", "deadly", "death jab"
    ]
    
    def __init__(self):
        self.profiles: Dict[str, UserProfile] = {}
    
    def _detect_concerns(self, texts: List[str]) -> Dict[ConcernType, float]:
        """Detect concerns from user texts."""
        combined_text = " ".join(texts).lower()
        concern_scores = {}
        
        for concern, keywords in self.CONCERN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in combined_text)
            if score > 0:
                concern_scores[concern] = min(1.0, score / 3)  # Normalize
        
        return concern_scores
    
    def _assess_trust_level(self, texts: List[str]) -> TrustLevel:
        """Assess user's trust level from their texts."""
        combined_text = " ".join(texts).lower()
        
        distrust_count = sum(
            1 for indicator in self.DISTRUST_INDICATORS
            if indicator in combined_text
        )
        
        # Also count misinformation phrases as strong distrust indicators
        misinfo_count = sum(
            1 for phrase in self.MISINFO_PHRASES
            if phrase in combined_text
        )
        
        # Misinformation phrases are weighted more heavily
        total_distrust = distrust_count + (misinfo_count * 2)
        
        if total_distrust >= 4:
            return TrustLevel.VERY_LOW
        elif total_distrust >= 2:
            return TrustLevel.LOW
        elif total_distrust >= 1:
            return TrustLevel.MODERATE
        else:
            return TrustLevel.HIGH
    
    def _infer_preferred_style(
        self,
        trust_level: TrustLevel,
        concerns: List[ConcernType]
    ) -> InformationStyle:
        """Infer preferred information style based on profile."""
        # Low trust users often respond better to peer voices
        if trust_level in [TrustLevel.LOW, TrustLevel.VERY_LOW]:
            return InformationStyle.PEER
        
        # Scientific concerns may want data
        if ConcernType.EFFICACY in concerns or ConcernType.SAFETY in concerns:
            return InformationStyle.SCIENTIFIC
        
        # Emotional concerns may prefer stories
        if ConcernType.RELIGIOUS in concerns or ConcernType.PERSONAL_FREEDOM in concerns:
            return InformationStyle.STORYTELLING
        
        # Default to conversational
        return InformationStyle.CONVERSATIONAL
    
    def analyze_and_create_profile(
        self,
        user_id: str,
        texts: List[str],
        demographics: Optional[Dict[str, str]] = None
    ) -> UserProfile:
        """
        Analyze user texts and create a profile.
        
        Args:
            user_id: Unique user identifier
            texts: List of texts from the user (posts, comments, questions)
            demographics: Optional demographic information
        
        Returns:
            UserProfile with inferred attributes
        """
        # Detect concerns
        concern_scores = self._detect_concerns(texts)
        primary_concerns = [
            concern for concern, score in 
            sorted(concern_scores.items(), key=lambda x: -x[1])[:3]
            if score > 0.3
        ]
        
        if not primary_concerns:
            primary_concerns = [ConcernType.NONE]
        
        # Assess trust
        trust_level = self._assess_trust_level(texts)
        
        # Infer style
        preferred_style = self._infer_preferred_style(trust_level, primary_concerns)
        
        # Calculate openness
        openness = 0.5
        if trust_level == TrustLevel.HIGH:
            openness = 0.8
        elif trust_level == TrustLevel.VERY_LOW:
            openness = 0.2
        
        # Check for question marks (indicates seeking information)
        question_count = sum(text.count('?') for text in texts)
        if question_count > 0:
            openness = min(1.0, openness + 0.1 * question_count)
        
        # Create profile
        profile = UserProfile(
            user_id=user_id,
            primary_concerns=primary_concerns,
            trust_level=trust_level,
            preferred_style=preferred_style,
            openness_to_info=openness,
            age_group=demographics.get("age_group") if demographics else None,
            education_level=demographics.get("education") if demographics else None,
            location_type=demographics.get("location_type") if demographics else None
        )
        
        # Track misinformation encountered
        profile.misinformation_encountered = texts
        
        # Store profile
        self.profiles[user_id] = profile
        
        logger.info(f"Created profile for {user_id}: concerns={[c.value for c in primary_concerns]}, trust={trust_level.value}")
        
        return profile
    
    def update_profile(
        self,
        user_id: str,
        new_texts: Optional[List[str]] = None,
        interaction_data: Optional[Dict[str, Any]] = None
    ) -> UserProfile:
        """Update an existing profile with new data."""
        if user_id not in self.profiles:
            raise ValueError(f"Profile not found for {user_id}")
        
        profile = self.profiles[user_id]
        
        if new_texts:
            # Re-analyze with new texts
            all_texts = profile.misinformation_encountered + new_texts
            concern_scores = self._detect_concerns(all_texts)
            
            # Update concerns if new ones emerge
            new_concerns = [
                concern for concern, score in concern_scores.items()
                if score > 0.3 and concern not in profile.primary_concerns
            ]
            profile.primary_concerns.extend(new_concerns[:2])
            profile.misinformation_encountered.extend(new_texts)
        
        if interaction_data:
            # Update engagement based on interactions
            if interaction_data.get("clicked_content"):
                profile.update_engagement("content_click", positive=True)
            if interaction_data.get("shared_content"):
                profile.update_engagement("content_share", positive=True)
            if interaction_data.get("dismissed_content"):
                profile.update_engagement("content_dismiss", positive=False)
            if interaction_data.get("viewed_topic"):
                profile.topics_viewed.append(interaction_data["viewed_topic"])
        
        profile.last_interaction = datetime.now()
        return profile
    
    def get_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get a user profile by ID."""
        return self.profiles.get(user_id)
    
    def get_similar_profiles(
        self,
        profile: UserProfile,
        top_n: int = 5
    ) -> List[UserProfile]:
        """Find similar user profiles for community matching."""
        similarities = []
        
        for other_id, other_profile in self.profiles.items():
            if other_id == profile.user_id:
                continue
            
            # Calculate similarity based on concerns and trust
            concern_overlap = len(
                set(profile.primary_concerns) & set(other_profile.primary_concerns)
            )
            trust_match = 1 if profile.trust_level == other_profile.trust_level else 0
            style_match = 1 if profile.preferred_style == other_profile.preferred_style else 0
            
            similarity = concern_overlap * 0.5 + trust_match * 0.3 + style_match * 0.2
            similarities.append((other_profile, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: -x[1])
        
        return [profile for profile, _ in similarities[:top_n]]
    
    def segment_profiles(self) -> Dict[str, List[UserProfile]]:
        """Segment profiles into groups for targeted campaigns."""
        segments = {
            "high_trust_engaged": [],
            "moderate_trust_seekers": [],
            "low_trust_safety_concerns": [],
            "low_trust_conspiracy": [],
            "freedom_focused": [],
            "information_seekers": []
        }
        
        for profile in self.profiles.values():
            if profile.trust_level == TrustLevel.HIGH and profile.engagement_score > 0.6:
                segments["high_trust_engaged"].append(profile)
            elif profile.trust_level == TrustLevel.MODERATE and profile.openness_to_info > 0.5:
                segments["moderate_trust_seekers"].append(profile)
            elif profile.trust_level in [TrustLevel.LOW, TrustLevel.VERY_LOW]:
                if ConcernType.SAFETY in profile.primary_concerns:
                    segments["low_trust_safety_concerns"].append(profile)
                elif ConcernType.TRUST in profile.primary_concerns:
                    segments["low_trust_conspiracy"].append(profile)
            if ConcernType.PERSONAL_FREEDOM in profile.primary_concerns:
                segments["freedom_focused"].append(profile)
            if profile.openness_to_info > 0.7:
                segments["information_seekers"].append(profile)
        
        return segments
    
    def get_segment_stats(self) -> pd.DataFrame:
        """Get statistics about user segments."""
        segments = self.segment_profiles()
        
        stats = []
        for segment_name, profiles in segments.items():
            if profiles:
                stats.append({
                    "segment": segment_name,
                    "count": len(profiles),
                    "avg_engagement": np.mean([p.engagement_score for p in profiles]),
                    "avg_openness": np.mean([p.openness_to_info for p in profiles])
                })
        
        return pd.DataFrame(stats)
    
    def save(self, filepath: str):
        """Save profiler state."""
        joblib.dump(self.profiles, filepath)
        logger.info(f"Profiles saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "UserProfiler":
        """Load profiler state."""
        profiler = cls()
        profiler.profiles = joblib.load(filepath)
        logger.info(f"Loaded {len(profiler.profiles)} profiles from {filepath}")
        return profiler
