"""
Content Matcher for RiboReach

Matches educational content to user profiles and misinformation categories.
Uses semantic similarity and rule-based matching.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import logging
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

from user_profiler import UserProfile, ConcernType, InformationStyle, TrustLevel
from dynamic_classifier import MisinfoCategory

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Types of educational content."""
    ARTICLE = "article"
    VIDEO = "video"
    INFOGRAPHIC = "infographic"
    FAQ = "faq"
    EXPERT_INTERVIEW = "expert_interview"
    PERSONAL_STORY = "personal_story"
    FACT_CHECK = "fact_check"
    INTERACTIVE = "interactive"


class ContentDifficulty(Enum):
    """Reading/comprehension level."""
    BASIC = "basic"           # Simple language, 6th grade level
    INTERMEDIATE = "intermediate"  # General audience
    ADVANCED = "advanced"     # Scientific/technical detail


@dataclass
class EducationalContent:
    """Educational content item."""
    content_id: str
    title: str
    summary: str
    full_text: str
    content_type: ContentType
    difficulty: ContentDifficulty
    
    # Targeting
    target_concerns: List[ConcernType]
    counters_misinfo: List[MisinfoCategory]
    suitable_trust_levels: List[TrustLevel]
    
    # Metadata
    source: str  # e.g., "CDC", "Local Doctor", "Vaccine Recipient"
    author_type: str  # "health_authority", "doctor", "peer", "scientist"
    emotional_tone: str  # "empathetic", "neutral", "authoritative"
    
    # Engagement tracking
    view_count: int = 0
    share_count: int = 0
    effectiveness_score: float = 0.5  # 0-1, how effective at changing views
    
    # Optional media
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content_id": self.content_id,
            "title": self.title,
            "summary": self.summary,
            "content_type": self.content_type.value,
            "difficulty": self.difficulty.value,
            "target_concerns": [c.value for c in self.target_concerns],
            "counters_misinfo": [m.value for m in self.counters_misinfo],
            "source": self.source,
            "author_type": self.author_type,
            "effectiveness_score": self.effectiveness_score
        }


@dataclass
class ContentMatch:
    """Result of content matching."""
    content: EducationalContent
    relevance_score: float      # How relevant to user's concerns
    style_match_score: float    # How well it matches preferred style
    trust_appropriate: bool     # Appropriate for user's trust level
    overall_score: float        # Combined match score
    match_reasons: List[str]    # Why this content was matched


class ContentMatcher:
    """
    Matches educational content to users based on their profiles and 
    the misinformation they've encountered.
    
    Example usage:
    ```python
    matcher = ContentMatcher()
    matcher.load_content_library(content_df)
    
    # Match content to user
    matches = matcher.match_for_user(user_profile, top_n=5)
    
    # Match content to counter specific misinformation
    matches = matcher.match_for_misinfo(MisinfoCategory.VACCINE_SAFETY, top_n=3)
    ```
    """
    
    # Mapping of information styles to content types
    STYLE_TO_CONTENT = {
        InformationStyle.SCIENTIFIC: [ContentType.ARTICLE, ContentType.FACT_CHECK],
        InformationStyle.STORYTELLING: [ContentType.PERSONAL_STORY, ContentType.VIDEO],
        InformationStyle.VISUAL: [ContentType.INFOGRAPHIC, ContentType.VIDEO],
        InformationStyle.CONVERSATIONAL: [ContentType.FAQ, ContentType.INTERACTIVE],
        InformationStyle.AUTHORITATIVE: [ContentType.EXPERT_INTERVIEW, ContentType.ARTICLE],
        InformationStyle.PEER: [ContentType.PERSONAL_STORY, ContentType.VIDEO]
    }
    
    # Mapping of trust levels to preferred author types
    TRUST_TO_AUTHOR = {
        TrustLevel.HIGH: ["health_authority", "scientist", "doctor"],
        TrustLevel.MODERATE: ["doctor", "scientist", "peer"],
        TrustLevel.LOW: ["peer", "doctor", "local_expert"],
        TrustLevel.VERY_LOW: ["peer", "former_skeptic", "community_member"]
    }
    
    def __init__(self):
        self.content_library: Dict[str, EducationalContent] = {}
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self._content_vectors = None
        self._is_vectorized = False
    
    def add_content(self, content: EducationalContent):
        """Add content to the library."""
        self.content_library[content.content_id] = content
        self._is_vectorized = False  # Need to re-vectorize
        logger.debug(f"Added content: {content.content_id}")
    
    def load_content_library(self, data: pd.DataFrame):
        """
        Load content library from DataFrame.
        
        Expected columns: content_id, title, summary, full_text, content_type,
                         difficulty, target_concerns, counters_misinfo, source,
                         author_type, emotional_tone
        """
        for _, row in data.iterrows():
            # Parse list fields
            target_concerns = self._parse_enum_list(
                row.get("target_concerns", []), ConcernType
            )
            counters_misinfo = self._parse_enum_list(
                row.get("counters_misinfo", []), MisinfoCategory
            )
            suitable_trust = self._parse_enum_list(
                row.get("suitable_trust_levels", ["moderate"]), TrustLevel
            )
            
            content = EducationalContent(
                content_id=str(row["content_id"]),
                title=row["title"],
                summary=row["summary"],
                full_text=row.get("full_text", row["summary"]),
                content_type=ContentType(row.get("content_type", "article")),
                difficulty=ContentDifficulty(row.get("difficulty", "intermediate")),
                target_concerns=target_concerns,
                counters_misinfo=counters_misinfo,
                suitable_trust_levels=suitable_trust,
                source=row.get("source", "Unknown"),
                author_type=row.get("author_type", "expert"),
                emotional_tone=row.get("emotional_tone", "neutral"),
                effectiveness_score=row.get("effectiveness_score", 0.5)
            )
            self.content_library[content.content_id] = content
        
        self._vectorize_content()
        logger.info(f"Loaded {len(self.content_library)} content items")
    
    def _parse_enum_list(self, value: Any, enum_class) -> List:
        """Parse a list of enum values."""
        if isinstance(value, str):
            value = [v.strip() for v in value.split(",")]
        elif not isinstance(value, list):
            value = [value] if value else []
        
        result = []
        for v in value:
            try:
                if isinstance(v, enum_class):
                    result.append(v)
                else:
                    result.append(enum_class(v))
            except (ValueError, KeyError):
                pass
        return result
    
    def _vectorize_content(self):
        """Create TF-IDF vectors for content matching."""
        if not self.content_library:
            return
        
        texts = []
        self._content_ids = []
        
        for content_id, content in self.content_library.items():
            # Combine title, summary, and concern keywords
            text = f"{content.title} {content.summary}"
            texts.append(text)
            self._content_ids.append(content_id)
        
        self._content_vectors = self.vectorizer.fit_transform(texts)
        self._is_vectorized = True
    
    def _calculate_relevance(
        self,
        content: EducationalContent,
        user_profile: UserProfile,
        misinfo_category: Optional[MisinfoCategory] = None
    ) -> float:
        """Calculate relevance score for content given user profile."""
        score = 0.0
        
        # Check if content targets user's concerns
        user_concerns = set(user_profile.primary_concerns)
        content_concerns = set(content.target_concerns)
        concern_overlap = len(user_concerns & content_concerns)
        score += concern_overlap * 0.3
        
        # Check if content counters encountered misinformation
        if misinfo_category and misinfo_category in content.counters_misinfo:
            score += 0.4
        
        # Boost for effectiveness
        score += content.effectiveness_score * 0.2
        
        return min(1.0, score)
    
    def _calculate_style_match(
        self,
        content: EducationalContent,
        user_profile: UserProfile
    ) -> float:
        """Calculate how well content style matches user preference."""
        preferred_types = self.STYLE_TO_CONTENT.get(
            user_profile.preferred_style,
            [ContentType.ARTICLE]
        )
        
        # Type match
        type_score = 0.5 if content.content_type in preferred_types else 0.2
        
        # Author type match
        preferred_authors = self.TRUST_TO_AUTHOR.get(
            user_profile.trust_level,
            ["doctor"]
        )
        author_score = 0.3 if content.author_type in preferred_authors else 0.1
        
        # Difficulty match (simplified - could use education level)
        difficulty_score = 0.2  # Default moderate match
        
        return type_score + author_score + difficulty_score
    
    def _check_trust_appropriate(
        self,
        content: EducationalContent,
        user_profile: UserProfile
    ) -> bool:
        """Check if content is appropriate for user's trust level."""
        return user_profile.trust_level in content.suitable_trust_levels
    
    def match_for_user(
        self,
        user_profile: UserProfile,
        misinfo_category: Optional[MisinfoCategory] = None,
        top_n: int = 5,
        exclude_viewed: bool = True
    ) -> List[ContentMatch]:
        """
        Find best matching content for a user.
        
        Args:
            user_profile: User's profile
            misinfo_category: Optional specific misinformation to counter
            top_n: Number of matches to return
            exclude_viewed: Whether to exclude already viewed content
        
        Returns:
            List of ContentMatch objects ranked by relevance
        """
        matches = []
        viewed_set = set(user_profile.topics_viewed) if exclude_viewed else set()
        
        for content_id, content in self.content_library.items():
            if content_id in viewed_set:
                continue
            
            # Calculate scores
            relevance = self._calculate_relevance(content, user_profile, misinfo_category)
            style_match = self._calculate_style_match(content, user_profile)
            trust_ok = self._check_trust_appropriate(content, user_profile)
            
            # Trust penalty if not appropriate
            trust_factor = 1.0 if trust_ok else 0.5
            
            # Combined score
            overall = (relevance * 0.5 + style_match * 0.3) * trust_factor
            overall += 0.2 * content.effectiveness_score
            
            # Build match reasons
            reasons = []
            if relevance > 0.3:
                reasons.append("Addresses your concerns")
            if style_match > 0.5:
                reasons.append("Matches your information preferences")
            if trust_ok:
                reasons.append("From a source you may trust")
            if misinfo_category and misinfo_category in content.counters_misinfo:
                reasons.append("Directly addresses common misconceptions")
            
            if not reasons:
                reasons.append("General educational content")
            
            matches.append(ContentMatch(
                content=content,
                relevance_score=relevance,
                style_match_score=style_match,
                trust_appropriate=trust_ok,
                overall_score=overall,
                match_reasons=reasons
            ))
        
        # Sort by overall score
        matches.sort(key=lambda x: -x.overall_score)
        
        return matches[:top_n]
    
    def match_for_misinfo(
        self,
        category: MisinfoCategory,
        difficulty: Optional[ContentDifficulty] = None,
        top_n: int = 3
    ) -> List[EducationalContent]:
        """Find content that counters specific misinformation."""
        matches = []
        
        for content in self.content_library.values():
            if category in content.counters_misinfo:
                if difficulty and content.difficulty != difficulty:
                    continue
                matches.append(content)
        
        # Sort by effectiveness
        matches.sort(key=lambda x: -x.effectiveness_score)
        
        return matches[:top_n]
    
    def search_content(self, query: str, top_n: int = 5) -> List[EducationalContent]:
        """Search content library by text similarity."""
        if not self._is_vectorized:
            self._vectorize_content()
        
        if not self._is_vectorized:
            return []
        
        # Vectorize query
        query_vec = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vec, self._content_vectors)[0]
        
        # Get top matches
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        return [
            self.content_library[self._content_ids[i]]
            for i in top_indices
            if similarities[i] > 0.1  # Minimum threshold
        ]
    
    def update_effectiveness(
        self,
        content_id: str,
        engagement_data: Dict[str, Any]
    ):
        """Update content effectiveness based on engagement data."""
        if content_id not in self.content_library:
            return
        
        content = self.content_library[content_id]
        
        # Update view/share counts
        content.view_count += engagement_data.get("views", 0)
        content.share_count += engagement_data.get("shares", 0)
        
        # Update effectiveness score based on engagement signals
        if engagement_data.get("trust_increased"):
            content.effectiveness_score = min(1.0, content.effectiveness_score + 0.05)
        if engagement_data.get("negative_feedback"):
            content.effectiveness_score = max(0.1, content.effectiveness_score - 0.03)
    
    def get_content_stats(self) -> pd.DataFrame:
        """Get statistics about the content library."""
        if not self.content_library:
            return pd.DataFrame()
        
        records = []
        for content in self.content_library.values():
            records.append({
                "content_id": content.content_id,
                "title": content.title[:50],
                "type": content.content_type.value,
                "difficulty": content.difficulty.value,
                "views": content.view_count,
                "shares": content.share_count,
                "effectiveness": content.effectiveness_score
            })
        
        return pd.DataFrame(records)
    
    def save(self, filepath: str):
        """Save content library."""
        joblib.dump({
            "content_library": self.content_library,
            "vectorizer": self.vectorizer,
            "content_vectors": self._content_vectors,
            "content_ids": self._content_ids if hasattr(self, '_content_ids') else []
        }, filepath)
        logger.info(f"Content library saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "ContentMatcher":
        """Load content library."""
        data = joblib.load(filepath)
        matcher = cls()
        matcher.content_library = data["content_library"]
        matcher.vectorizer = data["vectorizer"]
        matcher._content_vectors = data["content_vectors"]
        matcher._content_ids = data["content_ids"]
        matcher._is_vectorized = True
        logger.info(f"Loaded content library from {filepath}")
        return matcher


def generate_sample_content() -> pd.DataFrame:
    """Generate sample educational content for testing."""
    content = [
        {
            "content_id": "C001",
            "title": "How Vaccines Are Tested for Safety",
            "summary": "Learn about the rigorous multi-phase testing process all vaccines go through before approval.",
            "content_type": "article",
            "difficulty": "intermediate",
            "target_concerns": "safety",
            "counters_misinfo": "vaccine_safety",
            "source": "FDA",
            "author_type": "health_authority",
            "emotional_tone": "authoritative",
            "effectiveness_score": 0.75
        },
        {
            "content_id": "C002",
            "title": "I Was Hesitant Too: One Mom's Journey",
            "summary": "Sarah shares her story of overcoming vaccine fears and making an informed decision for her family.",
            "content_type": "personal_story",
            "difficulty": "basic",
            "target_concerns": "safety,trust",
            "counters_misinfo": "vaccine_safety",
            "source": "Community Member",
            "author_type": "peer",
            "emotional_tone": "empathetic",
            "effectiveness_score": 0.82
        },
        {
            "content_id": "C003",
            "title": "Understanding Vaccine Ingredients",
            "summary": "A simple breakdown of what's actually in vaccines and why each ingredient is used.",
            "content_type": "infographic",
            "difficulty": "basic",
            "target_concerns": "ingredients",
            "counters_misinfo": "ingredient_fear",
            "source": "Children's Hospital",
            "author_type": "doctor",
            "emotional_tone": "neutral",
            "effectiveness_score": 0.68
        },
        {
            "content_id": "C004",
            "title": "mRNA Technology: 30 Years in the Making",
            "summary": "The COVID vaccine wasn't rushed - discover the decades of research behind mRNA technology.",
            "content_type": "video",
            "difficulty": "intermediate",
            "target_concerns": "speed",
            "counters_misinfo": "rushed_development",
            "source": "Science Channel",
            "author_type": "scientist",
            "emotional_tone": "informative",
            "effectiveness_score": 0.71
        },
        {
            "content_id": "C005",
            "title": "Common Vaccine Questions Answered",
            "summary": "Your doctor answers the most frequently asked questions about vaccines in simple terms.",
            "content_type": "faq",
            "difficulty": "basic",
            "target_concerns": "safety,efficacy",
            "counters_misinfo": "vaccine_safety,vaccine_efficacy",
            "source": "Local Doctor",
            "author_type": "doctor",
            "emotional_tone": "conversational",
            "effectiveness_score": 0.77
        },
        {
            "content_id": "C006",
            "title": "From Skeptic to Advocate: A Nurse's Story",
            "summary": "Nurse James was skeptical until he saw the data firsthand. Here's what changed his mind.",
            "content_type": "personal_story",
            "difficulty": "basic",
            "target_concerns": "trust,safety",
            "counters_misinfo": "distrust_authority",
            "source": "Healthcare Worker",
            "author_type": "peer",
            "emotional_tone": "empathetic",
            "effectiveness_score": 0.85
        },
        {
            "content_id": "C007",
            "title": "Natural vs. Vaccine Immunity Explained",
            "summary": "Understanding why vaccines provide safer protection than getting sick.",
            "content_type": "article",
            "difficulty": "intermediate",
            "target_concerns": "efficacy",
            "counters_misinfo": "natural_immunity",
            "source": "Immunologist",
            "author_type": "scientist",
            "emotional_tone": "neutral",
            "effectiveness_score": 0.69
        },
        {
            "content_id": "C008",
            "title": "Side Effects: What to Actually Expect",
            "summary": "Real data on common side effects and how they compare to disease risks.",
            "content_type": "fact_check",
            "difficulty": "intermediate",
            "target_concerns": "safety",
            "counters_misinfo": "side_effects",
            "source": "Medical Journal",
            "author_type": "scientist",
            "emotional_tone": "neutral",
            "effectiveness_score": 0.73
        },
        {
            "content_id": "C009",
            "title": "Free and Low-Cost Vaccine Options",
            "summary": "Many vaccines are available for free or at reduced cost. Learn about programs that can help.",
            "content_type": "faq",
            "difficulty": "basic",
            "target_concerns": "cost_access",
            "counters_misinfo": "",
            "source": "CDC",
            "author_type": "health_authority",
            "emotional_tone": "helpful",
            "effectiveness_score": 0.80
        },
        {
            "content_id": "C010",
            "title": "Finding Vaccines in Your Community",
            "summary": "A guide to locating vaccine providers near you, including free clinics and pharmacy programs.",
            "content_type": "interactive",
            "difficulty": "basic",
            "target_concerns": "cost_access",
            "counters_misinfo": "",
            "source": "Vaccines.gov",
            "author_type": "health_authority",
            "emotional_tone": "supportive",
            "effectiveness_score": 0.78
        },
        {
            "content_id": "C011",
            "title": "Insurance Coverage for Vaccines Explained",
            "summary": "Most insurance plans cover vaccines at no cost. Here's what you need to know about your coverage.",
            "content_type": "article",
            "difficulty": "intermediate",
            "target_concerns": "cost_access",
            "counters_misinfo": "",
            "source": "Healthcare.gov",
            "author_type": "health_authority",
            "emotional_tone": "informative",
            "effectiveness_score": 0.72
        }
    ]
    
    return pd.DataFrame(content)
