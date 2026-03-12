"""
Personalization Engine for RiboReach

Generates personalized educational responses tailored to individual users.
Combines user profiles, misinformation context, and content matching.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import logging
from datetime import datetime
import json
import re

import numpy as np
import pandas as pd

from user_profiler import UserProfile, ConcernType, InformationStyle, TrustLevel
from dynamic_classifier import MisinfoCategory, ClassificationResult
from content_matcher import ContentMatcher, ContentMatch, EducationalContent, ContentType

logger = logging.getLogger(__name__)


class ResponseTone(Enum):
    """Tone for personalized responses."""
    EMPATHETIC = "empathetic"       # Understanding, validating concerns
    INFORMATIVE = "informative"     # Fact-focused, educational
    CONVERSATIONAL = "conversational"  # Friendly, approachable
    SCIENTIFIC = "scientific"       # Data-driven, detailed
    ENCOURAGING = "encouraging"     # Supportive, positive


class ResponseFormat(Enum):
    """Format for response delivery."""
    SHORT_MESSAGE = "short_message"   # 1-2 sentences
    SUMMARY = "summary"               # Brief paragraph
    DETAILED = "detailed"             # Full explanation
    BULLET_POINTS = "bullet_points"   # Easy to scan
    QA_STYLE = "qa_style"             # Question-answer format


@dataclass
class PersonalizedResponse:
    """A personalized educational response."""
    response_id: str
    user_id: str
    
    # Core message
    headline: str
    main_message: str
    supporting_points: List[str]
    
    # Personalization details
    tone: ResponseTone
    format: ResponseFormat
    difficulty_level: str
    
    # Referenced content
    content_recommendations: List[str]  # Content IDs
    source_citations: List[str]
    
    # Call to action
    next_step: Optional[str] = None
    follow_up_questions: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    targeting_reason: str = ""
    estimated_read_time: int = 0  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "response_id": self.response_id,
            "user_id": self.user_id,
            "headline": self.headline,
            "main_message": self.main_message,
            "supporting_points": self.supporting_points,
            "tone": self.tone.value,
            "format": self.format.value,
            "content_recommendations": self.content_recommendations,
            "next_step": self.next_step,
            "estimated_read_time": self.estimated_read_time
        }
    
    def to_text(self) -> str:
        """Convert to readable text format."""
        text = f"# {self.headline}\n\n"
        text += f"{self.main_message}\n\n"
        
        if self.supporting_points:
            text += "**Key Points:**\n"
            for point in self.supporting_points:
                text += f"• {point}\n"
            text += "\n"
        
        if self.next_step:
            text += f"**Next Step:** {self.next_step}\n"
        
        return text


class PersonalizationEngine:
    """
    Engine for generating personalized educational responses.
    
    Combines user profiling, misinformation classification, and content
    matching to create tailored messages that resonate with users.
    
    Example usage:
    ```python
    engine = PersonalizationEngine(content_matcher)
    
    # Generate response for user
    response = engine.generate_response(
        user_profile=profile,
        misinfo_result=classification,
        context="User asked about vaccine ingredients"
    )
    ```
    """
    
    # Templates for different tones
    TONE_TEMPLATES = {
        ResponseTone.EMPATHETIC: {
            "opener": "It's completely understandable to have questions about {topic}.",
            "transition": "Many people share your concern, and here's what we know:",
            "closer": "Your health decisions matter, and getting good information is the first step."
        },
        ResponseTone.INFORMATIVE: {
            "opener": "Here's what the research shows about {topic}:",
            "transition": "The key facts to consider are:",
            "closer": "Based on this evidence, you can make an informed decision."
        },
        ResponseTone.CONVERSATIONAL: {
            "opener": "Great question about {topic}!",
            "transition": "Here's the thing:",
            "closer": "Hope this helps! Feel free to ask more questions."
        },
        ResponseTone.SCIENTIFIC: {
            "opener": "The scientific consensus on {topic} is based on extensive research.",
            "transition": "Studies have demonstrated that:",
            "closer": "These findings have been replicated across multiple peer-reviewed studies."
        },
        ResponseTone.ENCOURAGING: {
            "opener": "You're doing the right thing by asking about {topic}.",
            "transition": "Here's some good news:",
            "closer": "Every step toward understanding helps you make confident decisions."
        }
    }
    
    # Mapping trust levels to tones
    TRUST_TO_TONE = {
        TrustLevel.HIGH: ResponseTone.INFORMATIVE,
        TrustLevel.MODERATE: ResponseTone.INFORMATIVE,
        TrustLevel.LOW: ResponseTone.EMPATHETIC,
        TrustLevel.VERY_LOW: ResponseTone.EMPATHETIC
    }
    
    # Mapping styles to formats
    STYLE_TO_FORMAT = {
        InformationStyle.SCIENTIFIC: ResponseFormat.DETAILED,
        InformationStyle.STORYTELLING: ResponseFormat.SUMMARY,
        InformationStyle.VISUAL: ResponseFormat.BULLET_POINTS,
        InformationStyle.CONVERSATIONAL: ResponseFormat.QA_STYLE,
        InformationStyle.AUTHORITATIVE: ResponseFormat.DETAILED,
        InformationStyle.PEER: ResponseFormat.SUMMARY
    }
    
    # Counter-messages for misinformation categories
    COUNTER_MESSAGES = {
        MisinfoCategory.VACCINE_SAFETY: {
            "headline": "Understanding Vaccine Safety",
            "key_points": [
                "Vaccines go through rigorous multi-phase clinical trials",
                "Side effects are tracked continuously after approval",
                "Serious side effects are extremely rare compared to disease risks",
                "VAERS data is monitored to detect any safety signals"
            ]
        },
        MisinfoCategory.INGREDIENT_FEAR: {
            "headline": "What's Really in Vaccines",
            "key_points": [
                "Every ingredient serves a specific, safe purpose",
                "Amounts are far below levels that could cause harm",
                "Common ingredients like aluminum have been used safely for decades",
                "Preservatives like thimerosal are removed from most vaccines"
            ]
        },
        MisinfoCategory.CONSPIRACY: {
            "headline": "Separating Fact from Fiction",
            "key_points": [
                "Vaccine research involves thousands of independent scientists",
                "Data is publicly available for scrutiny",
                "Multiple competing organizations reach the same conclusions",
                "Consider the source and motivations of health claims"
            ]
        },
        MisinfoCategory.RUSHED_DEVELOPMENT: {
            "headline": "The Speed of Vaccine Development",
            "key_points": [
                "mRNA technology has been researched for over 30 years",
                "Previous coronavirus research accelerated COVID vaccine development",
                "Regulatory steps were overlapped, not skipped",
                "Massive funding and global collaboration sped the process"
            ]
        },
        MisinfoCategory.NATURAL_IMMUNITY: {
            "headline": "Natural vs. Vaccine Immunity",
            "key_points": [
                "Both provide protection, but vaccines are safer",
                "Natural immunity requires getting sick first with real risks",
                "Vaccine immunity is more consistent and predictable",
                "Hybrid immunity (infection + vaccine) provides strong protection"
            ]
        },
        MisinfoCategory.SIDE_EFFECTS: {
            "headline": "Understanding Side Effects",
            "key_points": [
                "Most side effects are mild and short-lived",
                "Side effects often indicate your immune system is responding",
                "Serious reactions are very rare and closely monitored",
                "Disease complications are far more dangerous than vaccine side effects"
            ]
        },
        MisinfoCategory.VACCINE_EFFICACY: {
            "headline": "How Vaccine Effectiveness Works",
            "key_points": [
                "Vaccines significantly reduce severe illness and death",
                "Effectiveness varies by disease, variant, and individual",
                "Boosters help maintain protection over time",
                "Even partial protection saves lives at population level"
            ]
        },
        MisinfoCategory.DISTRUST_AUTHORITY: {
            "headline": "Making Your Own Informed Decision",
            "key_points": [
                "You have the right to understand what you're putting in your body",
                "Seek information from multiple independent sources",
                "Talk to healthcare providers you personally trust",
                "Focus on data and outcomes, not institutions"
            ]
        }
    }
    
    def __init__(self, content_matcher: Optional[ContentMatcher] = None):
        self.content_matcher = content_matcher or ContentMatcher()
        self._response_counter = 0
    
    def _generate_id(self) -> str:
        """Generate unique response ID."""
        self._response_counter += 1
        return f"R{datetime.now().strftime('%Y%m%d')}{self._response_counter:04d}"
    
    def _select_tone(self, user_profile: UserProfile) -> ResponseTone:
        """Select appropriate tone based on user profile."""
        return self.TRUST_TO_TONE.get(
            user_profile.trust_level,
            ResponseTone.INFORMATIVE
        )
    
    def _select_format(self, user_profile: UserProfile) -> ResponseFormat:
        """Select response format based on user style."""
        return self.STYLE_TO_FORMAT.get(
            user_profile.preferred_style,
            ResponseFormat.SUMMARY
        )
    
    def _build_main_message(
        self,
        topic: str,
        tone: ResponseTone,
        misinfo_category: Optional[MisinfoCategory]
    ) -> str:
        """Build the main message using templates."""
        templates = self.TONE_TEMPLATES[tone]
        
        opener = templates["opener"].format(topic=topic)
        
        if misinfo_category and misinfo_category in self.COUNTER_MESSAGES:
            counter = self.COUNTER_MESSAGES[misinfo_category]
            transition = templates["transition"]
            points = counter["key_points"][:2]  # Take first 2 points
            message = f"{opener} {transition} {' '.join(points)}"
        else:
            message = f"{opener} Let's explore the facts together."
        
        return message
    
    def _get_supporting_points(
        self,
        misinfo_category: Optional[MisinfoCategory],
        user_concerns: List[ConcernType]
    ) -> List[str]:
        """Get supporting points based on misinformation and concerns."""
        points = []
        
        # Add counter-message points
        if misinfo_category and misinfo_category in self.COUNTER_MESSAGES:
            points.extend(self.COUNTER_MESSAGES[misinfo_category]["key_points"])
        
        # Add concern-specific points
        for concern in user_concerns[:2]:
            if concern == ConcernType.SAFETY:
                points.append("Vaccine safety is monitored by multiple independent systems")
            elif concern == ConcernType.EFFICACY:
                points.append("Effectiveness data is continuously updated based on real-world evidence")
            elif concern == ConcernType.TRUST:
                points.append("You can verify claims through independent, peer-reviewed research")
        
        return list(set(points))[:5]  # Unique, max 5
    
    def _get_follow_ups(self, user_profile: UserProfile) -> List[str]:
        """Generate relevant follow-up questions."""
        questions = []
        
        for concern in user_profile.primary_concerns:
            if concern == ConcernType.SAFETY:
                questions.append("Would you like to know more about how side effects are monitored?")
            elif concern == ConcernType.INGREDIENTS:
                questions.append("Should I explain what specific ingredients do?")
            elif concern == ConcernType.EFFICACY:
                questions.append("Want to see the latest effectiveness data?")
            elif concern == ConcernType.DEVELOPMENT_SPEED:
                questions.append("Curious about how the development process worked?")
        
        return questions[:3]
    
    def _get_next_step(
        self,
        user_profile: UserProfile,
        misinfo_category: Optional[MisinfoCategory]
    ) -> str:
        """Suggest a next step for the user."""
        if user_profile.trust_level in [TrustLevel.VERY_LOW, TrustLevel.LOW]:
            return "Consider talking to a healthcare provider you trust about your specific situation."
        elif misinfo_category:
            return "Explore the recommended resources below for more detailed information."
        else:
            return "Feel free to ask any follow-up questions you might have."
    
    def _estimate_read_time(self, response: PersonalizedResponse) -> int:
        """Estimate reading time in seconds (assuming ~200 words/minute)."""
        text = response.main_message + " ".join(response.supporting_points)
        words = len(text.split())
        return int((words / 200) * 60)
    
    def generate_response(
        self,
        user_profile: UserProfile,
        misinfo_result: Optional[ClassificationResult] = None,
        context: str = "",
        include_content: bool = True
    ) -> PersonalizedResponse:
        """
        Generate a personalized response for a user.
        
        Args:
            user_profile: The user's profile
            misinfo_result: Classification result if misinformation was detected
            context: Additional context about the user's query
            include_content: Whether to include content recommendations
        
        Returns:
            PersonalizedResponse tailored to the user
        """
        misinfo_category = misinfo_result.primary_category if misinfo_result else None
        
        # Determine tone and format
        tone = self._select_tone(user_profile)
        format_ = self._select_format(user_profile)
        
        # Build headline
        if misinfo_category and misinfo_category in self.COUNTER_MESSAGES:
            headline = self.COUNTER_MESSAGES[misinfo_category]["headline"]
        else:
            headline = "Understanding Vaccine Information"
        
        # Extract topic from context or concerns
        topic = context if context else "vaccines"
        if user_profile.primary_concerns:
            concern_str = user_profile.primary_concerns[0].value
            topic = f"vaccine {concern_str}"
        
        # Build main message
        main_message = self._build_main_message(topic, tone, misinfo_category)
        
        # Get supporting points
        supporting_points = self._get_supporting_points(
            misinfo_category,
            user_profile.primary_concerns
        )
        
        # Get content recommendations
        content_ids = []
        source_citations = []
        if include_content and self.content_matcher.content_library:
            matches = self.content_matcher.match_for_user(
                user_profile,
                misinfo_category=misinfo_category,
                top_n=3
            )
            for match in matches:
                content_ids.append(match.content.content_id)
                source_citations.append(match.content.source)
        
        # Create response
        response = PersonalizedResponse(
            response_id=self._generate_id(),
            user_id=user_profile.user_id,
            headline=headline,
            main_message=main_message,
            supporting_points=supporting_points,
            tone=tone,
            format=format_,
            difficulty_level="basic" if user_profile.trust_level in [TrustLevel.VERY_LOW, TrustLevel.LOW] else "intermediate",
            content_recommendations=content_ids,
            source_citations=list(set(source_citations)),
            next_step=self._get_next_step(user_profile, misinfo_category),
            follow_up_questions=self._get_follow_ups(user_profile),
            targeting_reason=f"Based on your {user_profile.preferred_style.value} information preference"
        )
        
        response.estimated_read_time = self._estimate_read_time(response)
        
        logger.info(f"Generated response {response.response_id} for user {user_profile.user_id}")
        return response
    
    def adapt_content(
        self,
        content: EducationalContent,
        user_profile: UserProfile
    ) -> str:
        """
        Adapt existing content to match user's style preferences.
        
        Returns simplified/modified version of content text.
        """
        text = content.summary
        
        # Adjust based on trust level
        if user_profile.trust_level == TrustLevel.VERY_LOW:
            # Remove authoritative language
            text = text.replace("studies show", "some research suggests")
            text = text.replace("experts say", "some healthcare providers believe")
        
        # Adjust based on style
        if user_profile.preferred_style == InformationStyle.CONVERSATIONAL:
            # Make more conversational
            text = f"Here's what you might want to know: {text}"
        elif user_profile.preferred_style == InformationStyle.SCIENTIFIC:
            # Keep technical
            pass
        
        return text
    
    def generate_batch_responses(
        self,
        profiles: List[UserProfile],
        misinfo_category: Optional[MisinfoCategory] = None
    ) -> List[PersonalizedResponse]:
        """Generate responses for multiple users."""
        responses = []
        
        for profile in profiles:
            # Create mock classification result if category specified
            misinfo_result = None
            if misinfo_category:
                misinfo_result = ClassificationResult(
                    category=misinfo_category,
                    confidence=0.8,
                    severity="medium",
                    key_phrases=[],
                    explanation="Batch processing",
                    counter_topics=[]
                )
            
            response = self.generate_response(profile, misinfo_result)
            responses.append(response)
        
        return responses
    
    def save_response(self, response: PersonalizedResponse, filepath: str):
        """Save response to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(response.to_dict(), f, indent=2, default=str)
    
    def export_responses(
        self,
        responses: List[PersonalizedResponse]
    ) -> pd.DataFrame:
        """Export responses to DataFrame for analysis."""
        records = []
        for r in responses:
            records.append({
                "response_id": r.response_id,
                "user_id": r.user_id,
                "headline": r.headline,
                "tone": r.tone.value,
                "format": r.format.value,
                "num_recommendations": len(r.content_recommendations),
                "read_time_seconds": r.estimated_read_time,
                "created_at": r.created_at
            })
        return pd.DataFrame(records)
