"""
RiboReach - AI-Powered Misinformation Counter & Personalized Education

A platform to improve public trust by:
- Detecting and classifying health misinformation
- Profiling audience concerns and beliefs
- Matching personalized educational content
- Tracking trust and engagement metrics
"""

__version__ = "0.1.0"
__author__ = "AI Horizons"

from dynamic_classifier import (
    DynamicMisinformationClassifier,
    MisinfoCategory,
    PatternLoader,
    EmbeddingMatcher,
    MLClassifier
)
# Backwards compatibility alias
MisinformationClassifier = DynamicMisinformationClassifier

from user_profiler import UserProfiler, UserProfile, ConcernType, TrustLevel
from content_matcher import ContentMatcher, EducationalContent
from personalization_engine import PersonalizationEngine, PersonalizedResponse
from trust_metrics import TrustMetricsTracker, EngagementType

__all__ = [
    "DynamicMisinformationClassifier",
    "MisinformationClassifier",  # Backwards compat
    "MisinfoCategory",
    "PatternLoader",
    "EmbeddingMatcher", 
    "MLClassifier",
    "UserProfiler", 
    "UserProfile",
    "ConcernType",
    "TrustLevel",
    "ContentMatcher",
    "EducationalContent",
    "PersonalizationEngine",
    "PersonalizedResponse",
    "TrustMetricsTracker",
    "EngagementType",
]
