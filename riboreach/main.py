#!/usr/bin/env python3
"""
RiboReach - AI-Powered Vaccine Misinformation Counter-Education Platform

Main entry point for the RiboReach system that improves public trust
by targeting misinformation with personalized education.

Usage:
    python main.py --demo               # Run interactive demo
    python main.py --classify TEXT      # Classify misinformation in text
    python main.py --profile            # Create user profile interactively
    python main.py --metrics            # View platform metrics
"""

import argparse
import logging
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from dynamic_classifier import (
    DynamicMisinformationClassifier,
    MisinfoCategory,
    generate_training_data
)
from user_profiler import (
    UserProfiler,
    UserProfile,
    ConcernType,
    InformationStyle,
    TrustLevel
)
from content_matcher import (
    ContentMatcher,
    EducationalContent,
    ContentMatch,
    generate_sample_content
)
from personalization_engine import (
    PersonalizationEngine,
    PersonalizedResponse
)
from trust_metrics import (
    TrustMetricsTracker,
    EngagementType,
    generate_sample_events
)
from dynamic_responder import DynamicResponder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RiboReach:
    """
    Main RiboReach platform coordinator.
    
    Integrates all modules to provide personalized counter-misinformation
    education to improve public health trust.
    """
    
    def __init__(self, load_sample_data: bool = True):
        """Initialize RiboReach with all components."""
        print("Initializing RiboReach Platform...")
        
        # Initialize components
        self.classifier = DynamicMisinformationClassifier(use_embeddings=True)
        self.profiler = UserProfiler()
        self.content_matcher = ContentMatcher()
        self.personalization = PersonalizationEngine(self.content_matcher)
        self.metrics = TrustMetricsTracker()
        self.dynamic_responder = DynamicResponder(self.classifier)
        
        # Load sample data if requested
        if load_sample_data:
            self._load_sample_data()
        
        print("RiboReach Platform Ready!")
    
    def _load_sample_data(self):
        """Load sample content and train classifier."""
        # Load sample content
        content_df = generate_sample_content()
        self.content_matcher.load_content_library(content_df)
        print(f"  - Loaded {len(self.content_matcher.content_library)} content items")
        
        # Train classifier on sample misinformation
        claims_df = generate_training_data()
        if not claims_df.empty and len(claims_df) > 5:
            metrics = self.classifier.train(claims_df)
            stats = self.classifier.get_stats()
            print(f"  - Trained classifier on {len(claims_df)} samples")
            print(f"  - {stats['num_categories']} categories, {stats['total_patterns']} patterns")
            if stats['embeddings_available']:
                print(f"  - Semantic embeddings enabled")
    
    def analyze_text(self, text: str) -> dict:
        """
        Analyze text for misinformation.
        
        Returns:
            Dictionary with classification results
        """
        result = self.classifier.classify(text)
        return {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "category": result.primary_category.value,
            "confidence": round(result.confidence, 2),
            "severity": result.severity,
            "key_phrases": result.detected_claims,
            "counter_topics": [result.primary_category.value],
            "explanation": f"Classified as {result.primary_category.value} with {result.confidence:.0%} confidence"
        }
    
    def create_user_profile(
        self,
        user_id: str,
        text_input: str = "",
        demographic_data: dict = None
    ) -> UserProfile:
        """Create a user profile from interaction data."""
        if text_input:
            profile = self.profiler.analyze_and_create_profile(
                user_id=user_id,
                texts=[text_input]
            )
        else:
            # Create with defaults, augment with demographics
            profile = UserProfile(
                user_id=user_id,
                trust_level=TrustLevel.MODERATE,
                primary_concerns=[ConcernType.SAFETY],
                preferred_style=InformationStyle.CONVERSATIONAL
            )
        
        return profile
    
    def get_personalized_response(
        self,
        user_profile: UserProfile,
        text: str = ""
    ) -> PersonalizedResponse:
        """Generate personalized educational response."""
        # Classify the text if provided
        misinfo_result = None
        if text:
            misinfo_result = self.classifier.classify(text)
        
        # Generate personalized response
        response = self.personalization.generate_response(
            user_profile=user_profile,
            misinfo_result=misinfo_result,
            context=text
        )
        
        return response
    
    def get_content_recommendations(
        self,
        user_profile: UserProfile,
        misinfo_category: MisinfoCategory = None,
        top_n: int = 3
    ) -> list:
        """Get content recommendations for a user."""
        matches = self.content_matcher.match_for_user(
            user_profile,
            misinfo_category=misinfo_category,
            top_n=top_n
        )
        
        return [
            {
                "title": m.content.title,
                "summary": m.content.summary,
                "type": m.content.content_type.value,
                "source": m.content.source,
                "relevance": round(m.overall_score, 2),
                "reasons": m.match_reasons
            }
            for m in matches
        ]
    
    def log_engagement(
        self,
        user_id: str,
        content_id: str,
        event_type: str,
        time_spent: int = 0
    ):
        """Log user engagement event."""
        self.metrics.log_event(
            user_id=user_id,
            content_id=content_id,
            event_type=EngagementType(event_type),
            time_spent_seconds=time_spent
        )
    
    def get_platform_metrics(self) -> dict:
        """Get overall platform performance metrics."""
        return self.metrics.calculate_platform_metrics()
    
    def get_content_effectiveness(self) -> pd.DataFrame:
        """Get content effectiveness report."""
        return self.metrics.generate_effectiveness_report()
    
    # Dynamic classifier methods
    def add_misinfo_pattern(self, category: str, pattern: str):
        """Add a new regex pattern to detect misinformation."""
        self.classifier.add_pattern(category, pattern)
        print(f"Added pattern to {category}")
    
    def add_misinfo_keyword(self, category: str, keyword: str):
        """Add a new keyword to detect misinformation."""
        self.classifier.add_keyword(category, keyword)
        print(f"Added keyword '{keyword}' to {category}")
    
    def flag_misinformation(self, text: str, category: str):
        """Flag content as misinformation for learning."""
        self.classifier.flag_as_misinfo(text, category)
        print(f"Flagged as {category} - will be used for training")
    
    def flag_factual(self, text: str):
        """Flag content as factual (not misinformation)."""
        self.classifier.flag_as_factual(text)
        print("Flagged as factual")
    
    def retrain_classifier(self):
        """Retrain classifier with accumulated flagged content."""
        metrics = self.classifier.retrain()
        print(f"Retrained classifier: {metrics}")
        return metrics
    
    def reload_patterns(self):
        """Reload patterns from configuration file."""
        self.classifier.reload_config()
        print("Reloaded patterns from config")
    
    def get_classifier_stats(self) -> dict:
        """Get classifier statistics."""
        return self.classifier.get_stats()
    
    def get_dynamic_response(self, text: str):
        """
        Generate a dynamic, contextual response based on actual user input.
        This creates responses tailored to what the user specifically said,
        not from a hardcoded content library.
        """
        return self.dynamic_responder.generate_response(text)

def run_interactive_demo():
    """Run an interactive demonstration of RiboReach capabilities."""
    print("\n" + "="*60)
    print("        RIBOREACH INTERACTIVE DEMO")
    print("   AI-Powered Vaccine Education Platform")
    print("="*60 + "\n")
    
    # Initialize platform
    riboreach = RiboReach(load_sample_data=True)
    
    print("\n--- STEP 1: Create Your Profile ---\n")
    
    # Get user input for profile
    user_id = "DEMO_USER"
    print("Please share your thoughts or concerns about vaccines.")
    print("(This helps us personalize information for you)")
    print("-" * 40)
    
    try:
        user_text = input("\nYour thoughts: ").strip()
        if not user_text:
            user_text = "I'm worried about vaccine side effects and the speed of development"
    except EOFError:
        user_text = "I'm worried about vaccine side effects and the speed of development"
        print(f"Using default: {user_text}")
    
    # Create profile
    profile = riboreach.create_user_profile(user_id, user_text)
    
    print("\n--- Your Profile ---")
    print(f"Identified Trust Level: {profile.trust_level.value.title()}")
    print(f"Primary Concerns: {', '.join(c.value for c in profile.primary_concerns)}")
    print(f"Preferred Information Style: {profile.preferred_style.value.title()}")
    
    print("\n--- STEP 2: Dynamic Response to Your Input ---\n")
    
    # Generate dynamic response based on EXACTLY what the user typed
    dynamic_response = riboreach.get_dynamic_response(user_text)
    print(dynamic_response.to_text())
    
    print("\n--- STEP 3: Try Another Input ---\n")
    
    # Let user try more inputs
    print("Enter any vaccine-related text to see how the system responds dynamically.")
    print("(Press Enter with no text to skip)")
    print("-" * 40)
    
    try:
        extra_input = input("\nYour input: ").strip()
        if extra_input:
            extra_response = riboreach.get_dynamic_response(extra_input)
            print(extra_response.to_text())
    except EOFError:
        pass
    
    print("\n" + "="*60)
    print("Demo Complete! RiboReach personalizes education to build trust.")
    print("="*60 + "\n")


def classify_text(text: str):
    """Classify misinformation in provided text."""
    riboreach = RiboReach(load_sample_data=True)
    
    print("\n--- Misinformation Analysis ---\n")
    analysis = riboreach.analyze_text(text)
    
    print(f"Text: {analysis['text']}")
    print(f"Category: {analysis['category']}")
    print(f"Confidence: {analysis['confidence']}")
    print(f"Severity: {analysis['severity']}")
    print(f"Key Phrases: {', '.join(analysis['key_phrases'])}")
    print(f"Explanation: {analysis['explanation']}")
    print(f"\nCounter Topics to Address:")
    for topic in analysis['counter_topics']:
        print(f"  • {topic}")


def interactive_profile():
    """Create user profile interactively."""
    riboreach = RiboReach(load_sample_data=True)
    
    print("\n--- User Profiling ---\n")
    print("Answer a few questions to help us understand your perspective.\n")
    
    try:
        text = input("What concerns do you have about vaccines? ")
    except EOFError:
        text = "I'm not sure vaccines are safe"
    
    user_id = f"USER_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    profile = riboreach.create_user_profile(user_id, text)
    
    print("\n--- Profile Results ---")
    print(f"User ID: {profile.user_id}")
    print(f"Trust Level: {profile.trust_level.value}")
    print(f"Primary Concerns: {[c.value for c in profile.primary_concerns]}")
    print(f"Preferred Style: {profile.preferred_style.value}")
    
    # Get personalized content
    print("\n--- Recommended Content ---")
    recommendations = riboreach.get_content_recommendations(profile, top_n=3)
    for rec in recommendations:
        print(f"  • {rec['title']} ({rec['type']})")


def show_metrics():
    """Display platform metrics."""
    riboreach = RiboReach(load_sample_data=True)
    
    # Add some sample events for demo
    sample_events = generate_sample_events()
    for event in sample_events:
        riboreach.metrics.log_event(
            user_id=event["user_id"],
            content_id=event["content_id"],
            event_type=EngagementType(event["event_type"]),
            time_spent_seconds=event["time_spent_seconds"],
            scroll_depth_percent=event["scroll_depth_percent"]
        )
    
    print("\n--- Platform Metrics ---\n")
    metrics = riboreach.get_platform_metrics()
    
    print(f"Total Users: {metrics.get('total_users', 0)}")
    print(f"Total Events: {metrics.get('total_events', 0)}")
    print(f"Completion Rate: {metrics.get('overall_completion_rate', 0):.1%}")
    print(f"Trust Improvement Rate: {metrics.get('trust_improvement_rate', 0):.1%}")
    
    print("\n--- Events by Type ---")
    for event_type, count in metrics.get('events_by_type', {}).items():
        print(f"  {event_type}: {count}")
    
    print("\n--- Content Effectiveness ---")
    report = riboreach.get_content_effectiveness()
    if not report.empty:
        print(report[["content_id", "total_views", "completion_rate", "effectiveness_score"]].to_string(index=False))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="RiboReach - AI Vaccine Education Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo
  python main.py --classify "Vaccines cause autism"
  python main.py --profile
  python main.py --metrics
        """
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run interactive demonstration"
    )
    parser.add_argument(
        "--classify",
        type=str,
        metavar="TEXT",
        help="Classify misinformation in text"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Create user profile interactively"
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Show platform metrics"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.demo:
        run_interactive_demo()
    elif args.classify:
        classify_text(args.classify)
    elif args.profile:
        interactive_profile()
    elif args.metrics:
        show_metrics()
    else:
        # Default: show help and run brief demo
        parser.print_help()
        print("\n" + "-"*40)
        print("Running quick demo...")
        print("-"*40)
        
        riboreach = RiboReach(load_sample_data=True)
        
        # Quick demo
        print("\n[1] Analyzing: 'Vaccines haven't been tested properly'")
        result = riboreach.analyze_text("Vaccines haven't been tested properly")
        print(f"    Category: {result['category']} | Severity: {result['severity']}")
        
        print("\n[2] Creating profile for concerned user...")
        profile = riboreach.create_user_profile(
            "demo", 
            "I'm worried about giving vaccines to my children"
        )
        print(f"    Trust: {profile.trust_level.value} | Style: {profile.preferred_style.value}")
        
        print("\n[3] Top recommendation:")
        recs = riboreach.get_content_recommendations(profile, top_n=1)
        if recs:
            print(f"    {recs[0]['title']}")
        
        print("\nRun 'python main.py --demo' for full interactive experience.\n")


if __name__ == "__main__":
    main()
