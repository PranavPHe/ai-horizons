"""
Trust Metrics Tracker for RiboReach

Tracks user engagement, trust changes, and effectiveness metrics
to measure the impact of personalized education.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta
import logging
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from user_profiler import UserProfile, TrustLevel

logger = logging.getLogger(__name__)


class EngagementType(Enum):
    """Types of user engagement."""
    VIEW = "view"                 # Viewed content
    CLICK = "click"               # Clicked to learn more
    SHARE = "share"               # Shared content
    COMPLETE = "complete"         # Completed reading/watching
    BOOKMARK = "bookmark"         # Saved for later
    DISMISS = "dismiss"           # Dismissed content
    NEGATIVE_FEEDBACK = "negative_feedback"  # Reported as unhelpful
    POSITIVE_FEEDBACK = "positive_feedback"  # Marked as helpful
    FOLLOW_UP = "follow_up"       # Asked a follow-up question


class TrustChange(Enum):
    """Direction of trust change."""
    INCREASED = "increased"
    DECREASED = "decreased"
    UNCHANGED = "unchanged"
    UNKNOWN = "unknown"


@dataclass
class EngagementEvent:
    """Single engagement event."""
    event_id: str
    user_id: str
    content_id: str
    event_type: EngagementType
    timestamp: datetime
    
    # Optional context
    session_id: Optional[str] = None
    response_id: Optional[str] = None
    time_spent_seconds: int = 0
    scroll_depth_percent: float = 0.0
    
    # Sentiment if available
    feedback_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "user_id": self.user_id,
            "content_id": self.content_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "time_spent_seconds": self.time_spent_seconds,
            "scroll_depth_percent": self.scroll_depth_percent
        }


@dataclass
class TrustMeasurement:
    """Trust level measurement at a point in time."""
    user_id: str
    timestamp: datetime
    trust_level: TrustLevel
    trust_score: float  # 0-1 continuous score
    
    # What prompted this measurement
    trigger: str  # "survey", "behavior", "interaction"
    
    # Optional survey responses
    survey_responses: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserMetrics:
    """Aggregated metrics for a user."""
    user_id: str
    
    # Engagement metrics
    total_views: int = 0
    total_shares: int = 0
    total_time_spent: int = 0  # seconds
    avg_scroll_depth: float = 0.0
    completion_rate: float = 0.0
    
    # Trust journey
    initial_trust_level: Optional[TrustLevel] = None
    current_trust_level: Optional[TrustLevel] = None
    trust_measurements: List[TrustMeasurement] = field(default_factory=list)
    
    # Content preferences
    preferred_content_ids: List[str] = field(default_factory=list)
    dismissed_content_ids: List[str] = field(default_factory=list)
    
    # Timestamps
    first_interaction: Optional[datetime] = None
    last_interaction: Optional[datetime] = None
    
    @property
    def trust_changed(self) -> TrustChange:
        """Determine if trust level changed."""
        if not self.initial_trust_level or not self.current_trust_level:
            return TrustChange.UNKNOWN
        
        levels = list(TrustLevel)
        initial_idx = levels.index(self.initial_trust_level)
        current_idx = levels.index(self.current_trust_level)
        
        if current_idx < initial_idx:  # HIGH is 0, VERY_LOW is 3
            return TrustChange.INCREASED
        elif current_idx > initial_idx:
            return TrustChange.DECREASED
        else:
            return TrustChange.UNCHANGED
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "total_views": self.total_views,
            "total_shares": self.total_shares,
            "total_time_spent_minutes": self.total_time_spent // 60,
            "completion_rate": round(self.completion_rate, 2),
            "initial_trust": self.initial_trust_level.value if self.initial_trust_level else None,
            "current_trust": self.current_trust_level.value if self.current_trust_level else None,
            "trust_change": self.trust_changed.value
        }


@dataclass
class ContentMetrics:
    """Aggregated metrics for a content item."""
    content_id: str
    
    # Engagement
    total_views: int = 0
    unique_viewers: int = 0
    total_shares: int = 0
    total_completions: int = 0
    
    # Averages
    avg_time_spent: float = 0.0
    avg_scroll_depth: float = 0.0
    
    # Feedback
    positive_feedback_count: int = 0
    negative_feedback_count: int = 0
    
    # Trust impact
    users_with_increased_trust: int = 0
    users_with_decreased_trust: int = 0
    
    @property
    def completion_rate(self) -> float:
        if self.total_views == 0:
            return 0.0
        return self.total_completions / self.total_views
    
    @property
    def effectiveness_score(self) -> float:
        """Calculate effectiveness based on engagement and trust changes."""
        if self.total_views == 0:
            return 0.5
        
        # Engagement component (0-0.5)
        engagement = min(0.5, (
            (self.completion_rate * 0.2) +
            (min(1.0, self.avg_scroll_depth) * 0.15) +
            (min(0.15, (self.total_shares / max(1, self.total_views))))
        ))
        
        # Feedback component (0-0.3)
        total_feedback = self.positive_feedback_count + self.negative_feedback_count
        if total_feedback > 0:
            feedback_score = self.positive_feedback_count / total_feedback * 0.3
        else:
            feedback_score = 0.15  # Neutral default
        
        # Trust impact component (0-0.2)
        trust_impact_users = self.users_with_increased_trust + self.users_with_decreased_trust
        if trust_impact_users > 0:
            trust_score = self.users_with_increased_trust / trust_impact_users * 0.2
        else:
            trust_score = 0.1  # Neutral default
        
        return round(engagement + feedback_score + trust_score, 3)


class TrustMetricsTracker:
    """
    Tracks and analyzes engagement and trust metrics across the platform.
    
    Example usage:
    ```python
    tracker = TrustMetricsTracker()
    
    # Log engagement events
    tracker.log_event(event)
    
    # Record trust measurements
    tracker.record_trust_measurement(user_id, trust_level, "survey")
    
    # Get metrics
    user_metrics = tracker.get_user_metrics(user_id)
    content_metrics = tracker.get_content_metrics(content_id)
    
    # Generate reports
    report = tracker.generate_effectiveness_report()
    ```
    """
    
    def __init__(self):
        self.events: List[EngagementEvent] = []
        self.trust_measurements: Dict[str, List[TrustMeasurement]] = defaultdict(list)
        self._event_counter = 0
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        self._event_counter += 1
        return f"E{datetime.now().strftime('%Y%m%d%H%M%S')}{self._event_counter:04d}"
    
    def log_event(
        self,
        user_id: str,
        content_id: str,
        event_type: EngagementType,
        time_spent_seconds: int = 0,
        scroll_depth_percent: float = 0.0,
        session_id: Optional[str] = None,
        response_id: Optional[str] = None,
        feedback_text: Optional[str] = None
    ) -> EngagementEvent:
        """Log an engagement event."""
        event = EngagementEvent(
            event_id=self._generate_event_id(),
            user_id=user_id,
            content_id=content_id,
            event_type=event_type,
            timestamp=datetime.now(),
            session_id=session_id,
            response_id=response_id,
            time_spent_seconds=time_spent_seconds,
            scroll_depth_percent=scroll_depth_percent,
            feedback_text=feedback_text
        )
        self.events.append(event)
        logger.debug(f"Logged event {event.event_id}: {event_type.value} for user {user_id}")
        return event
    
    def record_trust_measurement(
        self,
        user_id: str,
        trust_level: TrustLevel,
        trigger: str = "behavior",
        trust_score: Optional[float] = None,
        survey_responses: Optional[Dict[str, Any]] = None
    ) -> TrustMeasurement:
        """Record a trust level measurement for a user."""
        # Convert trust level to score if not provided
        if trust_score is None:
            trust_scores = {
                TrustLevel.HIGH: 0.9,
                TrustLevel.MODERATE: 0.6,
                TrustLevel.LOW: 0.35,
                TrustLevel.VERY_LOW: 0.15
            }
            trust_score = trust_scores.get(trust_level, 0.5)
        
        measurement = TrustMeasurement(
            user_id=user_id,
            timestamp=datetime.now(),
            trust_level=trust_level,
            trust_score=trust_score,
            trigger=trigger,
            survey_responses=survey_responses or {}
        )
        
        self.trust_measurements[user_id].append(measurement)
        logger.info(f"Recorded trust measurement for {user_id}: {trust_level.value}")
        return measurement
    
    def get_user_metrics(self, user_id: str) -> UserMetrics:
        """Calculate metrics for a specific user."""
        user_events = [e for e in self.events if e.user_id == user_id]
        
        if not user_events:
            return UserMetrics(user_id=user_id)
        
        metrics = UserMetrics(user_id=user_id)
        
        # Count events by type
        views = [e for e in user_events if e.event_type == EngagementType.VIEW]
        shares = [e for e in user_events if e.event_type == EngagementType.SHARE]
        completions = [e for e in user_events if e.event_type == EngagementType.COMPLETE]
        
        metrics.total_views = len(views)
        metrics.total_shares = len(shares)
        metrics.total_time_spent = sum(e.time_spent_seconds for e in user_events)
        
        # Calculate averages
        if views:
            metrics.avg_scroll_depth = np.mean([e.scroll_depth_percent for e in views])
        
        if metrics.total_views > 0:
            metrics.completion_rate = len(completions) / metrics.total_views
        
        # Content preferences
        positive_events = [e for e in user_events 
                         if e.event_type in [EngagementType.SHARE, EngagementType.COMPLETE, 
                                            EngagementType.POSITIVE_FEEDBACK, EngagementType.BOOKMARK]]
        metrics.preferred_content_ids = list(set(e.content_id for e in positive_events))
        
        dismissed = [e for e in user_events if e.event_type == EngagementType.DISMISS]
        metrics.dismissed_content_ids = list(set(e.content_id for e in dismissed))
        
        # Timestamps
        timestamps = [e.timestamp for e in user_events]
        metrics.first_interaction = min(timestamps)
        metrics.last_interaction = max(timestamps)
        
        # Trust measurements
        if user_id in self.trust_measurements:
            measurements = sorted(self.trust_measurements[user_id], key=lambda x: x.timestamp)
            metrics.trust_measurements = measurements
            if measurements:
                metrics.initial_trust_level = measurements[0].trust_level
                metrics.current_trust_level = measurements[-1].trust_level
        
        return metrics
    
    def get_content_metrics(self, content_id: str) -> ContentMetrics:
        """Calculate metrics for content item."""
        content_events = [e for e in self.events if e.content_id == content_id]
        
        if not content_events:
            return ContentMetrics(content_id=content_id)
        
        metrics = ContentMetrics(content_id=content_id)
        
        # Count by type
        metrics.total_views = len([e for e in content_events if e.event_type == EngagementType.VIEW])
        metrics.unique_viewers = len(set(e.user_id for e in content_events if e.event_type == EngagementType.VIEW))
        metrics.total_shares = len([e for e in content_events if e.event_type == EngagementType.SHARE])
        metrics.total_completions = len([e for e in content_events if e.event_type == EngagementType.COMPLETE])
        
        # Averages
        views = [e for e in content_events if e.event_type == EngagementType.VIEW]
        if views:
            metrics.avg_time_spent = np.mean([e.time_spent_seconds for e in views])
            metrics.avg_scroll_depth = np.mean([e.scroll_depth_percent for e in views])
        
        # Feedback
        metrics.positive_feedback_count = len([e for e in content_events 
                                               if e.event_type == EngagementType.POSITIVE_FEEDBACK])
        metrics.negative_feedback_count = len([e for e in content_events 
                                               if e.event_type == EngagementType.NEGATIVE_FEEDBACK])
        
        # Trust impact - compare users' trust before/after viewing this content
        viewers = set(e.user_id for e in content_events)
        for user_id in viewers:
            user_metrics = self.get_user_metrics(user_id)
            if user_metrics.trust_changed == TrustChange.INCREASED:
                metrics.users_with_increased_trust += 1
            elif user_metrics.trust_changed == TrustChange.DECREASED:
                metrics.users_with_decreased_trust += 1
        
        return metrics
    
    def get_trust_trend(
        self,
        user_id: str,
        days: int = 30
    ) -> List[Tuple[datetime, float]]:
        """Get trust score trend for a user over time."""
        if user_id not in self.trust_measurements:
            return []
        
        cutoff = datetime.now() - timedelta(days=days)
        recent = [m for m in self.trust_measurements[user_id] if m.timestamp > cutoff]
        
        return [(m.timestamp, m.trust_score) for m in sorted(recent, key=lambda x: x.timestamp)]
    
    def calculate_platform_metrics(self) -> Dict[str, Any]:
        """Calculate overall platform metrics."""
        if not self.events:
            return {}
        
        # Unique users
        all_users = set(e.user_id for e in self.events)
        
        # Event counts
        event_counts = defaultdict(int)
        for e in self.events:
            event_counts[e.event_type.value] += 1
        
        # Trust changes
        trust_changes = {
            TrustChange.INCREASED: 0,
            TrustChange.DECREASED: 0,
            TrustChange.UNCHANGED: 0,
            TrustChange.UNKNOWN: 0
        }
        
        for user_id in all_users:
            user_metrics = self.get_user_metrics(user_id)
            trust_changes[user_metrics.trust_changed] += 1
        
        # Calculate rates
        total_views = event_counts.get("view", 0)
        total_completions = event_counts.get("complete", 0)
        
        return {
            "total_users": len(all_users),
            "total_events": len(self.events),
            "events_by_type": dict(event_counts),
            "overall_completion_rate": total_completions / max(1, total_views),
            "trust_changes": {k.value: v for k, v in trust_changes.items()},
            "trust_improvement_rate": trust_changes[TrustChange.INCREASED] / max(1, len(all_users)),
            "measured_since": min(e.timestamp for e in self.events).isoformat(),
            "last_event": max(e.timestamp for e in self.events).isoformat()
        }
    
    def generate_effectiveness_report(self) -> pd.DataFrame:
        """Generate content effectiveness report."""
        content_ids = set(e.content_id for e in self.events)
        
        records = []
        for content_id in content_ids:
            metrics = self.get_content_metrics(content_id)
            records.append({
                "content_id": content_id,
                "total_views": metrics.total_views,
                "unique_viewers": metrics.unique_viewers,
                "shares": metrics.total_shares,
                "completion_rate": round(metrics.completion_rate, 2),
                "avg_time_spent_sec": round(metrics.avg_time_spent, 1),
                "positive_feedback": metrics.positive_feedback_count,
                "negative_feedback": metrics.negative_feedback_count,
                "trust_increased": metrics.users_with_increased_trust,
                "trust_decreased": metrics.users_with_decreased_trust,
                "effectiveness_score": metrics.effectiveness_score
            })
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("effectiveness_score", ascending=False)
        
        return df
    
    def generate_user_journey_report(self, user_id: str) -> Dict[str, Any]:
        """Generate detailed journey report for a user."""
        metrics = self.get_user_metrics(user_id)
        user_events = sorted(
            [e for e in self.events if e.user_id == user_id],
            key=lambda x: x.timestamp
        )
        
        # Build event timeline
        timeline = []
        for e in user_events:
            timeline.append({
                "timestamp": e.timestamp.isoformat(),
                "event": e.event_type.value,
                "content_id": e.content_id,
                "time_spent": e.time_spent_seconds
            })
        
        # Trust journey
        trust_journey = []
        for m in metrics.trust_measurements:
            trust_journey.append({
                "timestamp": m.timestamp.isoformat(),
                "trust_level": m.trust_level.value,
                "trust_score": m.trust_score,
                "trigger": m.trigger
            })
        
        return {
            "user_id": user_id,
            "summary": metrics.to_dict(),
            "engagement_timeline": timeline,
            "trust_journey": trust_journey,
            "recommendations": self._generate_recommendations(metrics)
        }
    
    def _generate_recommendations(self, metrics: UserMetrics) -> List[str]:
        """Generate recommendations based on user metrics."""
        recommendations = []
        
        if metrics.completion_rate < 0.3:
            recommendations.append("Consider shorter, more digestible content")
        
        if metrics.avg_scroll_depth < 0.5:
            recommendations.append("User may prefer visual content over long text")
        
        if metrics.dismissed_content_ids:
            recommendations.append(f"Avoid content similar to: {metrics.dismissed_content_ids[0]}")
        
        if metrics.trust_changed == TrustChange.DECREASED:
            recommendations.append("Re-evaluate content tone - may need more empathetic approach")
        
        if metrics.trust_changed == TrustChange.INCREASED:
            recommendations.append("Current approach is working - continue with similar content")
        
        return recommendations
    
    def export_events(self, filepath: str):
        """Export all events to JSON."""
        data = [e.to_dict() for e in self.events]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported {len(data)} events to {filepath}")
    
    def import_events(self, filepath: str):
        """Import events from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for item in data:
            event = EngagementEvent(
                event_id=item["event_id"],
                user_id=item["user_id"],
                content_id=item["content_id"],
                event_type=EngagementType(item["event_type"]),
                timestamp=datetime.fromisoformat(item["timestamp"]),
                time_spent_seconds=item.get("time_spent_seconds", 0),
                scroll_depth_percent=item.get("scroll_depth_percent", 0.0)
            )
            self.events.append(event)
        
        logger.info(f"Imported {len(data)} events from {filepath}")


def generate_sample_events() -> List[Dict[str, Any]]:
    """Generate sample engagement events for testing."""
    import random
    
    users = ["U001", "U002", "U003", "U004", "U005"]
    content = ["C001", "C002", "C003", "C004", "C005"]
    events = []
    
    base_time = datetime.now() - timedelta(days=7)
    
    for i in range(50):
        user = random.choice(users)
        content_id = random.choice(content)
        event_type = random.choice(list(EngagementType))
        
        events.append({
            "user_id": user,
            "content_id": content_id,
            "event_type": event_type.value,
            "timestamp": (base_time + timedelta(hours=random.randint(0, 168))).isoformat(),
            "time_spent_seconds": random.randint(10, 300),
            "scroll_depth_percent": random.random()
        })
    
    return events
