"""
Dynamic Misinformation Classifier for RiboReach

Uses a hybrid approach combining:
1. Transformer-based semantic similarity (sentence-transformers)
2. TF-IDF + ML classification
3. Configurable pattern matching from JSON
4. Continual learning from flagged content

This replaces the hardcoded pattern lists with a dynamic, configurable system.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
from enum import Enum
from pathlib import Path
import re
import logging
import json
from datetime import datetime
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib

logger = logging.getLogger(__name__)


class MisinfoCategory(Enum):
    """Categories of health misinformation."""
    VACCINE_SAFETY = "vaccine_safety"
    VACCINE_EFFICACY = "vaccine_efficacy"
    CONSPIRACY = "conspiracy"
    NATURAL_IMMUNITY = "natural_immunity"
    INGREDIENT_FEAR = "ingredient_fear"
    RUSHED_DEVELOPMENT = "rushed_development"
    SIDE_EFFECTS = "side_effects"
    RELIGIOUS_ETHICAL = "religious_ethical"
    DISTRUST_AUTHORITY = "distrust_authority"
    FACTUAL = "factual"


@dataclass
class ClassificationResult:
    """Result of misinformation classification."""
    text: str
    is_misinformation: bool
    primary_category: MisinfoCategory
    confidence: float
    category_scores: Dict[str, float]
    detected_claims: List[str]
    severity: str
    embedding_similarity: float = 0.0
    matched_patterns: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "is_misinformation": self.is_misinformation,
            "category": self.primary_category.value,
            "confidence": round(self.confidence, 3),
            "severity": self.severity,
            "detected_claims": self.detected_claims,
            "embedding_similarity": round(self.embedding_similarity, 3)
        }


class PatternLoader:
    """Loads and manages misinformation patterns from configuration."""
    
    DEFAULT_CONFIG_PATH = Path(__file__).parent / "config" / "misinfo_patterns.json"
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.patterns: Dict[str, Dict] = {}
        self.severity_indicators: Dict[str, List[str]] = {}
        self.factual_indicators: List[str] = []
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        self._load_config()
    
    def _load_config(self):
        """Load patterns from JSON configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            self.patterns = config.get("categories", {})
            self.severity_indicators = config.get("severity_indicators", {})
            self.factual_indicators = config.get("factual_indicators", [])
            
            # Compile regex patterns
            self._compile_patterns()
            logger.info(f"Loaded {len(self.patterns)} categories from {self.config_path}")
        else:
            logger.warning(f"Config not found at {self.config_path}, using defaults")
            self._load_defaults()
    
    def _load_defaults(self):
        """Load default patterns if config file doesn't exist."""
        self.patterns = {
            "vaccine_safety": {
                "patterns": [r"vaccines?\s+cause", r"vaccine\s+injury"],
                "keywords": ["autism", "death", "injury"],
                "example_claims": ["Vaccines cause autism"]
            },
            "conspiracy": {
                "patterns": [r"microchip", r"5g", r"big\s+pharma"],
                "keywords": ["microchip", "tracking", "control"],
                "example_claims": ["Vaccines contain microchips"]
            }
        }
        self.severity_indicators = {
            "high": ["death", "kill", "poison"],
            "medium": ["dangerous", "harmful"],
            "low": ["concern", "question"]
        }
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self._compiled_patterns = {}
        for category, data in self.patterns.items():
            patterns = data.get("patterns", [])
            self._compiled_patterns[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def reload(self):
        """Reload patterns from config file."""
        self._load_config()
    
    def add_pattern(self, category: str, pattern: str, persist: bool = True):
        """Add a new pattern dynamically."""
        if category not in self.patterns:
            self.patterns[category] = {"patterns": [], "keywords": [], "example_claims": []}
        
        self.patterns[category]["patterns"].append(pattern)
        self._compiled_patterns.setdefault(category, []).append(
            re.compile(pattern, re.IGNORECASE)
        )
        
        if persist:
            self._save_config()
    
    def add_keyword(self, category: str, keyword: str, persist: bool = True):
        """Add a new keyword dynamically."""
        if category not in self.patterns:
            self.patterns[category] = {"patterns": [], "keywords": [], "example_claims": []}
        
        if keyword not in self.patterns[category]["keywords"]:
            self.patterns[category]["keywords"].append(keyword)
        
        if persist:
            self._save_config()
    
    def add_example(self, category: str, claim: str, persist: bool = True):
        """Add a new example claim for training."""
        if category not in self.patterns:
            self.patterns[category] = {"patterns": [], "keywords": [], "example_claims": []}
        
        if claim not in self.patterns[category].get("example_claims", []):
            self.patterns[category].setdefault("example_claims", []).append(claim)
        
        if persist:
            self._save_config()
    
    def _save_config(self):
        """Save current patterns to config file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config = {
            "version": "1.0.0",
            "last_updated": datetime.now().isoformat(),
            "categories": self.patterns,
            "severity_indicators": self.severity_indicators,
            "factual_indicators": self.factual_indicators
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"Saved patterns to {self.config_path}")
    
    def get_compiled_patterns(self, category: str) -> List[re.Pattern]:
        """Get compiled regex patterns for a category."""
        return self._compiled_patterns.get(category, [])
    
    def get_keywords(self, category: str) -> List[str]:
        """Get keywords for a category."""
        return self.patterns.get(category, {}).get("keywords", [])
    
    def get_examples(self, category: str) -> List[str]:
        """Get example claims for a category."""
        return self.patterns.get(category, {}).get("example_claims", [])
    
    def get_all_examples(self) -> pd.DataFrame:
        """Get all example claims as training data."""
        records = []
        for category, data in self.patterns.items():
            for claim in data.get("example_claims", []):
                records.append({"text": claim, "category": category})
        return pd.DataFrame(records)


class EmbeddingMatcher:
    """
    Semantic similarity matching using sentence embeddings.
    Falls back gracefully if sentence-transformers is not available.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.reference_embeddings: Dict[str, np.ndarray] = {}
        self.reference_texts: Dict[str, List[str]] = {}
        self._load_model()
    
    def _load_model(self):
        """Try to load sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Loaded embedding model: {self.model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Semantic similarity features disabled. "
                "Install with: pip install sentence-transformers"
            )
            self.model = None
    
    @property
    def is_available(self) -> bool:
        """Check if embedding model is available."""
        return self.model is not None
    
    def add_reference(self, category: str, texts: List[str]):
        """Add reference texts for a category."""
        if not self.is_available:
            return
        
        self.reference_texts[category] = texts
        embeddings = self.model.encode(texts)
        # Store mean embedding for the category
        self.reference_embeddings[category] = np.mean(embeddings, axis=0)
    
    def compute_similarity(self, text: str) -> Dict[str, float]:
        """Compute similarity to each category."""
        if not self.is_available or not self.reference_embeddings:
            return {}
        
        text_embedding = self.model.encode([text])[0]
        
        similarities = {}
        for category, ref_embedding in self.reference_embeddings.items():
            # Cosine similarity
            similarity = np.dot(text_embedding, ref_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(ref_embedding)
            )
            similarities[category] = float(similarity)
        
        return similarities
    
    def find_most_similar(self, text: str, threshold: float = 0.5) -> Optional[Tuple[str, float]]:
        """Find the most similar category above threshold."""
        similarities = self.compute_similarity(text)
        
        if not similarities:
            return None
        
        best_category = max(similarities, key=similarities.get)
        best_score = similarities[best_category]
        
        if best_score >= threshold:
            return (best_category, best_score)
        return None


class MLClassifier:
    """
    Machine learning classifier using TF-IDF + ensemble methods.
    Supports multiple model types and incremental learning.
    """
    
    MODEL_TYPES = {
        "logistic": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier
    }
    
    def __init__(
        self,
        model_type: str = "logistic",
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2)
    ):
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            sublinear_tf=True
        )
        
        model_class = self.MODEL_TYPES.get(model_type, LogisticRegression)
        if model_type == "logistic":
            self.model = model_class(max_iter=1000, class_weight='balanced')
        else:
            self.model = model_class(n_estimators=100, random_state=42)
        
        self.label_encoder = LabelEncoder()
        self._is_fitted = False
        self._training_data: List[Dict] = []
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    def train(self, data: pd.DataFrame, text_col: str = "text", label_col: str = "category"):
        """Train the classifier on labeled data."""
        if data.empty:
            logger.warning("No training data provided")
            return {}
        
        # Store training data for incremental learning
        self._training_data.extend(data.to_dict('records'))
        
        X = self.vectorizer.fit_transform(data[text_col])
        y = self.label_encoder.fit_transform(data[label_col])
        
        # Train with cross-validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        
        logger.info(f"Trained {self.model_type} classifier. Accuracy: {accuracy:.2%}")
        
        return {
            "accuracy": accuracy,
            "num_samples": len(data),
            "num_categories": len(self.label_encoder.classes_)
        }
    
    def add_training_example(self, text: str, category: str):
        """Add a single training example (for incremental learning)."""
        self._training_data.append({"text": text, "category": category})
    
    def retrain(self):
        """Retrain model with accumulated examples."""
        if not self._training_data:
            return {}
        
        df = pd.DataFrame(self._training_data)
        return self.train(df)
    
    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """Predict category for text."""
        if not self._is_fitted:
            return ("factual", 0.5, {"factual": 0.5})
        
        X = self.vectorizer.transform([text])
        
        # Get prediction
        pred_idx = self.model.predict(X)[0]
        category = self.label_encoder.inverse_transform([pred_idx])[0]
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X)[0]
            confidence = float(max(probs))
            category_scores = {
                self.label_encoder.inverse_transform([i])[0]: float(p)
                for i, p in enumerate(probs)
            }
        else:
            confidence = 0.8
            category_scores = {category: 0.8}
        
        return (category, confidence, category_scores)
    
    def save(self, filepath: str):
        """Save model to disk."""
        joblib.dump({
            "vectorizer": self.vectorizer,
            "model": self.model,
            "label_encoder": self.label_encoder,
            "training_data": self._training_data
        }, filepath)
    
    @classmethod
    def load(cls, filepath: str) -> "MLClassifier":
        """Load model from disk."""
        data = joblib.load(filepath)
        instance = cls()
        instance.vectorizer = data["vectorizer"]
        instance.model = data["model"]
        instance.label_encoder = data["label_encoder"]
        instance._training_data = data.get("training_data", [])
        instance._is_fitted = True
        return instance


class DynamicMisinformationClassifier:
    """
    Hybrid misinformation classifier combining multiple approaches:
    
    1. Pattern matching (from configurable JSON)
    2. ML classification (TF-IDF + Logistic Regression/Random Forest)
    3. Semantic similarity (sentence-transformers embeddings)
    
    Example usage:
    ```python
    classifier = DynamicMisinformationClassifier()
    
    # Classify text
    result = classifier.classify("Vaccines cause autism")
    print(f"Category: {result.primary_category.value}")
    print(f"Confidence: {result.confidence:.2%}")
    
    # Add new pattern dynamically
    classifier.add_pattern("conspiracy", r"deep\s+state")
    
    # Add new training example (for learning)
    classifier.flag_as_misinfo("New misinformation claim", "vaccine_safety")
    classifier.retrain()
    ```
    """
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        use_embeddings: bool = True,
        model_type: str = "logistic"
    ):
        # Initialize components
        self.pattern_loader = PatternLoader(config_path)
        self.ml_classifier = MLClassifier(model_type=model_type)
        self.embedding_matcher = EmbeddingMatcher() if use_embeddings else None
        
        # Weights for combining scores
        self.weights = {
            "pattern": 0.3,
            "ml": 0.4,
            "embedding": 0.3
        }
        
        # Initialize embeddings with example claims
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embedding references from pattern config."""
        if self.embedding_matcher and self.embedding_matcher.is_available:
            for category, data in self.pattern_loader.patterns.items():
                examples = data.get("example_claims", [])
                if examples:
                    self.embedding_matcher.add_reference(category, examples)
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Train the ML classifier on labeled data."""
        # Combine with examples from patterns
        pattern_examples = self.pattern_loader.get_all_examples()
        if not pattern_examples.empty:
            data = pd.concat([data, pattern_examples], ignore_index=True)
        
        return self.ml_classifier.train(data)
    
    def _detect_patterns(self, text: str) -> Dict[str, List[str]]:
        """Detect misinformation patterns in text."""
        matches = {}
        text_lower = text.lower()
        
        for category in self.pattern_loader.patterns:
            patterns = self.pattern_loader.get_compiled_patterns(category)
            category_matches = []
            
            for pattern in patterns:
                if pattern.search(text_lower):
                    category_matches.append(pattern.pattern)
            
            # Also check keywords
            keywords = self.pattern_loader.get_keywords(category)
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    category_matches.append(f"keyword:{keyword}")
            
            if category_matches:
                matches[category] = category_matches
        
        return matches
    
    def _assess_severity(self, text: str) -> str:
        """Assess severity of misinformation."""
        text_lower = text.lower()
        
        for severity, keywords in self.pattern_loader.severity_indicators.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return severity
        
        return "low"
    
    def _check_factual(self, text: str) -> bool:
        """Check if text contains factual indicators."""
        text_lower = text.lower()
        
        for indicator in self.pattern_loader.factual_indicators:
            if indicator in text_lower:
                return True
        return False
    
    def classify(self, text: str) -> ClassificationResult:
        """
        Classify text using hybrid approach.
        
        Combines:
        - Pattern matching score
        - ML classification score
        - Embedding similarity score
        """
        # 1. Pattern matching
        pattern_matches = self._detect_patterns(text)
        pattern_scores = {
            cat: len(matches) / 5.0  # Normalize by expected max matches
            for cat, matches in pattern_matches.items()
        }
        
        detected_claims = []
        for matches in pattern_matches.values():
            detected_claims.extend(matches)
        
        # 2. ML classification
        ml_category, ml_confidence, ml_scores = self.ml_classifier.predict(text)
        
        # 3. Embedding similarity
        embedding_scores = {}
        embedding_similarity = 0.0
        if self.embedding_matcher and self.embedding_matcher.is_available:
            embedding_scores = self.embedding_matcher.compute_similarity(text)
            if embedding_scores:
                embedding_similarity = max(embedding_scores.values())
        
        # Combine scores
        combined_scores = {}
        all_categories = set(pattern_scores.keys()) | set(ml_scores.keys()) | set(embedding_scores.keys())
        
        for category in all_categories:
            pattern_score = pattern_scores.get(category, 0)
            ml_score = ml_scores.get(category, 0)
            embed_score = embedding_scores.get(category, 0)
            
            combined = (
                self.weights["pattern"] * pattern_score +
                self.weights["ml"] * ml_score +
                self.weights["embedding"] * embed_score
            )
            combined_scores[category] = combined
        
        # Determine primary category
        if combined_scores:
            primary_category_str = max(combined_scores, key=combined_scores.get)
            confidence = combined_scores[primary_category_str]
        else:
            primary_category_str = "factual"
            confidence = 0.5
        
        # Check for factual content
        is_factual = self._check_factual(text)
        if is_factual and confidence < 0.7:
            primary_category_str = "factual"
            confidence = 0.6
        
        # Convert to enum
        try:
            primary_category = MisinfoCategory(primary_category_str)
        except ValueError:
            primary_category = MisinfoCategory.FACTUAL
        
        is_misinfo = primary_category != MisinfoCategory.FACTUAL
        severity = self._assess_severity(text) if is_misinfo else "none"
        
        return ClassificationResult(
            text=text,
            is_misinformation=is_misinfo,
            primary_category=primary_category,
            confidence=min(1.0, confidence),
            category_scores=combined_scores,
            detected_claims=detected_claims,
            severity=severity,
            embedding_similarity=embedding_similarity,
            matched_patterns=detected_claims[:5]
        )
    
    def add_pattern(self, category: str, pattern: str):
        """Add a new regex pattern for a category."""
        self.pattern_loader.add_pattern(category, pattern)
        logger.info(f"Added pattern '{pattern}' to {category}")
    
    def add_keyword(self, category: str, keyword: str):
        """Add a new keyword for a category."""
        self.pattern_loader.add_keyword(category, keyword)
        logger.info(f"Added keyword '{keyword}' to {category}")
    
    def flag_as_misinfo(self, text: str, category: str):
        """
        Flag content as misinformation for learning.
        This adds it as a training example for future model updates.
        """
        self.pattern_loader.add_example(category, text)
        self.ml_classifier.add_training_example(text, category)
        
        # Update embeddings if available
        if self.embedding_matcher and self.embedding_matcher.is_available:
            examples = self.pattern_loader.get_examples(category)
            self.embedding_matcher.add_reference(category, examples)
        
        logger.info(f"Flagged as {category}: {text[:50]}...")
    
    def flag_as_factual(self, text: str):
        """Flag content as factual (not misinformation)."""
        self.ml_classifier.add_training_example(text, "factual")
    
    def retrain(self) -> Dict[str, Any]:
        """Retrain ML model with accumulated examples."""
        return self.ml_classifier.retrain()
    
    def reload_config(self):
        """Reload patterns from configuration file."""
        self.pattern_loader.reload()
        self._init_embeddings()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get classifier statistics."""
        return {
            "num_categories": len(self.pattern_loader.patterns),
            "total_patterns": sum(
                len(data.get("patterns", [])) 
                for data in self.pattern_loader.patterns.values()
            ),
            "total_keywords": sum(
                len(data.get("keywords", []))
                for data in self.pattern_loader.patterns.values()
            ),
            "ml_fitted": self.ml_classifier.is_fitted,
            "embeddings_available": (
                self.embedding_matcher is not None and 
                self.embedding_matcher.is_available
            ),
            "weights": self.weights
        }
    
    def save(self, filepath: str):
        """Save classifier state."""
        self.ml_classifier.save(filepath)
    
    @classmethod
    def load(cls, filepath: str, config_path: Optional[Path] = None) -> "DynamicMisinformationClassifier":
        """Load classifier from saved state."""
        instance = cls(config_path=config_path)
        instance.ml_classifier = MLClassifier.load(filepath)
        return instance


# Backwards compatibility - alias to maintain existing imports
MisinformationClassifier = DynamicMisinformationClassifier


def generate_training_data(n_samples: int = 500) -> pd.DataFrame:
    """Generate training data from pattern examples."""
    loader = PatternLoader()
    base_examples = loader.get_all_examples()
    
    if base_examples.empty:
        # Fallback hardcoded examples
        examples = {
            "vaccine_safety": [
                "Vaccines cause autism in children",
                "The vaccine killed thousands of people",
                "These toxic vaccines are injuring our kids"
            ],
            "conspiracy": [
                "Vaccines contain tracking microchips",
                "Bill Gates wants to depopulate the world",
                "Big pharma is covering up the truth"
            ],
            "ingredient_fear": [
                "Vaccines contain dangerous mercury levels",
                "They put aborted fetal cells in vaccines",
                "The vaccine has graphene oxide"
            ],
            "natural_immunity": [
                "Natural immunity is better than vaccines",
                "I survived COVID so I don't need a shot",
                "My immune system is strong enough"
            ],
            "rushed_development": [
                "The vaccine was rushed and untested",
                "We're all guinea pigs for this experiment",
                "They skipped important safety trials"
            ],
            "side_effects": [
                "The vaccine causes blood clots",
                "Vaccines are causing infertility",
                "VAERS shows thousands of deaths"
            ],
            "vaccine_efficacy": [
                "The vaccine doesn't even work",
                "Vaccinated people are still spreading COVID",
                "The vaccine is useless"
            ],
            "distrust_authority": [
                "The CDC is lying to us",
                "Fauci has conflicts of interest",
                "They're silencing honest doctors"
            ],
            "factual": [
                "Vaccines help protect against serious illness",
                "The vaccine went through rigorous testing",
                "I got vaccinated and feel fine",
                "My doctor recommended vaccination",
                "Clinical trials show the vaccine is effective"
            ]
        }
        
        records = []
        for category, texts in examples.items():
            for text in texts:
                records.append({"text": text, "category": category})
        base_examples = pd.DataFrame(records)
    
    # Augment with variations
    augmented = []
    variations = ["vaccine", "vax", "jab", "shot"]
    
    for _, row in base_examples.iterrows():
        augmented.append(row.to_dict())
        
        # Add variations
        for _ in range(n_samples // len(base_examples)):
            text = row["text"]
            for var in variations[1:]:
                if "vaccine" in text.lower():
                    new_text = re.sub(r"vaccine", var, text, flags=re.IGNORECASE)
                    augmented.append({"text": new_text, "category": row["category"]})
                    break
    
    return pd.DataFrame(augmented[:n_samples])
