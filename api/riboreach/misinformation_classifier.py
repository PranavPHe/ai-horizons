"""
Misinformation Classifier for RiboReach

Detects and categorizes health/vaccine misinformation using NLP and ML.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import re
import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib

logger = logging.getLogger(__name__)


class MisinfoCategory(Enum):
    """Categories of health misinformation."""
    VACCINE_SAFETY = "vaccine_safety"           # False claims about vaccine dangers
    VACCINE_EFFICACY = "vaccine_efficacy"       # Doubts about vaccine effectiveness
    CONSPIRACY = "conspiracy"                   # Conspiracy theories (govt, pharma, etc.)
    NATURAL_IMMUNITY = "natural_immunity"       # Overconfidence in natural immunity
    INGREDIENT_FEAR = "ingredient_fear"         # Fear of specific ingredients
    RUSHED_DEVELOPMENT = "rushed_development"   # Concerns about development speed
    SIDE_EFFECTS = "side_effects"               # Exaggerated side effect claims
    RELIGIOUS_ETHICAL = "religious_ethical"     # Religious/ethical objections
    DISTRUST_AUTHORITY = "distrust_authority"   # General distrust of health authorities
    FACTUAL = "factual"                         # Accurate/factual content (not misinfo)


@dataclass
class ClassificationResult:
    """Result of misinformation classification."""
    text: str
    is_misinformation: bool
    primary_category: MisinfoCategory
    confidence: float
    category_scores: Dict[str, float]
    detected_claims: List[str]
    severity: str  # 'low', 'medium', 'high'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text[:200] + "..." if len(self.text) > 200 else self.text,
            "is_misinformation": self.is_misinformation,
            "category": self.primary_category.value,
            "confidence": self.confidence,
            "severity": self.severity,
            "detected_claims": self.detected_claims
        }


class MisinformationClassifier:
    """
    ML-powered classifier for detecting and categorizing health misinformation.
    
    Example usage:
    ```python
    classifier = MisinformationClassifier()
    classifier.train(training_data)
    
    result = classifier.classify("Vaccines contain microchips for tracking")
    print(f"Misinformation: {result.is_misinformation}")
    print(f"Category: {result.primary_category.value}")
    print(f"Confidence: {result.confidence:.2%}")
    ```
    """
    
    # Common misinformation patterns and keywords
    MISINFO_PATTERNS = {
        MisinfoCategory.VACCINE_SAFETY: [
            r"vaccines?\s+(cause|causes|causing)\s+\w+",
            r"dangerous\s+vaccine", r"vaccine\s+injury",
            r"harmed?\s+by\s+vaccine", r"vaccine\s+death",
            r"toxic\s+vaccine", r"poison"
        ],
        MisinfoCategory.CONSPIRACY: [
            r"big\s+pharma", r"government\s+control",
            r"population\s+control", r"microchip", r"5g",
            r"bill\s+gates", r"new\s+world\s+order", r"plandemic",
            r"cover[\s-]?up", r"they\s+don'?t\s+want\s+you\s+to\s+know"
        ],
        MisinfoCategory.INGREDIENT_FEAR: [
            r"mercury", r"aluminum", r"aborted\s+fetal",
            r"fetal\s+cells", r"graphene", r"magnetic",
            r"nano[\s-]?particles", r"chemicals?"
        ],
        MisinfoCategory.NATURAL_IMMUNITY: [
            r"natural\s+immunity\s+(is\s+)?better",
            r"immune\s+system\s+can\s+handle",
            r"don'?t\s+need\s+vaccine", r"my\s+body\s+my\s+choice",
            r"survived\s+covid\s+without"
        ],
        MisinfoCategory.RUSHED_DEVELOPMENT: [
            r"rushed", r"experimental", r"not\s+tested",
            r"guinea\s+pig", r"long[\s-]?term\s+effects\s+unknown",
            r"too\s+fast", r"skipped\s+trials?"
        ],
        MisinfoCategory.SIDE_EFFECTS: [
            r"myocarditis", r"blood\s+clots?", r"infertility",
            r"sterile", r"miscarriage", r"heart\s+(attack|problem)",
            r"sudden\s+death", r"vaers"
        ],
        MisinfoCategory.VACCINE_EFFICACY: [
            r"doesn'?t\s+(work|prevent)", r"still\s+got\s+(sick|covid)",
            r"vaccinated\s+people\s+spreading",
            r"useless", r"doesn'?t\s+stop\s+transmission"
        ],
        MisinfoCategory.DISTRUST_AUTHORITY: [
            r"cdc\s+(lies?|lying)", r"fda\s+corrupt",
            r"can'?t\s+trust", r"paid\s+off", r"conflicts?\s+of\s+interest",
            r"fauci", r"silencing\s+doctors?"
        ]
    }
    
    # Severity indicators
    SEVERITY_KEYWORDS = {
        "high": ["death", "kill", "murder", "genocide", "depopulation", "poison"],
        "medium": ["dangerous", "harmful", "toxic", "experimental", "untested"],
        "low": ["concern", "question", "worry", "unsure", "skeptical"]
    }
    
    def __init__(
        self,
        model_type: str = "logistic_regression",
        use_patterns: bool = True
    ):
        self.model_type = model_type
        self.use_patterns = use_patterns
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=2
        )
        self.model = None
        self.label_encoder: Dict[str, int] = {}
        self.label_decoder: Dict[int, str] = {}
        self._is_fitted = False
        
        # Compile regex patterns
        self._compiled_patterns = {}
        for category, patterns in self.MISINFO_PATTERNS.items():
            self._compiled_patterns[category] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
    
    def _create_model(self):
        """Create the underlying classifier."""
        if self.model_type == "naive_bayes":
            return MultinomialNB(alpha=0.1)
        elif self.model_type == "logistic_regression":
            return LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                C=1.0
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                class_weight='balanced',
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for classification."""
        # Lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _detect_patterns(self, text: str) -> Dict[MisinfoCategory, List[str]]:
        """Detect misinformation patterns in text."""
        detected = {}
        text_lower = text.lower()
        
        for category, patterns in self._compiled_patterns.items():
            matches = []
            for pattern in patterns:
                found = pattern.findall(text_lower)
                matches.extend(found)
            if matches:
                detected[category] = list(set(matches))
        
        return detected
    
    def _assess_severity(self, text: str) -> str:
        """Assess severity level of misinformation."""
        text_lower = text.lower()
        
        for level in ["high", "medium", "low"]:
            for keyword in self.SEVERITY_KEYWORDS[level]:
                if keyword in text_lower:
                    return level
        
        return "low"
    
    def train(
        self,
        data: pd.DataFrame,
        text_column: str = "text",
        label_column: str = "category",
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the misinformation classifier.
        
        Args:
            data: DataFrame with text and category labels
            text_column: Name of text column
            label_column: Name of label column
            test_size: Fraction for test split
        
        Returns:
            Dict with training metrics
        """
        logger.info(f"Training misinformation classifier on {len(data)} samples")
        
        # Preprocess
        texts = data[text_column].apply(self._preprocess_text)
        labels = data[label_column]
        
        # Encode labels
        unique_labels = labels.unique()
        self.label_encoder = {label: i for i, label in enumerate(unique_labels)}
        self.label_decoder = {i: label for label, i in self.label_encoder.items()}
        y = labels.map(self.label_encoder).values
        
        # Split data
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            texts, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Vectorize
        X_train = self.vectorizer.fit_transform(X_train_text)
        X_test = self.vectorizer.transform(X_test_text)
        
        # Train model
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self._create_model(),
            self.vectorizer.transform(texts),
            y,
            cv=5,
            scoring='f1_weighted'
        )
        
        self._is_fitted = True
        
        metrics = {
            "accuracy": (y_pred == y_test).mean(),
            "cv_f1_mean": cv_scores.mean(),
            "cv_f1_std": cv_scores.std(),
            "classification_report": classification_report(
                y_test, y_pred,
                target_names=[self.label_decoder[i] for i in range(len(self.label_decoder))]
            )
        }
        
        logger.info(f"Training complete. Accuracy: {metrics['accuracy']:.2%}")
        return metrics
    
    def classify(self, text: str) -> ClassificationResult:
        """
        Classify a text for misinformation.
        
        Args:
            text: Text to classify
        
        Returns:
            ClassificationResult with category, confidence, and detected claims
        """
        preprocessed = self._preprocess_text(text)
        
        # Pattern-based detection
        pattern_matches = self._detect_patterns(text)
        detected_claims = []
        for matches in pattern_matches.values():
            detected_claims.extend(matches)
        
        # ML-based classification if model is trained
        if self._is_fitted:
            X = self.vectorizer.transform([preprocessed])
            
            # Get probabilities
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(X)[0]
                pred_idx = np.argmax(probs)
                confidence = probs[pred_idx]
                
                category_scores = {
                    self.label_decoder[i]: float(p)
                    for i, p in enumerate(probs)
                }
            else:
                pred_idx = self.model.predict(X)[0]
                confidence = 0.8  # Default confidence for models without predict_proba
                category_scores = {self.label_decoder[pred_idx]: 0.8}
            
            primary_category_str = self.label_decoder[pred_idx]
            
            try:
                primary_category = MisinfoCategory(primary_category_str)
            except ValueError:
                primary_category = MisinfoCategory.FACTUAL
        
        else:
            # Use pattern-based classification only
            if pattern_matches:
                # Use category with most matches
                category_counts = {
                    cat: len(matches) for cat, matches in pattern_matches.items()
                }
                primary_category = max(category_counts, key=category_counts.get)
                total_matches = sum(category_counts.values())
                confidence = min(0.9, 0.5 + total_matches * 0.1)
                category_scores = {
                    cat.value: count / total_matches
                    for cat, count in category_counts.items()
                }
            else:
                primary_category = MisinfoCategory.FACTUAL
                confidence = 0.6
                category_scores = {"factual": 0.6}
        
        # Determine if it's misinformation
        is_misinfo = primary_category != MisinfoCategory.FACTUAL
        
        # Assess severity
        severity = self._assess_severity(text) if is_misinfo else "none"
        
        return ClassificationResult(
            text=text,
            is_misinformation=is_misinfo,
            primary_category=primary_category,
            confidence=confidence,
            category_scores=category_scores,
            detected_claims=detected_claims[:5],  # Top 5 claims
            severity=severity
        )
    
    def classify_batch(self, texts: List[str]) -> List[ClassificationResult]:
        """Classify multiple texts."""
        return [self.classify(text) for text in texts]
    
    def get_counter_topics(self, category: MisinfoCategory) -> List[str]:
        """Get suggested educational topics to counter specific misinformation."""
        counter_topics = {
            MisinfoCategory.VACCINE_SAFETY: [
                "How vaccines are tested for safety",
                "Understanding vaccine adverse event reporting",
                "The vaccine approval process explained",
                "Real vs. perceived vaccine risks"
            ],
            MisinfoCategory.CONSPIRACY: [
                "How vaccine development really works",
                "Understanding scientific peer review",
                "Public health transparency measures",
                "History of successful vaccination programs"
            ],
            MisinfoCategory.INGREDIENT_FEAR: [
                "What's actually in vaccines and why",
                "Understanding preservatives and adjuvants",
                "Dose makes the poison: toxicology basics",
                "Comparing vaccine ingredients to everyday exposures"
            ],
            MisinfoCategory.NATURAL_IMMUNITY: [
                "How immunity works: natural vs. vaccine",
                "Risks of disease-acquired immunity",
                "Why vaccines provide safer protection",
                "Understanding herd immunity"
            ],
            MisinfoCategory.RUSHED_DEVELOPMENT: [
                "mRNA vaccine technology: decades of research",
                "How COVID vaccines were developed quickly AND safely",
                "Understanding emergency use authorization",
                "The role of parallel vs. sequential testing"
            ],
            MisinfoCategory.SIDE_EFFECTS: [
                "Common vaccine side effects explained",
                "Rare adverse events in context",
                "Risk-benefit analysis of vaccination",
                "How side effects are monitored"
            ],
            MisinfoCategory.VACCINE_EFFICACY: [
                "What vaccine efficacy percentages mean",
                "How vaccines reduce severe disease",
                "Understanding breakthrough infections",
                "Population-level vaccine impact"
            ],
            MisinfoCategory.DISTRUST_AUTHORITY: [
                "How health agencies operate independently",
                "Understanding conflicts of interest policies",
                "The scientific consensus process",
                "Questions to ask your own doctor"
            ],
            MisinfoCategory.RELIGIOUS_ETHICAL: [
                "Religious perspectives on vaccination",
                "Ethical frameworks for public health decisions",
                "Vaccine alternatives and accommodations",
                "Community responsibility discussions"
            ]
        }
        
        return counter_topics.get(category, ["General vaccine education"])
    
    def save(self, filepath: str):
        """Save trained classifier."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save untrained classifier")
        joblib.dump({
            "model": self.model,
            "vectorizer": self.vectorizer,
            "label_encoder": self.label_encoder,
            "label_decoder": self.label_decoder,
            "model_type": self.model_type
        }, filepath)
        logger.info(f"Classifier saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "MisinformationClassifier":
        """Load trained classifier."""
        data = joblib.load(filepath)
        classifier = cls(model_type=data["model_type"])
        classifier.model = data["model"]
        classifier.vectorizer = data["vectorizer"]
        classifier.label_encoder = data["label_encoder"]
        classifier.label_decoder = data["label_decoder"]
        classifier._is_fitted = True
        logger.info(f"Classifier loaded from {filepath}")
        return classifier


def generate_training_data(n_samples: int = 500) -> pd.DataFrame:
    """Generate synthetic training data for the classifier."""
    np.random.seed(42)
    
    # Example texts for each category
    examples = {
        "vaccine_safety": [
            "Vaccines cause autism in children",
            "My child got sick right after the vaccine",
            "Vaccines are dangerous and harming our kids",
            "There are too many vaccines given at once",
            "The vaccine injured my family member"
        ],
        "conspiracy": [
            "Big pharma is hiding the truth about vaccines",
            "The government is using vaccines for population control",
            "Bill Gates wants to microchip everyone through vaccines",
            "This is all a plandemic to control us",
            "They don't want you to know the real side effects"
        ],
        "ingredient_fear": [
            "Vaccines contain mercury and aluminum",
            "They use aborted fetal cells in vaccines",
            "There's graphene oxide in the vaccine making people magnetic",
            "The chemicals in vaccines are toxic",
            "I don't want those ingredients in my body"
        ],
        "natural_immunity": [
            "Natural immunity is better than vaccine immunity",
            "My immune system can handle it without vaccines",
            "I survived COVID without the vaccine so I don't need it",
            "Our bodies were designed to fight disease naturally",
            "Vaccines weaken the natural immune system"
        ],
        "rushed_development": [
            "The vaccine was rushed and not properly tested",
            "We don't know the long-term effects yet",
            "I'm not a guinea pig for an experimental vaccine",
            "They skipped important safety trials",
            "It usually takes 10 years to develop a vaccine"
        ],
        "side_effects": [
            "The vaccine causes blood clots and heart problems",
            "So many people are dying from the vaccine",
            "The vaccine causes infertility",
            "VAERS shows thousands of deaths from vaccines",
            "More people are harmed by the vaccine than COVID"
        ],
        "vaccine_efficacy": [
            "The vaccine doesn't even work, people still get sick",
            "Vaccinated people are spreading the virus too",
            "Why get vaccinated if I can still catch COVID",
            "The vaccine is useless against new variants",
            "Natural immunity works better than the vaccine"
        ],
        "distrust_authority": [
            "The CDC and FDA are lying to us",
            "Fauci has conflicts of interest with pharma",
            "You can't trust what the health authorities say",
            "They're silencing doctors who speak out",
            "Follow the money - it's all about profit"
        ],
        "factual": [
            "Vaccines help protect against serious illness",
            "The vaccine went through rigorous testing",
            "I got vaccinated and had mild side effects",
            "My doctor recommended vaccination",
            "Vaccines have saved millions of lives",
            "I'm doing research to make an informed decision",
            "What are the actual statistics on vaccine effectiveness?",
            "Can someone explain how mRNA vaccines work?"
        ]
    }
    
    records = []
    samples_per_category = n_samples // len(examples)
    
    for category, texts in examples.items():
        for _ in range(samples_per_category):
            # Add variation
            base_text = np.random.choice(texts)
            # Add some noise/variation
            if np.random.random() > 0.5:
                base_text = base_text.replace("vaccine", np.random.choice(["vax", "jab", "shot"]))
            records.append({"text": base_text, "category": category})
    
    return pd.DataFrame(records)
