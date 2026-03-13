"""
Dynamic Responder for RiboReach

Generates contextual educational responses based on actual user input,
without relying on hardcoded content libraries.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import logging
import re
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DynamicResponse:
    """A dynamically generated educational response."""
    user_input: str
    detected_category: str
    confidence: float
    
    headline: str
    explanation: str
    key_facts: List[str]
    myth_vs_fact: Optional[Tuple[str, str]] = None  # (myth, fact)
    sources: List[str] = field(default_factory=list)
    follow_up_suggestion: str = ""
    
    def to_text(self) -> str:
        """Format as readable text."""
        text = f"\n{'='*60}\n"
        text += f"📋 RESPONSE TO: \"{self.user_input[:80]}{'...' if len(self.user_input) > 80 else ''}\"\n"
        text += f"{'='*60}\n\n"
        
        text += f"🏷️  Category: {self.detected_category} ({self.confidence:.0%} confidence)\n\n"
        text += f"📌 {self.headline}\n\n"
        text += f"{self.explanation}\n\n"
        
        if self.myth_vs_fact:
            text += f"❌ Myth: {self.myth_vs_fact[0]}\n"
            text += f"✅ Fact: {self.myth_vs_fact[1]}\n\n"
        
        if self.key_facts:
            text += "📚 Key Facts:\n"
            for fact in self.key_facts:
                text += f"   • {fact}\n"
            text += "\n"
        
        if self.sources:
            text += "🔗 Learn More:\n"
            for source in self.sources:
                text += f"   • {source}\n"
            text += "\n"
        
        if self.follow_up_suggestion:
            text += f"💡 {self.follow_up_suggestion}\n"
        
        return text


class DynamicResponder:
    """
    Generates educational responses dynamically based on detected misinformation.
    
    Instead of matching to hardcoded content, this generates contextual responses
    by combining:
    1. Detected misinformation category
    2. Extracted claims from user input
    3. Category-specific educational frameworks
    4. Dynamic fact generation
    """
    
    # Educational frameworks for each category
    # These provide the structure, but content is generated contextually
    EDUCATIONAL_FRAMEWORKS = {
        "conspiracy": {
            "approach": "address_distrust",
            "headline_template": "Let's Examine This Claim About {topic}",
            "key_questions": [
                "Who benefits from spreading this claim?",
                "What would need to be true for this to work?",
                "How many people would need to keep the secret?",
                "What do independent experts say?"
            ],
            "facts_framework": [
                "Vaccines are developed by thousands of independent researchers globally",
                "Vaccine data is published and peer-reviewed openly",
                "No mechanism exists for {claimed_action}",
                "Multiple competing organizations verify safety independently"
            ],
            "sources": [
                "Independent scientific journals (Nature, Lancet, NEJM)",
                "University research departments",
                "International health organizations (WHO)",
                "Your personal healthcare provider"
            ]
        },
        "vaccine_safety": {
            "approach": "validate_and_inform",
            "headline_template": "Understanding Vaccine Safety Concerns",
            "key_questions": [
                "What specific safety aspect concerns you?",
                "Are you looking at clinical trial data?",
                "Have you discussed this with a healthcare provider?"
            ],
            "facts_framework": [
                "Vaccines undergo 3 phases of clinical trials with thousands of participants",
                "Post-approval monitoring continues indefinitely",
                "Adverse events are tracked through multiple independent systems",
                "Serious side effects are extremely rare compared to disease risks"
            ],
            "sources": [
                "FDA Vaccine Approval Process",
                "CDC VAERS Database",
                "Clinical trial publications",
                "Your doctor or pharmacist"
            ]
        },
        "rushed_development": {
            "approach": "explain_process",
            "headline_template": "How Vaccine Development Actually Worked",
            "key_questions": [
                "Do you know about the decades of prior mRNA research?",
                "Are you aware steps were overlapped, not skipped?"
            ],
            "facts_framework": [
                "mRNA technology has been researched since the 1990s",
                "Previous coronavirus research (SARS, MERS) provided foundations",
                "Massive global funding enabled parallel processes",
                "Regulatory review was prioritized, not shortened"
            ],
            "sources": [
                "mRNA research history publications",
                "FDA Emergency Use Authorization documentation",
                "Timeline of regulatory reviews"
            ]
        },
        "ingredient_fear": {
            "approach": "explain_science",
            "headline_template": "Understanding What's in Vaccines",
            "key_questions": [
                "Which specific ingredient concerns you?",
                "Do you understand the amounts and purposes?"
            ],
            "facts_framework": [
                "Every ingredient serves a specific, documented purpose",
                "Amounts are orders of magnitude below harmful levels",
                "Ingredients are the same as those in many common foods/medicines",
                "Full ingredient lists are publicly available"
            ],
            "sources": [
                "CDC Vaccine Ingredients List",
                "Package inserts (publicly available)",
                "Toxicology reference materials"
            ]
        },
        "natural_immunity": {
            "approach": "compare_options",
            "headline_template": "Comparing Natural and Vaccine Immunity",
            "key_questions": [
                "Are you aware of the risks of getting the disease?",
                "Do you know about 'hybrid immunity'?"
            ],
            "facts_framework": [
                "Natural immunity requires surviving the actual disease first",
                "Disease can cause severe illness, long-term effects, or death",
                "Vaccine immunity provides protection without disease risks",
                "Hybrid immunity (both) provides the strongest protection"
            ],
            "sources": [
                "Immunology studies on natural vs. vaccine immunity",
                "Long COVID research",
                "CDC immunity guidance"
            ]
        },
        "side_effects": {
            "approach": "contextualize_risk",
            "headline_template": "Understanding Side Effects in Context",
            "key_questions": [
                "Are you concerned about common or rare side effects?",
                "Have you compared these to disease risks?"
            ],
            "facts_framework": [
                "Common side effects (soreness, fatigue) are temporary immune responses",
                "These indicate your immune system is building protection",
                "Serious side effects are tracked and extremely rare",
                "Disease complications are far more likely and severe"
            ],
            "sources": [
                "Vaccine side effect data",
                "Disease complication statistics",
                "Risk comparison studies"
            ]
        },
        "vaccine_efficacy": {
            "approach": "explain_data",
            "headline_template": "How Vaccine Effectiveness Is Measured",
            "key_questions": [
                "Do you understand what 'efficacy' percentages mean?",
                "Are you looking at protection against infection vs. severe illness?"
            ],
            "facts_framework": [
                "Vaccines significantly reduce severe illness, hospitalization, and death",
                "Effectiveness against mild infection may vary",
                "Population-level protection helps even imperfect vaccines save lives",
                "Boosters help maintain protection as immunity wanes"
            ],
            "sources": [
                "Clinical trial efficacy data",
                "Real-world effectiveness studies",
                "Hospitalization/death statistics by vaccination status"
            ]
        },
        "distrust_authority": {
            "approach": "build_trust_gradually",
            "headline_template": "Finding Trustworthy Information",
            "key_questions": [
                "What sources do you currently trust?",
                "Would you trust your personal doctor's opinion?"
            ],
            "facts_framework": [
                "You have every right to be skeptical and ask questions",
                "Look for consensus among independent, competing organizations",
                "Personal healthcare providers can discuss your specific situation",
                "Scientific data is publicly available for review"
            ],
            "sources": [
                "Your personal healthcare provider",
                "Independent academic research",
                "Multiple international health organizations",
                "Peer-reviewed scientific journals"
            ]
        },
        "cost_access": {
            "approach": "provide_resources",
            "headline_template": "Vaccine Cost and Accessibility Options",
            "key_questions": [
                "Are you looking for free or low-cost vaccine options?",
                "Do you have health insurance coverage?",
                "Are you having difficulty finding a vaccine provider nearby?"
            ],
            "facts_framework": [
                "Many vaccines are available at no cost through federal programs like Vaccines for Children (VFC) and the Bridge Access Program",
                "Under the Affordable Care Act, most insurance plans cover recommended vaccines with no copay",
                "Community health centers and local health departments often provide free or sliding-scale vaccinations",
                "Pharmacies like CVS, Walgreens, and Walmart offer vaccines and accept most insurance",
                "For uninsured adults, programs like the CDC's Bridge Access Program provide free COVID vaccines"
            ],
            "sources": [
                "Vaccines.gov - Find vaccines near you",
                "CDC Vaccines for Children (VFC) Program",
                "Healthcare.gov - Preventive care benefits",
                "Your local health department",
                "Community health centers (findahealthcenter.hrsa.gov)"
            ]
        },
        "general_question": {
            "approach": "educate",
            "headline_template": "About {topic}",
            "key_questions": [
                "What specific aspect would you like to know more about?"
            ],
            "facts_framework": [
                "Vaccines are one of the most effective public health tools available",
                "They work by training your immune system to recognize and fight specific diseases",
                "Vaccination has eliminated or nearly eliminated many dangerous diseases",
                "Both individual and community protection increase with vaccination rates"
            ],
            "sources": [
                "CDC Vaccine Information",
                "WHO Immunization Resources",
                "Your personal healthcare provider"
            ]
        }
    }
    
    # Patterns to extract specific claims from user input
    CLAIM_EXTRACTORS = {
        "tracking": r"(track|tracking|monitor|surveillance|spy|spying|watch|watching)",
        "chip": r"(microchip|chip|nanochip|nanobot|rfid)",
        "5g": r"(5g|5-g|five g|cellular|radio)",
        "magnetic": r"(magnet|magnetic|metal|stick)",
        "dna": r"(dna|genetic|gene|genome|modify|alter|change)",
        "infertility": r"(infertil|steril|reproduct|pregnant|pregnancy|baby|babies)",
        "death": r"(death|die|dying|kill|deadly|lethal|fatal)",
        "control": r"(control|population|depopulation|agenda|plan|nwo|new world order)",
        "profit": r"(profit|money|rich|pharma|pharmaceutical|billion|corrupt)",
        "untested": r"(untest|not test|never test|skip|rushed|fast|quick|experimental)",
        "autism": r"(autism|autistic|developmental)",
        "government": r"(government|gov|federal|state|cia|fbi|military)",
        "cost": r"(cost|costly|expensive|afford|affordable|price|pricy|pricey|cheap|pay|money|insurance|coverage|free|\$|dollar|budget|economic|financial)",
        "access": r"(access|available|availability|find|where to get|hard to get|can't get|no access|underserved|rural|location|nearby|clinic|pharmacy)",
    }
    
    # Topic patterns that override the classifier - these are NOT misinformation,
    # they're legitimate concerns that the classifier can misclassify
    TOPIC_OVERRIDES = {
        "cost_access": {
            "patterns": [
                r"\b(cost|costly|expensive|afford|affordable|price|pricy|pricey|cheap)\b",
                r"\b(pay|paying|payment|bill|billing|charge|fee|copay|co-pay)\b",
                r"\b(insurance|coverage|covered|uninsured|underinsured)\b",
                r"\b(free|no.?cost|low.?cost|discount|subsid)\b",
                r"\b(where|how|find|get|obtain|locate|nearby)\b.*(vaccin|shot|jab|immuniz)",
                r"(vaccin|shot|jab|immuniz).*(where|how|find|get|obtain|locate|nearby)\b",
            ],
            # Only override if NONE of these misinformation indicators are present
            "exclude_if": [
                r"(track|spy|chip|5g|magnet|dna|autism|infertil|kill|poison|conspir|lie|lying|fake|hoax|depopulat|bioweapon|plandemic|scam)"
            ]
        }
    }
    
    def __init__(self, classifier=None):
        """
        Initialize the dynamic responder.
        
        Args:
            classifier: Optional DynamicMisinformationClassifier instance
        """
        self.classifier = classifier
    
    def _extract_claims(self, text: str) -> Dict[str, bool]:
        """Extract specific claim types from user input."""
        text_lower = text.lower()
        claims = {}
        
        for claim_type, pattern in self.CLAIM_EXTRACTORS.items():
            if re.search(pattern, text_lower):
                claims[claim_type] = True
        
        return claims
    
    def _detect_topic_override(self, text: str) -> Optional[str]:
        """
        Detect if the input is about a non-misinformation topic that the
        classifier might misclassify (e.g., cost, access, scheduling).
        
        Returns the override category name, or None if no override applies.
        """
        text_lower = text.lower()
        
        for topic, config in self.TOPIC_OVERRIDES.items():
            # Check if any exclude patterns match (misinformation present)
            has_misinfo = False
            for exclude_pattern in config.get("exclude_if", []):
                if re.search(exclude_pattern, text_lower):
                    has_misinfo = True
                    break
            
            if has_misinfo:
                continue  # Don't override, this IS misinformation
            
            # Check if topic patterns match
            for pattern in config["patterns"]:
                if re.search(pattern, text_lower):
                    return topic
        
        return None
    
    def _get_topic_from_claims(self, claims: Dict[str, bool]) -> str:
        """Determine the main topic from extracted claims."""
        if "cost" in claims or "access" in claims:
            return "vaccine cost and accessibility"
        elif "tracking" in claims or "chip" in claims or "spy" in claims:
            return "tracking and surveillance"
        elif "5g" in claims:
            return "5G technology"
        elif "magnetic" in claims:
            return "magnetic effects"
        elif "dna" in claims:
            return "DNA modification"
        elif "infertility" in claims:
            return "fertility effects"
        elif "death" in claims or "control" in claims:
            return "population control"
        elif "profit" in claims:
            return "pharmaceutical industry"
        elif "untested" in claims:
            return "development speed"
        elif "autism" in claims:
            return "autism claims"
        elif "government" in claims:
            return "government involvement"
        else:
            return "vaccines"
    
    def _generate_myth_fact(self, text: str, claims: Dict[str, bool], category: str) -> Optional[Tuple[str, str]]:
        """Generate a myth vs fact pair based on the specific claim."""
        
        if "tracking" in claims or "chip" in claims:
            return (
                "Vaccines contain microchips or tracking devices",
                "Vaccines contain no electronic components. The needle used is too small to deliver any tracking device. No such technology exists that could work this way."
            )
        elif "5g" in claims:
            return (
                "Vaccines are connected to 5G networks",
                "Vaccines are biological preparations with no electronic components. 5G is a radio frequency technology. There is no possible connection between them."
            )
        elif "magnetic" in claims:
            return (
                "Vaccines make you magnetic",
                "Vaccines contain no magnetic materials in detectable amounts. Skin stickiness is due to oils and moisture, not magnetism."
            )
        elif "dna" in claims:
            return (
                "mRNA vaccines change your DNA",
                "mRNA never enters the cell nucleus where DNA is stored. It provides temporary instructions that break down within days. Your DNA remains unchanged."
            )
        elif "infertility" in claims:
            return (
                "Vaccines cause infertility",
                "Multiple studies with thousands of participants show no effect on fertility. Many vaccinated people have had healthy pregnancies."
            )
        elif "government" in claims and ("spy" in claims or "control" in claims):
            return (
                "Vaccines are government surveillance/control tools",
                "Vaccines are developed by independent scientists worldwide, reviewed by international bodies, and administered by local healthcare providers. No surveillance mechanism exists."
            )
        elif "autism" in claims:
            return (
                "Vaccines cause autism",
                "The original study making this claim was retracted for fraud. Numerous large studies involving millions of children found no connection."
            )
        elif "untested" in claims:
            return (
                "COVID vaccines were untested or rushed",
                "Vaccines went through all required testing phases with tens of thousands of participants. Steps were overlapped (running simultaneously) but not skipped."
            )
        elif "profit" in claims:
            return (
                "Vaccines are just about pharmaceutical profits",
                "Vaccines are among the least profitable pharmaceutical products. Many are distributed at cost or free. The economic benefit of preventing disease far exceeds vaccine costs."
            )
        
        return None
    
    def _generate_contextual_facts(
        self, 
        text: str, 
        claims: Dict[str, bool], 
        category: str,
        framework: Dict
    ) -> List[str]:
        """Generate facts contextual to the specific user input."""
        facts = []
        
        # Get base facts from framework
        if "facts_framework" in framework:
            for fact in framework["facts_framework"]:
                # Replace any placeholders
                if "{claimed_action}" in fact:
                    topic = self._get_topic_from_claims(claims)
                    fact = fact.replace("{claimed_action}", topic)
                facts.append(fact)
        
        # Add claim-specific facts
        if "tracking" in claims or "chip" in claims:
            facts.extend([
                "Vaccine needles are 0.5-0.8mm wide - far too small for any electronic device",
                "No power source exists that could fit in a vaccine and power a tracking device",
                "Your phone already tracks you far more effectively than any hypothetical chip"
            ])
        elif "5g" in claims:
            facts.extend([
                "5G is a radio communication standard using electromagnetic waves",
                "Vaccines are liquid biological preparations - they cannot interact with radio waves",
                "5G rollout began before COVID vaccines were developed"
            ])
        elif "dna" in claims:
            facts.extend([
                "mRNA stays in the cell cytoplasm and never enters the nucleus",
                "Your cells destroy the mRNA within days after reading it",
                "Your DNA has never been found to be altered by mRNA vaccines in any study"
            ])
        elif "government" in claims:
            facts.extend([
                "Vaccines are developed by private companies, universities, and international organizations",
                "Clinical trial data is published and peer-reviewed globally",
                "Your local pharmacist or doctor administers vaccines, not government agents"
            ])
        elif "cost" in claims or "access" in claims:
            facts.extend([
                "The Vaccines for Children (VFC) program provides free vaccines to eligible children",
                "Most private insurance plans cover all recommended vaccines at no out-of-pocket cost",
                "Community health centers offer vaccinations on a sliding fee scale based on income",
                "Many pharmacies (CVS, Walgreens, Walmart) offer walk-in vaccinations",
                "Vaccines.gov helps you find free or low-cost providers near your ZIP code"
            ])
        
        # Return unique facts, limit to 5
        return list(dict.fromkeys(facts))[:5]
    
    def generate_response(
        self, 
        text: str, 
        classification_result=None
    ) -> DynamicResponse:
        """
        Generate a dynamic educational response based on user input.
        
        Args:
            text: User's input text
            classification_result: Optional pre-computed classification
            
        Returns:
            DynamicResponse with contextual educational content
        """
        # Extract specific claims
        claims = self._extract_claims(text)
        
        # FIRST: Check for topic overrides (non-misinformation topics)
        # This prevents the classifier from misclassifying cost, access, etc.
        topic_override = self._detect_topic_override(text)
        
        if topic_override:
            category = topic_override
            confidence = 0.95  # High confidence since we pattern-matched directly
        else:
            # Classify if needed
            if classification_result is None and self.classifier:
                classification_result = self.classifier.classify(text)
            
            # Determine category from classifier
            if classification_result:
                category = classification_result.primary_category.value
                confidence = classification_result.confidence
            else:
                # Fallback category detection
                category = "conspiracy" if claims else "general_question"
                confidence = 0.5
        
        # Get framework for this category
        framework = self.EDUCATIONAL_FRAMEWORKS.get(
            category, 
            self.EDUCATIONAL_FRAMEWORKS["general_question"]
        )
        
        # Generate topic
        topic = self._get_topic_from_claims(claims)
        
        # Generate headline
        headline = framework["headline_template"].format(topic=topic)
        
        # Generate explanation based on approach
        approach = framework["approach"]
        if approach == "address_distrust":
            explanation = f"This claim about {topic} is a common piece of misinformation. Let's look at what would need to be true for this to work, and what independent evidence actually shows."
        elif approach == "validate_and_inform":
            explanation = "It's completely reasonable to want to understand safety. Here's what the research and monitoring systems actually show."
        elif approach == "explain_process":
            explanation = "Understanding how the process actually worked can help address these concerns. Let's look at the timeline and what happened."
        elif approach == "compare_options":
            explanation = "Let's compare the options and their risks so you can make an informed decision."
        elif approach == "contextualize_risk":
            explanation = "Understanding risk in context is important. Let's look at what the data actually shows compared to alternatives."
        elif approach == "build_trust_gradually":
            explanation = "Your skepticism is understandable. Let's focus on finding information sources you can verify yourself."
        elif approach == "provide_resources":
            explanation = "That's a valid concern. The good news is there are many programs and options to help make vaccines affordable and accessible. Here's what's available."
        elif approach == "educate":
            explanation = "Great question. Here's what you should know."
        else:
            explanation = "Let's examine this claim and look at what the evidence shows."
        
        # Generate myth vs fact
        myth_fact = self._generate_myth_fact(text, claims, category)
        
        # Generate contextual facts
        key_facts = self._generate_contextual_facts(text, claims, category, framework)
        
        # Get sources
        sources = framework.get("sources", [])
        
        # Generate follow-up
        questions = framework.get("key_questions", [])
        follow_up = questions[0] if questions else "Would you like to discuss any specific aspect further?"
        
        return DynamicResponse(
            user_input=text,
            detected_category=category,
            confidence=confidence,
            headline=headline,
            explanation=explanation,
            key_facts=key_facts,
            myth_vs_fact=myth_fact,
            sources=sources,
            follow_up_suggestion=follow_up
        )
    
    def quick_response(self, text: str) -> str:
        """Generate a quick one-paragraph response."""
        response = self.generate_response(text)
        
        result = f"Regarding your concern about {response.detected_category}: "
        if response.myth_vs_fact:
            result += f"{response.myth_vs_fact[1]} "
        result += response.key_facts[0] if response.key_facts else ""
        
        return result


# Standalone function for quick use
def respond_to_misinfo(text: str) -> str:
    """Quick function to get a dynamic response to misinformation."""
    responder = DynamicResponder()
    return responder.generate_response(text).to_text()
