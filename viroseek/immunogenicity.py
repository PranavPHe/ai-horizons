"""
Immunogenicity Analysis Module for ViroSeek

Compares viral sequences against known immunogenic regions from:
- IEDB (Immune Epitope Database)
- NCBI Virus database

Predicts and ranks vaccine target candidates based on:
- Immunogenicity score
- Mutation stability
- Safety likelihood
"""

import json
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Generator
from pathlib import Path
import csv
import re

# Try importing ML libraries (optional for basic functionality)
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class Epitope:
    """Represents an immunogenic epitope region."""
    sequence: str
    start: int
    end: int
    source: str  # 'iedb', 'predicted', 'ncbi'
    epitope_type: str  # 'linear', 'conformational', 'T-cell', 'B-cell'
    immunogenicity_score: float = 0.0
    mhc_restriction: Optional[str] = None
    host_species: Optional[str] = None
    assay_type: Optional[str] = None
    reference_id: Optional[str] = None


@dataclass
class VaccineCandidate:
    """Ranked vaccine target candidate."""
    gene_name: str
    protein_name: str
    sequence: str
    start: int
    end: int
    epitopes: List[Epitope] = field(default_factory=list)
    immunogenicity_score: float = 0.0
    mutation_stability: float = 0.0
    safety_score: float = 0.0
    overall_score: float = 0.0
    suggested_mrna: Optional[str] = None
    # Detailed scoring breakdown
    score_details: Dict = field(default_factory=dict)
    safety_warnings: List[str] = field(default_factory=list)
    
    def calculate_overall_score(self, weights: Dict[str, float] = None):
        """Calculate weighted overall score."""
        weights = weights or {
            'immunogenicity': 0.5,
            'stability': 0.3,
            'safety': 0.2
        }
        self.overall_score = (
            self.immunogenicity_score * weights['immunogenicity'] +
            self.mutation_stability * weights['stability'] +
            self.safety_score * weights['safety']
        )
        return self.overall_score
    
    def get_rank_grade(self) -> str:
        """Get letter grade based on overall score."""
        if self.overall_score >= 0.9:
            return "A+"
        elif self.overall_score >= 0.8:
            return "A"
        elif self.overall_score >= 0.7:
            return "B"
        elif self.overall_score >= 0.6:
            return "C"
        elif self.overall_score >= 0.5:
            return "D"
        else:
            return "F"
    
    def get_recommendation(self) -> str:
        """Get recommendation based on scores."""
        if self.overall_score >= 0.8 and self.safety_score >= 0.8:
            return "HIGHLY RECOMMENDED - Strong candidate for further validation"
        elif self.overall_score >= 0.7 and self.safety_score >= 0.7:
            return "RECOMMENDED - Good candidate with moderate confidence"
        elif self.overall_score >= 0.6:
            return "CONSIDER - Potential candidate, requires more analysis"
        elif self.safety_score < 0.6:
            return "CAUTION - Safety concerns identified"
        else:
            return "NOT RECOMMENDED - Low predicted efficacy"


class IEDBClient:
    """Client for querying IEDB (Immune Epitope Database) API."""
    
    BASE_URL = "https://query-api.iedb.org/epitope_search"
    TCELL_URL = "https://query-api.iedb.org/tcell_search"
    BCELL_URL = "https://query-api.iedb.org/bcell_search"
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
    
    def search_epitopes(
        self,
        sequence: str = None,
        organism: str = None,
        host: str = None,
        epitope_type: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Search IEDB for known epitopes.
        
        Args:
            sequence: Peptide sequence to search
            organism: Source organism (e.g., 'SARS-CoV-2')
            host: Host species (e.g., 'human')
            epitope_type: 'linear' or 'conformational'
            limit: Maximum results to return
        """
        params = {}
        if sequence:
            params['linear_sequence'] = sequence
        if organism:
            params['source_organism'] = organism
        if host:
            params['host_organism'] = host
        if epitope_type:
            params['epitope_type'] = epitope_type
        params['limit'] = limit
        
        return self._make_request(self.BASE_URL, params)
    
    def search_tcell_epitopes(
        self,
        sequence: str = None,
        organism: str = None,
        mhc_class: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """Search for T-cell epitopes in IEDB."""
        params = {}
        if sequence:
            params['linear_sequence'] = sequence
        if organism:
            params['source_organism'] = organism
        if mhc_class:
            params['mhc_class'] = mhc_class
        params['limit'] = limit
        
        return self._make_request(self.TCELL_URL, params)
    
    def search_bcell_epitopes(
        self,
        sequence: str = None,
        organism: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """Search for B-cell epitopes in IEDB."""
        params = {}
        if sequence:
            params['linear_sequence'] = sequence
        if organism:
            params['source_organism'] = organism
        params['limit'] = limit
        
        return self._make_request(self.BCELL_URL, params)
    
    def _make_request(self, url: str, params: Dict) -> List[Dict]:
        """Make HTTP request to IEDB API."""
        query_string = urllib.parse.urlencode(params)
        full_url = f"{url}?{query_string}"
        
        try:
            req = urllib.request.Request(
                full_url,
                headers={'Accept': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                data = json.loads(response.read().decode('utf-8'))
                return data if isinstance(data, list) else data.get('results', [])
        except Exception as e:
            print(f"IEDB API error: {e}")
            return []


class EpitopePrediction:
    """
    ML-based epitope prediction for viral sequences.
    
    Uses sequence-based features to predict immunogenic regions:
    - Hydrophobicity profiles
    - Accessibility predictions
    - Antigenicity indices
    - MHC binding predictions
    """
    
    # Amino acid properties for feature extraction
    HYDROPHOBICITY = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    
    # Parker hydrophilicity scale (for B-cell epitopes)
    HYDROPHILICITY = {
        'A': 2.1, 'R': 4.2, 'N': 7.0, 'D': 10.0, 'C': 1.4,
        'Q': 6.0, 'E': 7.8, 'G': 5.7, 'H': 2.1, 'I': -8.0,
        'L': -9.2, 'K': 5.7, 'M': -4.2, 'F': -9.2, 'P': 2.1,
        'S': 6.5, 'T': 5.2, 'W': -10.0, 'Y': -1.9, 'V': -3.7
    }
    
    # Antigenicity index (Kolaskar and Tongaonkar)
    ANTIGENICITY = {
        'A': 1.064, 'R': 0.873, 'N': 0.776, 'D': 0.866, 'C': 1.412,
        'Q': 0.761, 'E': 0.851, 'G': 0.874, 'H': 1.105, 'I': 1.152,
        'L': 1.250, 'K': 0.930, 'M': 0.826, 'F': 1.091, 'P': 1.064,
        'S': 1.012, 'T': 0.909, 'W': 0.893, 'Y': 1.161, 'V': 1.383
    }
    
    # Surface accessibility (Emini scale)
    ACCESSIBILITY = {
        'A': 0.815, 'R': 1.475, 'N': 1.296, 'D': 1.283, 'C': 0.394,
        'Q': 1.348, 'E': 1.445, 'G': 0.714, 'H': 1.180, 'I': 0.603,
        'L': 0.603, 'K': 1.545, 'M': 0.714, 'F': 0.603, 'P': 1.034,
        'S': 1.115, 'T': 1.184, 'W': 0.603, 'Y': 1.089, 'V': 0.603
    }
    
    def __init__(self, window_size: int = 9):
        self.window_size = window_size
    
    def predict_bcell_epitopes(
        self,
        sequence: str,
        threshold: float = 0.5
    ) -> List[Epitope]:
        """
        Predict B-cell epitopes using sequence properties.
        
        Uses combined score from:
        - Hydrophilicity
        - Surface accessibility
        - Antigenicity
        """
        epitopes = []
        seq = sequence.upper()
        
        if len(seq) < self.window_size:
            return epitopes
        
        scores = self._calculate_window_scores(seq, 'bcell')
        
        # Find regions above threshold
        in_epitope = False
        start = 0
        
        for i, score in enumerate(scores):
            if score >= threshold and not in_epitope:
                start = i
                in_epitope = True
            elif score < threshold and in_epitope:
                end = i + self.window_size - 1
                epitope_seq = seq[start:end]
                avg_score = sum(scores[start:i]) / (i - start)
                
                epitopes.append(Epitope(
                    sequence=epitope_seq,
                    start=start + 1,  # 1-indexed
                    end=end,
                    source='predicted',
                    epitope_type='B-cell',
                    immunogenicity_score=min(avg_score, 1.0)
                ))
                in_epitope = False
        
        # Handle epitope at end of sequence
        if in_epitope:
            end = len(seq)
            epitope_seq = seq[start:end]
            avg_score = sum(scores[start:]) / (len(scores) - start)
            epitopes.append(Epitope(
                sequence=epitope_seq,
                start=start + 1,
                end=end,
                source='predicted',
                epitope_type='B-cell',
                immunogenicity_score=min(avg_score, 1.0)
            ))
        
        return epitopes
    
    def predict_tcell_epitopes(
        self,
        sequence: str,
        threshold: float = 0.5
    ) -> List[Epitope]:
        """
        Predict T-cell epitopes (MHC binding peptides).
        
        Uses 9-mer windows typical for MHC-I binding.
        """
        epitopes = []
        seq = sequence.upper()
        
        # T-cell epitopes are typically 8-11 amino acids
        for length in [8, 9, 10, 11]:
            if len(seq) < length:
                continue
                
            for i in range(len(seq) - length + 1):
                peptide = seq[i:i + length]
                score = self._calculate_mhc_binding_score(peptide)
                
                if score >= threshold:
                    epitopes.append(Epitope(
                        sequence=peptide,
                        start=i + 1,
                        end=i + length,
                        source='predicted',
                        epitope_type='T-cell',
                        immunogenicity_score=score
                    ))
        
        # Merge overlapping epitopes
        return self._merge_overlapping_epitopes(epitopes)
    
    def _calculate_window_scores(
        self,
        sequence: str,
        epitope_type: str
    ) -> List[float]:
        """Calculate sliding window scores for epitope prediction."""
        scores = []
        
        for i in range(len(sequence) - self.window_size + 1):
            window = sequence[i:i + self.window_size]
            
            if epitope_type == 'bcell':
                # B-cell: hydrophilicity + accessibility + antigenicity
                hydro = self._calc_property_score(window, self.HYDROPHILICITY)
                access = self._calc_property_score(window, self.ACCESSIBILITY)
                antig = self._calc_property_score(window, self.ANTIGENICITY)
                
                # Normalize and combine
                score = (
                    self._normalize(hydro, -10, 10) * 0.4 +
                    self._normalize(access, 0, 2) * 0.3 +
                    self._normalize(antig, 0.7, 1.5) * 0.3
                )
            else:
                score = self._calculate_mhc_binding_score(window)
            
            scores.append(score)
        
        return scores
    
    def _calculate_mhc_binding_score(self, peptide: str) -> float:
        """
        Simplified MHC binding prediction.
        
        Real implementation would use trained neural networks
        (e.g., NetMHC, MHCflurry) but this provides a reasonable
        approximation based on anchor residue preferences.
        """
        score = 0.5  # Base score
        
        # MHC-I anchor positions (positions 2 and C-terminus)
        if len(peptide) >= 9:
            # Position 2 preferences
            pos2 = peptide[1]
            if pos2 in 'LMI':  # Hydrophobic preferred
                score += 0.15
            elif pos2 in 'AVFYW':
                score += 0.1
            
            # C-terminal preferences
            c_term = peptide[-1]
            if c_term in 'LVIKY':
                score += 0.15
            elif c_term in 'AW':
                score += 0.1
            
            # Internal hydrophobic residues
            hydro_count = sum(1 for aa in peptide[2:-1] if aa in 'LVIFWM')
            score += min(hydro_count * 0.02, 0.1)
        
        return min(score, 1.0)
    
    def _calc_property_score(self, peptide: str, prop_dict: Dict) -> float:
        """Calculate average property score for a peptide."""
        total = 0
        count = 0
        for aa in peptide:
            if aa in prop_dict:
                total += prop_dict[aa]
                count += 1
        return total / count if count > 0 else 0
    
    @staticmethod
    def _normalize(value: float, min_val: float, max_val: float) -> float:
        """Normalize value to 0-1 range."""
        return max(0, min(1, (value - min_val) / (max_val - min_val)))
    
    def _merge_overlapping_epitopes(
        self,
        epitopes: List[Epitope],
        max_gap: int = 3
    ) -> List[Epitope]:
        """Merge overlapping or adjacent epitopes."""
        if not epitopes:
            return []
        
        # Sort by start position
        sorted_epitopes = sorted(epitopes, key=lambda e: e.start)
        merged = [sorted_epitopes[0]]
        
        for epitope in sorted_epitopes[1:]:
            last = merged[-1]
            if epitope.start <= last.end + max_gap:
                # Merge
                merged[-1] = Epitope(
                    sequence=epitope.sequence,  # Will be recalculated
                    start=last.start,
                    end=max(last.end, epitope.end),
                    source=last.source,
                    epitope_type=last.epitope_type,
                    immunogenicity_score=max(
                        last.immunogenicity_score,
                        epitope.immunogenicity_score
                    )
                )
            else:
                merged.append(epitope)
        
        return merged


class MutationAnalyzer:
    """Analyze mutation stability of potential vaccine targets."""
    
    # Conservation weights (higher = more conserved)
    CONSERVATION_WEIGHTS = {
        'C': 0.9, 'W': 0.85, 'H': 0.8, 'Y': 0.75,
        'F': 0.7, 'P': 0.7, 'G': 0.65, 'D': 0.6,
        'E': 0.6, 'N': 0.55, 'Q': 0.55, 'M': 0.5,
        'K': 0.5, 'R': 0.5, 'I': 0.45, 'L': 0.45,
        'V': 0.4, 'A': 0.35, 'T': 0.35, 'S': 0.3
    }
    
    def calculate_stability_score(self, sequence: str) -> tuple[float, Dict]:
        """
        Calculate mutation stability score based on:
        - Amino acid conservation propensity
        - Structural constraints (Cys-Cys, Pro)
        
        Returns:
            Tuple of (score, details dict)
        """
        details = {
            'sequence_length': 0,
            'conservation_score': 0.0,
            'cysteine_count': 0,
            'cysteine_bonus': 0.0,
            'proline_count': 0,
            'proline_bonus': 0.0,
            'scoring_factors': []
        }
        
        if not sequence:
            return 0.0, details
        
        seq = sequence.upper()
        details['sequence_length'] = len(seq)
        
        # Base conservation score
        conservation = sum(
            self.CONSERVATION_WEIGHTS.get(aa, 0.3)
            for aa in seq
        ) / len(seq)
        details['conservation_score'] = conservation
        details['scoring_factors'].append(f"Base conservation: {conservation:.3f}")
        
        # Bonus for cysteines (potential disulfide bonds)
        cys_count = seq.count('C')
        cys_bonus = min(cys_count * 0.05, 0.15)
        details['cysteine_count'] = cys_count
        details['cysteine_bonus'] = cys_bonus
        if cys_bonus > 0:
            details['scoring_factors'].append(f"Cysteine bonds ({cys_count}): +{cys_bonus:.3f}")
        
        # Bonus for prolines (structural rigidity)
        pro_count = seq.count('P')
        pro_bonus = min(pro_count * 0.02, 0.1)
        details['proline_count'] = pro_count
        details['proline_bonus'] = pro_bonus
        if pro_bonus > 0:
            details['scoring_factors'].append(f"Proline rigidity ({pro_count}): +{pro_bonus:.3f}")
        
        final_score = min(conservation + cys_bonus + pro_bonus, 1.0)
        details['final_score'] = final_score
        
        return final_score, details
    
    def analyze_mutation_hotspots(
        self,
        reference_seq: str,
        variant_seqs: List[str]
    ) -> Dict[int, float]:
        """
        Identify mutation hotspots by comparing sequences.
        Returns position -> mutation frequency mapping.
        """
        if not variant_seqs:
            return {}
        
        hotspots = {}
        ref = reference_seq.upper()
        
        for pos, ref_aa in enumerate(ref):
            mutations = 0
            for variant in variant_seqs:
                if pos < len(variant) and variant[pos].upper() != ref_aa:
                    mutations += 1
            
            hotspots[pos] = mutations / len(variant_seqs)
        
        return hotspots


class SafetyAnalyzer:
    """Analyze safety profile of vaccine candidates."""
    
    # Potentially problematic motifs
    UNSAFE_MOTIFS = [
        ('RKKR', 'furin cleavage site'),
        ('RKRR', 'furin cleavage site'),
        ('PRRA', 'furin cleavage insertion'),
        # Potential autoimmune-triggering sequences
        # (simplified - real analysis would use epitope databases)
    ]
    
    # Known toxic peptide patterns
    TOXIC_PATTERNS = [
        r'C.{2}C.{3}C',  # Potential metal binding
    ]
    
    def calculate_safety_score(
        self,
        sequence: str,
        host_proteome_check: bool = False
    ) -> tuple[float, List[str]]:
        """
        Calculate safety score and return warnings.
        
        Checks for:
        - Known unsafe motifs
        - Potential cross-reactivity with host
        - Toxic peptide patterns
        """
        warnings = []
        score = 1.0
        seq = sequence.upper()
        
        # Check unsafe motifs
        for motif, description in self.UNSAFE_MOTIFS:
            if motif in seq:
                warnings.append(f"Contains {description}: {motif}")
                score -= 0.15
        
        # Check toxic patterns
        for pattern in self.TOXIC_PATTERNS:
            if re.search(pattern, seq):
                warnings.append(f"Potential toxic pattern: {pattern}")
                score -= 0.1
        
        # Length check (very long sequences may have issues)
        if len(seq) > 1000:
            warnings.append("Sequence length >1000 aa may affect expression")
            score -= 0.05
        
        # Hydrophobic stretches (aggregation prone)
        hydro_stretch = re.search(r'[LIVFWM]{8,}', seq)
        if hydro_stretch:
            warnings.append(f"Long hydrophobic stretch: {hydro_stretch.group()}")
            score -= 0.1
        
        return max(score, 0.0), warnings


class ImmunogenicityAnalyzer:
    """
    Main analyzer class for immunogenic region identification and ranking.
    """
    
    def __init__(
        self,
        use_iedb: bool = True,
        organism: str = None,
        host: str = "human"
    ):
        self.iedb_client = IEDBClient() if use_iedb else None
        self.epitope_predictor = EpitopePrediction()
        self.mutation_analyzer = MutationAnalyzer()
        self.safety_analyzer = SafetyAnalyzer()
        self.organism = organism
        self.host = host
    
    def analyze_sequence(
        self,
        sequence: str,
        gene_name: str = "unknown",
        protein_name: str = "unknown",
        compare_iedb: bool = True,
        predict_epitopes: bool = True
    ) -> VaccineCandidate:
        """
        Analyze a protein sequence for vaccine candidacy.
        
        Args:
            sequence: Protein/peptide sequence
            gene_name: Gene name/identifier
            protein_name: Protein name
            compare_iedb: Query IEDB for known epitopes
            predict_epitopes: Predict epitopes using ML
        
        Returns:
            VaccineCandidate with scores and epitope information
        """
        epitopes = []
        
        # Query IEDB for known epitopes
        if compare_iedb and self.iedb_client:
            iedb_epitopes = self._query_iedb(sequence)
            epitopes.extend(iedb_epitopes)
        
        # Predict epitopes using ML models
        if predict_epitopes:
            bcell = self.epitope_predictor.predict_bcell_epitopes(sequence)
            tcell = self.epitope_predictor.predict_tcell_epitopes(sequence)
            epitopes.extend(bcell)
            epitopes.extend(tcell)
        
        # Calculate scores
        immunogenicity, immuno_details = self._calculate_immunogenicity(epitopes, sequence)
        stability, stability_details = self.mutation_analyzer.calculate_stability_score(sequence)
        safety, warnings = self.safety_analyzer.calculate_safety_score(sequence)
        
        # Build detailed scoring breakdown
        score_details = {
            'immunogenicity': immuno_details,
            'stability': stability_details,
            'safety': {
                'score': safety,
                'warnings': warnings,
                'checks_passed': len(warnings) == 0
            }
        }
        
        candidate = VaccineCandidate(
            gene_name=gene_name,
            protein_name=protein_name,
            sequence=sequence,
            start=1,
            end=len(sequence),
            epitopes=epitopes,
            immunogenicity_score=immunogenicity,
            mutation_stability=stability,
            safety_score=safety,
            score_details=score_details,
            safety_warnings=warnings
        )
        candidate.calculate_overall_score()
        
        # Generate suggested mRNA sequence
        candidate.suggested_mrna = self._generate_mrna_suggestion(sequence)
        
        return candidate
    
    def _query_iedb(self, sequence: str) -> List[Epitope]:
        """Query IEDB for epitopes matching the sequence."""
        epitopes = []
        
        # Search in chunks (IEDB has length limits)
        chunk_size = 50
        for i in range(0, len(sequence), chunk_size - 10):
            chunk = sequence[i:i + chunk_size]
            
            results = self.iedb_client.search_epitopes(
                sequence=chunk,
                organism=self.organism,
                host=self.host,
                limit=50
            )
            
            for result in results:
                epitopes.append(Epitope(
                    sequence=result.get('linear_sequence', chunk),
                    start=i + 1,
                    end=i + len(chunk),
                    source='iedb',
                    epitope_type=result.get('epitope_type', 'unknown'),
                    immunogenicity_score=0.8,  # IEDB matches are high confidence
                    reference_id=result.get('epitope_id')
                ))
        
        return epitopes
    
    def _calculate_immunogenicity(
        self,
        epitopes: List[Epitope],
        sequence: str
    ) -> tuple[float, Dict]:
        """Calculate overall immunogenicity score with detailed breakdown."""
        details = {
            'total_epitopes': len(epitopes),
            'bcell_epitopes': 0,
            'tcell_epitopes': 0,
            'iedb_matches': 0,
            'avg_epitope_score': 0.0,
            'epitope_density': 0.0,
            'has_diversity_bonus': False,
            'scoring_factors': []
        }
        
        if not epitopes:
            details['scoring_factors'].append("No epitopes found - base score applied")
            return 0.3, details
        
        # Count epitope types
        bcell = [e for e in epitopes if e.epitope_type == 'B-cell']
        tcell = [e for e in epitopes if e.epitope_type == 'T-cell']
        iedb = [e for e in epitopes if e.source == 'iedb']
        
        details['bcell_epitopes'] = len(bcell)
        details['tcell_epitopes'] = len(tcell)
        details['iedb_matches'] = len(iedb)
        details['epitope_density'] = len(epitopes) / len(sequence) if sequence else 0
        
        # Average epitope scores weighted by source confidence
        source_weights = {'iedb': 1.0, 'ncbi': 0.9, 'predicted': 0.6}
        
        total_score = 0
        total_weight = 0
        
        for epitope in epitopes:
            weight = source_weights.get(epitope.source, 0.5)
            total_score += epitope.immunogenicity_score * weight
            total_weight += weight
        
        avg_score = total_score / total_weight if total_weight > 0 else 0.3
        details['avg_epitope_score'] = avg_score
        details['scoring_factors'].append(f"Base epitope score: {avg_score:.3f}")
        
        # Bonus for multiple epitopes
        epitope_bonus = min(len(epitopes) * 0.02, 0.2)
        if epitope_bonus > 0:
            details['scoring_factors'].append(f"Multiple epitopes bonus: +{epitope_bonus:.3f}")
        
        # Bonus for both T-cell and B-cell epitopes
        has_tcell = len(tcell) > 0
        has_bcell = len(bcell) > 0
        diversity_bonus = 0.1 if (has_tcell and has_bcell) else 0
        details['has_diversity_bonus'] = diversity_bonus > 0
        if diversity_bonus > 0:
            details['scoring_factors'].append("T-cell + B-cell diversity bonus: +0.1")
        
        final_score = min(avg_score + epitope_bonus + diversity_bonus, 1.0)
        details['final_score'] = final_score
        
        return final_score, details
    
    def _generate_mrna_suggestion(self, protein_sequence: str) -> str:
        """
        Generate suggested mRNA sequence for the protein.
        Uses human-optimized codon table.
        """
        # Human-optimized codon table (most frequent codons)
        CODON_TABLE = {
            'A': 'GCC', 'R': 'CGG', 'N': 'AAC', 'D': 'GAC',
            'C': 'TGC', 'Q': 'CAG', 'E': 'GAG', 'G': 'GGC',
            'H': 'CAC', 'I': 'ATC', 'L': 'CTG', 'K': 'AAG',
            'M': 'ATG', 'F': 'TTC', 'P': 'CCC', 'S': 'AGC',
            'T': 'ACC', 'W': 'TGG', 'Y': 'TAC', 'V': 'GTG',
            '*': 'TGA'  # Stop codon
        }
        
        mrna = []
        for aa in protein_sequence.upper():
            if aa in CODON_TABLE:
                mrna.append(CODON_TABLE[aa])
            else:
                mrna.append('NNN')  # Unknown amino acid
        
        # Add stop codon if not present
        if not protein_sequence.endswith('*'):
            mrna.append(CODON_TABLE['*'])
        
        return ''.join(mrna)
    
    def rank_candidates(
        self,
        candidates: List[VaccineCandidate],
        top_n: int = None
    ) -> List[VaccineCandidate]:
        """Rank vaccine candidates by overall score."""
        # Recalculate scores to ensure consistency
        for candidate in candidates:
            candidate.calculate_overall_score()
        
        ranked = sorted(
            candidates,
            key=lambda c: c.overall_score,
            reverse=True
        )
        
        return ranked[:top_n] if top_n else ranked


def generate_report(
    candidates: List[VaccineCandidate],
    output_path: str = None,
    format: str = "text"
) -> str:
    """
    Generate analysis report for vaccine candidates.
    
    Args:
        candidates: Ranked list of vaccine candidates
        output_path: Optional file path to save report
        format: 'text', 'csv', or 'json'
    
    Returns:
        Report content as string
    """
    if format == "json":
        report = _generate_json_report(candidates)
    elif format == "csv":
        report = _generate_csv_report(candidates)
    else:
        report = _generate_text_report(candidates)
    
    if output_path:
        Path(output_path).write_text(report)
        print(f"Report saved to: {output_path}")
    
    return report


def _generate_text_report(candidates: List[VaccineCandidate]) -> str:
    """Generate human-readable text report."""
    lines = [
        "=" * 70,
        "VIROSEEK - VACCINE TARGET ANALYSIS REPORT",
        "=" * 70,
        "",
        f"Total candidates analyzed: {len(candidates)}",
        "",
        "RANKING CRITERIA:",
        "  - Immunogenicity (50%): Predicted immune response strength",
        "  - Mutation Stability (30%): Resistance to viral mutation",
        "  - Safety (20%): Absence of harmful motifs",
        "",
    ]
    
    # Summary table
    lines.append("SUMMARY RANKING:")
    lines.append("-" * 70)
    lines.append(f"{'Rank':<5} {'Grade':<6} {'Gene':<15} {'Overall':<10} {'Immuno':<10} {'Stable':<10} {'Safe':<10}")
    lines.append("-" * 70)
    
    for i, c in enumerate(candidates, 1):
        lines.append(
            f"{i:<5} {c.get_rank_grade():<6} {c.gene_name[:14]:<15} "
            f"{c.overall_score:<10.3f} {c.immunogenicity_score:<10.3f} "
            f"{c.mutation_stability:<10.3f} {c.safety_score:<10.3f}"
        )
    lines.append("-" * 70)
    lines.append("")
    
    for i, candidate in enumerate(candidates, 1):
        lines.extend([
            f"{'=' * 70}",
            f"RANK #{i}: {candidate.gene_name} - {candidate.protein_name}",
            f"Grade: {candidate.get_rank_grade()}",
            f"{'=' * 70}",
            f"",
            f">>> {candidate.get_recommendation()}",
            f"",
            f"Sequence length: {len(candidate.sequence)} aa",
            f"Position: {candidate.start} - {candidate.end}",
            f"",
            f"SCORES:",
            f"  Overall Score:        {candidate.overall_score:.3f}",
            f"  Immunogenicity:       {candidate.immunogenicity_score:.3f}",
            f"  Mutation Stability:   {candidate.mutation_stability:.3f}",
            f"  Safety Score:         {candidate.safety_score:.3f}",
        ])
        
        # Detailed scoring breakdown
        if candidate.score_details:
            lines.append("")
            lines.append("SCORING BREAKDOWN:")
            
            # Immunogenicity details
            if 'immunogenicity' in candidate.score_details:
                imm = candidate.score_details['immunogenicity']
                lines.append("  Immunogenicity factors:")
                for factor in imm.get('scoring_factors', []):
                    lines.append(f"    - {factor}")
            
            # Stability details
            if 'stability' in candidate.score_details:
                stab = candidate.score_details['stability']
                lines.append("  Stability factors:")
                for factor in stab.get('scoring_factors', []):
                    lines.append(f"    - {factor}")
        
        # Safety warnings
        if candidate.safety_warnings:
            lines.append("")
            lines.append("⚠ SAFETY WARNINGS:")
            for warning in candidate.safety_warnings:
                lines.append(f"    - {warning}")
        
        lines.append("")
        lines.append(f"EPITOPES FOUND: {len(candidate.epitopes)}")
        
        # Group epitopes by type
        bcell = [e for e in candidate.epitopes if e.epitope_type == 'B-cell']
        tcell = [e for e in candidate.epitopes if e.epitope_type == 'T-cell']
        iedb = [e for e in candidate.epitopes if e.source == 'iedb']
        
        lines.append(f"  B-cell epitopes: {len(bcell)}")
        lines.append(f"  T-cell epitopes: {len(tcell)}")
        lines.append(f"  IEDB matches:    {len(iedb)}")
        
        if candidate.epitopes:
            lines.append("")
            lines.append("  Top 5 epitopes by score:")
            for epitope in sorted(
                candidate.epitopes,
                key=lambda e: e.immunogenicity_score,
                reverse=True
            )[:5]:
                seq_display = epitope.sequence[:20] + ('...' if len(epitope.sequence) > 20 else '')
                lines.append(
                    f"    - {seq_display:<23} "
                    f"({epitope.epitope_type}, pos: {epitope.start}-{epitope.end}, "
                    f"score: {epitope.immunogenicity_score:.2f})"
                )
        
        if candidate.suggested_mrna:
            lines.append("")
            lines.append(f"SUGGESTED mRNA SEQUENCE (first 90 nt):")
            lines.append(f"  5'-{candidate.suggested_mrna[:90]}...-3'")
            lines.append(f"  Total length: {len(candidate.suggested_mrna)} nt")
        
        lines.append("")
    
    return "\n".join(lines)


def _generate_csv_report(candidates: List[VaccineCandidate]) -> str:
    """Generate CSV format report."""
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        'Rank', 'Gene', 'Protein', 'Length', 'Overall_Score',
        'Immunogenicity', 'Stability', 'Safety',
        'Bcell_Epitopes', 'Tcell_Epitopes', 'IEDB_Matches'
    ])
    
    for i, c in enumerate(candidates, 1):
        bcell = len([e for e in c.epitopes if e.epitope_type == 'B-cell'])
        tcell = len([e for e in c.epitopes if e.epitope_type == 'T-cell'])
        iedb = len([e for e in c.epitopes if e.source == 'iedb'])
        
        writer.writerow([
            i, c.gene_name, c.protein_name, len(c.sequence),
            f"{c.overall_score:.3f}", f"{c.immunogenicity_score:.3f}",
            f"{c.mutation_stability:.3f}", f"{c.safety_score:.3f}",
            bcell, tcell, iedb
        ])
    
    return output.getvalue()


def _generate_json_report(candidates: List[VaccineCandidate]) -> str:
    """Generate JSON format report."""
    data = {
        'total_candidates': len(candidates),
        'candidates': []
    }
    
    for i, c in enumerate(candidates, 1):
        data['candidates'].append({
            'rank': i,
            'grade': c.get_rank_grade(),
            'recommendation': c.get_recommendation(),
            'gene_name': c.gene_name,
            'protein_name': c.protein_name,
            'sequence': c.sequence,
            'length': len(c.sequence),
            'scores': {
                'overall': c.overall_score,
                'immunogenicity': c.immunogenicity_score,
                'mutation_stability': c.mutation_stability,
                'safety': c.safety_score
            },
            'score_details': c.score_details,
            'safety_warnings': c.safety_warnings,
            'epitopes': [
                {
                    'sequence': e.sequence,
                    'start': e.start,
                    'end': e.end,
                    'type': e.epitope_type,
                    'source': e.source,
                    'score': e.immunogenicity_score
                }
                for e in c.epitopes
            ],
            'epitope_summary': {
                'total': len(c.epitopes),
                'bcell': len([e for e in c.epitopes if e.epitope_type == 'B-cell']),
                'tcell': len([e for e in c.epitopes if e.epitope_type == 'T-cell']),
                'iedb_matches': len([e for e in c.epitopes if e.source == 'iedb'])
            },
            'suggested_mrna': c.suggested_mrna,
            'mrna_length': len(c.suggested_mrna) if c.suggested_mrna else 0
        })
    
    return json.dumps(data, indent=2)
