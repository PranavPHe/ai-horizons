"""
ViroSeek Flask API Server

Provides REST endpoints for the web frontend to call the analysis backend.
"""

import json
import sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from Bio import Entrez, SeqIO
from io import StringIO

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from immunogenicity import ImmunogenicityAnalyzer, VaccineCandidate
from visualization import generate_mrna_variants

# Configure NCBI Entrez
Entrez.email = "viroseek_api@example.com"

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests


def parse_fasta_string(fasta_str: str):
    """Parse FASTA format string into sequence records."""
    handle = StringIO(fasta_str)
    return list(SeqIO.parse(handle, "fasta"))


def fetch_from_ncbi(accession_id: str, db: str = "nucleotide"):
    """Fetch sequence from NCBI by accession ID."""
    with Entrez.efetch(db=db, id=accession_id, rettype="gb", retmode="text") as handle:
        return list(SeqIO.parse(handle, "genbank"))


def candidate_to_dict(c: VaccineCandidate, include_mrna: bool = True) -> dict:
    """Convert VaccineCandidate to JSON-serializable dict."""
    result = {
        "gene_name": c.gene_name,
        "protein_name": c.protein_name,
        "sequence": c.sequence,
        "length": len(c.sequence),
        "scores": {
            "overall": round(c.overall_score, 4),
            "immunogenicity": round(c.immunogenicity_score, 4),
            "mutation_stability": round(c.mutation_stability, 4),
            "safety": round(c.safety_score, 4)
        },
        "grade": c.get_rank_grade(),
        "recommendation": c.get_recommendation(),
        "epitope_summary": {
            "total": len(c.epitopes),
            "bcell": len([e for e in c.epitopes if e.epitope_type == "B-cell"]),
            "tcell": len([e for e in c.epitopes if e.epitope_type == "T-cell"]),
            "iedb_matches": len([e for e in c.epitopes if e.source == "iedb"])
        },
        "safety_warnings": c.safety_warnings
    }
    
    if include_mrna and c.suggested_mrna:
        result["suggested_mrna"] = c.suggested_mrna
        result["mrna_length"] = len(c.suggested_mrna)
    
    return result


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "viroseek-api"})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Run immunogenicity analysis on submitted sequences.
    
    Expected JSON payload:
    {
        "input": "NC_045512" or FASTA string,
        "input_type": "accession" or "fasta",
        "virus_name": "SARS-CoV-2",
        "variant": "wild-type",
        "host_tax_id": "9606",
        "pdb_id": "6VXX" (optional),
        "top_n": 10,
        "weights": {
            "immunogenicity": 0.5,
            "stability": 0.3,
            "safety": 0.2
        },
        "use_iedb": true,
        "generate_mrna": true
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Extract parameters
        input_source = data.get("input", "")
        input_type = data.get("input_type", "accession")
        virus_name = data.get("virus_name", "Unknown Virus")
        variant = data.get("variant", "wild-type")
        host_tax_id = data.get("host_tax_id", "9606")
        pdb_id = data.get("pdb_id")
        top_n = int(data.get("top_n", 10))
        weights = data.get("weights", {
            "immunogenicity": 0.5,
            "stability": 0.3,
            "safety": 0.2
        })
        use_iedb = data.get("use_iedb", True)
        generate_mrna = data.get("generate_mrna", True)
        
        if not input_source:
            return jsonify({"error": "No input sequence provided"}), 400
        
        # Load sequences
        records = []
        if input_type == "accession":
            try:
                records = fetch_from_ncbi(input_source)
            except Exception as e:
                return jsonify({"error": f"Failed to fetch from NCBI: {str(e)}"}), 400
        else:
            # Parse FASTA string
            try:
                records = parse_fasta_string(input_source)
            except Exception as e:
                return jsonify({"error": f"Failed to parse FASTA: {str(e)}"}), 400
        
        if not records:
            return jsonify({"error": "No sequences found in input"}), 400
        
        # Initialize analyzer
        analyzer = ImmunogenicityAnalyzer(
            use_iedb=use_iedb,
            organism=virus_name,
            host="human"
        )
        
        candidates = []
        
        # Analyze each record
        for record in records:
            # Track if we found CDS features for this record
            found_cds_for_record = False
            
            # Extract protein sequences from CDS features
            if hasattr(record, 'features'):
                for feature in record.features:
                    if feature.type == "CDS":
                        gene_name = feature.qualifiers.get('gene', ['unknown'])[0]
                        protein_name = feature.qualifiers.get('product', ['unknown'])[0]
                        
                        # Get protein sequence
                        if 'translation' in feature.qualifiers:
                            protein_seq = feature.qualifiers['translation'][0]
                        else:
                            try:
                                protein_seq = str(feature.extract(record.seq).translate())
                            except Exception:
                                continue
                        
                        if len(protein_seq) < 10:
                            continue
                        
                        candidate = analyzer.analyze_sequence(
                            sequence=protein_seq,
                            gene_name=gene_name,
                            protein_name=protein_name,
                            compare_iedb=use_iedb,
                            predict_epitopes=True
                        )
                        candidates.append(candidate)
                        found_cds_for_record = True
            
            # If no CDS features found for this record, try to translate the whole sequence
            if not found_cds_for_record and len(record.seq) >= 30:
                try:
                    # Check if it's protein or nucleotide
                    seq_str = str(record.seq)
                    is_protein = all(c in "ACDEFGHIKLMNPQRSTVWY*" for c in seq_str.upper())
                    
                    if is_protein:
                        protein_seq = seq_str
                    else:
                        protein_seq = str(record.seq.translate())
                    
                    candidate = analyzer.analyze_sequence(
                        sequence=protein_seq,
                        gene_name=record.id,
                        protein_name=record.description[:50],
                        compare_iedb=use_iedb,
                        predict_epitopes=True
                    )
                    candidates.append(candidate)
                except Exception as e:
                    pass  # Skip sequences that can't be translated
        
        if not candidates:
            return jsonify({"error": "No protein sequences found for analysis"}), 400
        
        # Apply custom weights
        for c in candidates:
            c.calculate_overall_score(weights)
        
        # Rank candidates
        ranked = analyzer.rank_candidates(candidates, top_n=top_n)
        
        # Generate mRNA if requested
        if generate_mrna:
            for c in ranked:
                try:
                    variants = generate_mrna_variants(c.sequence)
                    # Use the human-optimized variant
                    if 'human' in variants:
                        c.suggested_mrna = variants['human']['sequence']
                except Exception:
                    pass
        
        # Build response
        response = {
            "success": True,
            "metadata": {
                "virus_name": virus_name,
                "variant": variant,
                "host_tax_id": host_tax_id,
                "pdb_id": pdb_id,
                "total_sequences": len(records),
                "total_candidates": len(candidates),
                "top_n": top_n,
                "weights": weights
            },
            "candidates": [candidate_to_dict(c, generate_mrna) for i, c in enumerate(ranked, 1)]
        }
        
        # Add rank numbers
        for i, c in enumerate(response["candidates"], 1):
            c["rank"] = i
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route("/api/fetch-sequence", methods=["POST"])
def fetch_sequence():
    """
    Fetch sequence info from NCBI without running analysis.
    
    Expected JSON payload:
    {
        "accession": "NC_045512"
    }
    """
    try:
        data = request.get_json()
        accession = data.get("accession", "")
        
        if not accession:
            return jsonify({"error": "No accession ID provided"}), 400
        
        records = fetch_from_ncbi(accession)
        
        if not records:
            return jsonify({"error": "No records found"}), 404
        
        result = []
        for record in records:
            result.append({
                "id": record.id,
                "name": record.name,
                "description": record.description,
                "length": len(record.seq),
                "seq_preview": str(record.seq[:100]) + ("..." if len(record.seq) > 100 else ""),
                "features": len(record.features) if hasattr(record, 'features') else 0
            })
        
        return jsonify({"success": True, "records": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting ViroSeek API server...")
    print("Endpoints:")
    print("  GET  /api/health   - Health check")
    print("  POST /api/analyze  - Run analysis")
    print("  POST /api/fetch-sequence - Fetch sequence info")
    print()
    app.run(host="0.0.0.0", port=5001, debug=True)
