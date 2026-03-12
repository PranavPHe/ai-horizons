import argparse
import csv
import re
import sys
import urllib.request
from pathlib import Path

from Bio import Entrez, SeqIO

from immunogenicity import (
    ImmunogenicityAnalyzer,
    VaccineCandidate,
    generate_report
)
from visualization import generate_dashboard, generate_mrna_variants

# Set your email for NCBI Entrez (required)
Entrez.email = "your_email@example.com"

# Pre-compile regex patterns
RE_ACCESSION = re.compile(r"^[A-Z]{1,2}_?\d+(\.\d+)?$", re.IGNORECASE)
RE_PDB_ID = re.compile(r"^[0-9][A-Z0-9]{3}$", re.IGNORECASE)


# Check if input looks like an NCBI accession ID.
def is_accession_id(input_str: str) -> bool:
    return bool(RE_ACCESSION.match(input_str.strip()))


# Check if input looks like a PDB ID (4 characters, e.g., 1ABC).
def is_pdb_id(input_str: str) -> bool:
    return bool(RE_PDB_ID.match(input_str.strip()))


# Fetch sequence from NCBI by accession ID (generator).
def fetch_from_ncbi(accession_id: str, db: str = "nucleotide"):
    print(f"Fetching {accession_id} from NCBI...")
    with Entrez.efetch(db=db, id=accession_id, rettype="gb", retmode="text") as handle:
        yield from SeqIO.parse(handle, "genbank")


# Detects the file format based on the file extension
def detect_format(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    if ext in (".fasta", ".fa", ".fna", ".faa"):
        return "fasta"
    elif ext in (".gb", ".gbk", ".genbank"):
        return "genbank"
    raise ValueError(f"Unknown file format: {ext}")


# Load sequences from file or NCBI accession ID (generator).
def load_sequences(input_source: str, format: str = None):
    if is_accession_id(input_source):
        yield from fetch_from_ncbi(input_source)
    else:
        file_path = Path(input_source)
        if not file_path.exists():
            raise FileNotFoundError(f"File '{input_source}' not found")
            
        fmt = format or detect_format(input_source)
        yield from SeqIO.parse(file_path, fmt)


# Fetch PDB file from RCSB by PDB ID.
def fetch_pdb(pdb_id: str, output_dir: Path = None):
    output_dir = output_dir or Path(".")
    pdb_path = output_dir / f"{pdb_id.upper()}.pdb"
    
    # Don't re-download if it already exists
    if pdb_path.exists():
        print(f"PDB {pdb_path} already exists, skipping download.")
        return pdb_path

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    print(f"Fetching {pdb_id.upper()} from RCSB PDB...")
    urllib.request.urlretrieve(url, pdb_path)
    return pdb_path


# Load PDB structure from file or PDB ID.
def load_structure(pdb_input: str):
    # Lazy import to speed up startup if PDB is not used
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    
    if is_pdb_id(pdb_input):
        pdb_path = fetch_pdb(pdb_input)
        structure = parser.get_structure(pdb_input.upper(), str(pdb_path))
    else:
        pdb_path = Path(pdb_input)
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB file '{pdb_input}' not found")
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    
    return structure


# Load gene expression data from CSV or count matrix file.
def load_expression_data(filepath: str):
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Expression file '{filepath}' not found")
    
    ext = path.suffix.lower()
    if ext not in (".csv", ".tsv", ".txt"):
        raise ValueError(f"Unsupported expression file format: {ext}")
    
    delimiter = "\t" if ext in (".tsv", ".txt") else ","
    
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, None)
        rows = list(reader)
    
    return {"header": header, "rows": rows, "path": str(path)}


def main():
    parser = argparse.ArgumentParser(
        description="Parse viral genome sequences and protein structures"
    )
    parser.add_argument(
        "input",
        help="Path to FASTA/GenBank file or NCBI accession ID (e.g., NC_001802)"
    )
    parser.add_argument(
        "-p", "--pdb",
        help="Path to PDB file or PDB ID (e.g., 1ABC)"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["fasta", "genbank"],
        help="File format (auto-detected if not specified, ignored for accession IDs)"
    )
    parser.add_argument(
        "-e", "--email",
        help="Email for NCBI Entrez (required for fetching accession IDs)"
    )
    parser.add_argument(
        "-n", "--virus-name",
        help="Name of the virus (e.g., 'SARS-CoV-2')"
    )
    parser.add_argument(
        "-v", "--variant",
        choices=["alpha", "beta", "gamma", "delta", "omicron", "wild-type", "other"],
        help="Variant type classification"
    )
    parser.add_argument(
        "-t", "--host",
        help="Host species NCBI taxonomy ID (e.g., 9606 for human)"
    )
    parser.add_argument(
        "-g", "--expression",
        help="Path to gene expression CSV or count matrix file"
    )
    parser.add_argument(
        "-a", "--analyze",
        action="store_true",
        help="Run immunogenicity analysis on sequences"
    )
    parser.add_argument(
        "--iedb",
        action="store_true",
        default=True,
        help="Query IEDB for known epitopes (default: True)"
    )
    parser.add_argument(
        "--no-iedb",
        action="store_true",
        help="Disable IEDB queries (offline mode)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path for analysis report"
    )
    parser.add_argument(
        "--report-format",
        choices=["text", "csv", "json"],
        default="text",
        help="Report output format (default: text)"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top candidates to show (default: 10)"
    )
    parser.add_argument(
        "--weight-immuno",
        type=float,
        default=0.5,
        help="Weight for immunogenicity score (default: 0.5)"
    )
    parser.add_argument(
        "--weight-stability",
        type=float,
        default=0.3,
        help="Weight for mutation stability score (default: 0.3)"
    )
    parser.add_argument(
        "--weight-safety",
        type=float,
        default=0.2,
        help="Weight for safety score (default: 0.2)"
    )
    parser.add_argument(
        "-d", "--dashboard",
        help="Generate interactive HTML dashboard at specified path"
    )
    parser.add_argument(
        "--mrna-variants",
        action="store_true",
        help="Generate mRNA variants for multiple expression systems"
    )
    parser.add_argument(
        "--mrna-output",
        help="Output path for mRNA sequences (FASTA format)"
    )
    args = parser.parse_args()

    if args.email:
        Entrez.email = args.email

    try:
        # Display metadata if provided
        if args.virus_name or args.variant or args.host:
            print("--- Metadata ---")
            if args.virus_name:
                print(f"Virus: {args.virus_name}")
            if args.variant:
                print(f"Variant: {args.variant}")
            if args.host:
                print(f"Host Taxonomy ID: {args.host}")
            print()

        # Load sequences
        count = 0
        for record in load_sequences(args.input, args.format):
            count += 1
            print(f"ID: {record.id}")
            print(f"Description: {record.description}")
            print(f"Length: {len(record.seq)} bp")
            print(f"Sequence: {record.seq[:50]}{'...' if len(record.seq) > 50 else ''}")
            print()
        print(f"Processed {count} sequence(s)")

        # Load PDB structure if provided
        if args.pdb:
            print("\n--- Protein Structure ---")
            structure = load_structure(args.pdb)
            print(f"Structure ID: {structure.id}")
            print(f"Models: {len(list(structure.get_models()))}")
            print(f"Chains: {[chain.id for chain in structure.get_chains()]}")
            print(f"Residues: {len(list(structure.get_residues()))}")
            print(f"Atoms: {len(list(structure.get_atoms()))}")

        # Load gene expression data if provided
        if args.expression:
            print("\n--- Gene Expression Data ---")
            expr_data = load_expression_data(args.expression)
            print(f"File: {expr_data['path']}")
            print(f"Columns: {len(expr_data['header']) if expr_data['header'] else 0}")
            print(f"Rows: {len(expr_data['rows'])}")
            if expr_data['header']:
                print(f"Header: {', '.join(expr_data['header'][:5])}{'...' if len(expr_data['header']) > 5 else ''}")

        # Run immunogenicity analysis if requested
        if args.analyze:
            print("\n--- Immunogenicity Analysis ---")
            print("Analyzing sequences for vaccine targets...")
            
            use_iedb = not args.no_iedb
            analyzer = ImmunogenicityAnalyzer(
                use_iedb=use_iedb,
                organism=args.virus_name,
                host="human"
            )
            
            candidates = []
            
            # Re-load sequences for analysis
            for record in load_sequences(args.input, args.format):
                # Extract protein sequences from CDS features
                proteins_found = False
                
                if hasattr(record, 'features'):
                    for feature in record.features:
                        if feature.type == "CDS":
                            gene_name = feature.qualifiers.get('gene', ['unknown'])[0]
                            protein_name = feature.qualifiers.get('product', ['unknown'])[0]
                            
                            # Get protein sequence
                            if 'translation' in feature.qualifiers:
                                protein_seq = feature.qualifiers['translation'][0]
                            else:
                                # Translate from nucleotide
                                try:
                                    protein_seq = str(feature.extract(record.seq).translate())
                                except Exception:
                                    continue
                            
                            if len(protein_seq) < 10:
                                continue
                            
                            print(f"  Analyzing: {gene_name} ({protein_name[:30]}...)")
                            
                            candidate = analyzer.analyze_sequence(
                                sequence=protein_seq,
                                gene_name=gene_name,
                                protein_name=protein_name,
                                compare_iedb=use_iedb,
                                predict_epitopes=True
                            )
                            candidates.append(candidate)
                            proteins_found = True
                
                # If no CDS features, try to translate the whole sequence
                if not proteins_found and len(record.seq) >= 30:
                    print(f"  Analyzing: {record.id} (full sequence)")
                    try:
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
                        print(f"  Warning: Could not translate {record.id}: {e}")
            
            if candidates:
                # Custom weights for ranking
                weights = {
                    'immunogenicity': args.weight_immuno,
                    'stability': args.weight_stability,
                    'safety': args.weight_safety
                }
                
                # Recalculate scores with custom weights
                for c in candidates:
                    c.calculate_overall_score(weights)
                
                # Rank candidates
                ranked = analyzer.rank_candidates(candidates, top_n=args.top)
                
                print(f"\nAnalysis complete! Found {len(candidates)} protein targets.")
                print(f"\nRanking weights: Immunogenicity={weights['immunogenicity']:.0%}, "
                      f"Stability={weights['stability']:.0%}, Safety={weights['safety']:.0%}")
                print(f"\nTop {min(args.top, len(ranked))} vaccine candidates:")
                print("-" * 60)
                
                for i, c in enumerate(ranked, 1):
                    grade = c.get_rank_grade()
                    print(f"\n  #{i} [{grade}] {c.gene_name} - {c.protein_name[:35]}")
                    print(f"      Overall: {c.overall_score:.3f}")
                    print(f"      ├─ Immunogenicity:     {c.immunogenicity_score:.3f} "
                          f"({len([e for e in c.epitopes if e.epitope_type == 'B-cell'])} B-cell, "
                          f"{len([e for e in c.epitopes if e.epitope_type == 'T-cell'])} T-cell epitopes)")
                    print(f"      ├─ Mutation Stability: {c.mutation_stability:.3f}")
                    print(f"      └─ Safety:             {c.safety_score:.3f}"
                          + (f" ⚠ {len(c.safety_warnings)} warning(s)" if c.safety_warnings else " ✓ No concerns"))
                    print(f"      → {c.get_recommendation()}")
                
                print("\n" + "-" * 60)
                
                # Generate report if output specified
                if args.output:
                    report = generate_report(
                        ranked,
                        output_path=args.output,
                        format=args.report_format
                    )
                    print(f"Full report saved to: {args.output}")
                
                # Generate interactive dashboard
                if args.dashboard:
                    dashboard_title = f"ViroSeek Analysis: {args.virus_name or 'Viral Genome'}"
                    generate_dashboard(
                        ranked,
                        output_path=args.dashboard,
                        title=dashboard_title
                    )
                    print(f"Interactive dashboard saved to: {args.dashboard}")
                
                # Generate mRNA variants
                if args.mrna_variants or args.mrna_output:
                    print("\n--- mRNA Sequence Variants ---")
                    mrna_fasta = []
                    
                    for c in ranked[:args.top]:
                        print(f"\n{c.gene_name}:")
                        variants = generate_mrna_variants(c.sequence)
                        
                        for system, data in variants.items():
                            print(f"  {system.upper():6} | {data['length']:5} nt | GC: {data['gc_content']:5.1f}% | {data['description']}")
                            mrna_fasta.append(f">{c.gene_name}_{system}_mRNA | {c.protein_name} | {data['description']}")
                            # Wrap sequence at 70 characters
                            seq = data['sequence']
                            for j in range(0, len(seq), 70):
                                mrna_fasta.append(seq[j:j+70])
                    
                    # Save mRNA FASTA if output specified
                    if args.mrna_output:
                        with open(args.mrna_output, 'w') as f:
                            f.write('\n'.join(mrna_fasta))
                        print(f"\nmRNA sequences saved to: {args.mrna_output}")
            else:
                print("No protein sequences found for analysis.")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()