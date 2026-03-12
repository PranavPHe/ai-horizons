"""
Visualization Dashboard for ViroSeek

Generates interactive HTML dashboards with:
- Protein epitope maps
- Ranking charts
- mRNA sequence viewer
- Exportable reports
"""

import json
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path

# Import from immunogenicity module
try:
    from immunogenicity import VaccineCandidate, Epitope
except ImportError:
    pass


def generate_dashboard(
    candidates: List['VaccineCandidate'],
    output_path: str,
    title: str = "ViroSeek Analysis Results"
) -> str:
    """
    Generate an interactive HTML dashboard for vaccine candidates.
    
    Args:
        candidates: Ranked list of vaccine candidates
        output_path: Path to save the HTML file
        title: Dashboard title
    
    Returns:
        Path to the generated HTML file
    """
    html = _generate_html_dashboard(candidates, title)
    
    Path(output_path).write_text(html)
    print(f"Dashboard saved to: {output_path}")
    
    return output_path


def _generate_html_dashboard(
    candidates: List['VaccineCandidate'],
    title: str
) -> str:
    """Generate the complete HTML dashboard."""
    
    # Generate candidate cards
    candidate_cards = ""
    for i, c in enumerate(candidates, 1):
        candidate_cards += _generate_candidate_card(c, i)
    
    # Generate ranking chart data
    chart_data = json.dumps([{
        'gene': c.gene_name[:15],
        'overall': round(c.overall_score, 3),
        'immunogenicity': round(c.immunogenicity_score, 3),
        'stability': round(c.mutation_stability, 3),
        'safety': round(c.safety_score, 3)
    } for c in candidates])
    
    # Generate comparison table
    comparison_table = _generate_comparison_table(candidates)
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            padding: 30px 0;
            border-bottom: 2px solid #0f3460;
            margin-bottom: 30px;
        }}
        
        h1 {{
            font-size: 2.5rem;
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        
        .subtitle {{
            color: #888;
            font-size: 1.1rem;
        }}
        
        .stats-bar {{
            display: flex;
            justify-content: center;
            gap: 40px;
            margin: 20px 0;
        }}
        
        .stat {{
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #00d4ff;
        }}
        
        .stat-label {{
            font-size: 0.85rem;
            color: #888;
        }}
        
        .section {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 25px;
            backdrop-filter: blur(10px);
        }}
        
        .section-title {{
            font-size: 1.4rem;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .section-title::before {{
            content: '';
            width: 4px;
            height: 24px;
            background: linear-gradient(180deg, #00d4ff, #7c3aed);
            border-radius: 2px;
        }}
        
        /* Charts */
        .charts-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px;
        }}
        
        .chart-container {{
            background: rgba(0, 0, 0, 0.3);
            border-radius: 12px;
            padding: 20px;
            height: 350px;
        }}
        
        /* Candidate Cards */
        .candidates-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
            gap: 20px;
        }}
        
        .candidate-card {{
            background: rgba(0, 0, 0, 0.3);
            border-radius: 12px;
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .candidate-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 40px rgba(0, 212, 255, 0.2);
        }}
        
        .card-header {{
            padding: 20px;
            background: linear-gradient(90deg, rgba(0, 212, 255, 0.1), rgba(124, 58, 237, 0.1));
            border-bottom: 1px solid #333;
        }}
        
        .card-rank {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, #00d4ff, #7c3aed);
            border-radius: 50%;
            font-weight: bold;
            margin-right: 12px;
        }}
        
        .card-grade {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: bold;
            margin-left: 10px;
        }}
        
        .grade-a {{ background: #10b981; }}
        .grade-b {{ background: #3b82f6; }}
        .grade-c {{ background: #f59e0b; }}
        .grade-d {{ background: #ef4444; }}
        .grade-f {{ background: #6b7280; }}
        
        .card-body {{
            padding: 20px;
        }}
        
        .score-bars {{
            margin: 15px 0;
        }}
        
        .score-row {{
            display: flex;
            align-items: center;
            margin-bottom: 12px;
        }}
        
        .score-label {{
            width: 140px;
            font-size: 0.9rem;
            color: #aaa;
        }}
        
        .score-bar {{
            flex: 1;
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .score-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }}
        
        .score-fill.immuno {{ background: linear-gradient(90deg, #10b981, #34d399); }}
        .score-fill.stability {{ background: linear-gradient(90deg, #3b82f6, #60a5fa); }}
        .score-fill.safety {{ background: linear-gradient(90deg, #8b5cf6, #a78bfa); }}
        .score-fill.overall {{ background: linear-gradient(90deg, #00d4ff, #7c3aed); }}
        
        .score-value {{
            width: 60px;
            text-align: right;
            font-weight: bold;
            font-size: 0.9rem;
        }}
        
        .recommendation {{
            margin-top: 15px;
            padding: 12px;
            background: rgba(16, 185, 129, 0.1);
            border-left: 3px solid #10b981;
            border-radius: 0 8px 8px 0;
            font-size: 0.9rem;
        }}
        
        .recommendation.caution {{
            background: rgba(239, 68, 68, 0.1);
            border-left-color: #ef4444;
        }}
        
        /* Protein Map */
        .protein-map {{
            position: relative;
            height: 80px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 8px;
            margin: 15px 0;
            overflow: hidden;
        }}
        
        .protein-backbone {{
            position: absolute;
            top: 50%;
            left: 10px;
            right: 10px;
            height: 8px;
            background: #444;
            transform: translateY(-50%);
            border-radius: 4px;
        }}
        
        .epitope-marker {{
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            height: 20px;
            border-radius: 4px;
            cursor: pointer;
            transition: height 0.2s;
        }}
        
        .epitope-marker:hover {{
            height: 30px;
        }}
        
        .epitope-marker.bcell {{ background: rgba(16, 185, 129, 0.7); }}
        .epitope-marker.tcell {{ background: rgba(239, 68, 68, 0.7); }}
        .epitope-marker.iedb {{ background: rgba(245, 158, 11, 0.7); }}
        
        .map-legend {{
            display: flex;
            gap: 20px;
            justify-content: center;
            font-size: 0.8rem;
            color: #888;
            margin-top: 10px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 3px;
        }}
        
        /* mRNA Section */
        .mrna-container {{
            background: rgba(0, 0, 0, 0.4);
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
        }}
        
        .mrna-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        
        .mrna-sequence {{
            font-family: 'Courier New', monospace;
            font-size: 0.75rem;
            word-break: break-all;
            line-height: 1.6;
            color: #00d4ff;
            max-height: 100px;
            overflow-y: auto;
        }}
        
        .mrna-stats {{
            display: flex;
            gap: 20px;
            margin-top: 10px;
            font-size: 0.85rem;
            color: #888;
        }}
        
        .copy-btn {{
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            border: none;
            color: white;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85rem;
            transition: opacity 0.2s;
        }}
        
        .copy-btn:hover {{
            opacity: 0.8;
        }}
        
        /* Comparison Table */
        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        
        .comparison-table th,
        .comparison-table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        
        .comparison-table th {{
            background: rgba(0, 212, 255, 0.1);
            font-weight: 600;
            color: #00d4ff;
        }}
        
        .comparison-table tr:hover {{
            background: rgba(255, 255, 255, 0.05);
        }}
        
        /* Export Section */
        .export-buttons {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }}
        
        .export-btn {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 12px 24px;
            border: 2px solid #333;
            border-radius: 8px;
            background: transparent;
            color: #eee;
            font-size: 0.95rem;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .export-btn:hover {{
            border-color: #00d4ff;
            background: rgba(0, 212, 255, 0.1);
        }}
        
        .export-btn svg {{
            width: 20px;
            height: 20px;
        }}
        
        /* Tabs */
        .tabs {{
            display: flex;
            gap: 5px;
            margin-bottom: 20px;
        }}
        
        .tab {{
            padding: 10px 20px;
            background: rgba(255, 255, 255, 0.05);
            border: none;
            border-radius: 8px 8px 0 0;
            color: #888;
            cursor: pointer;
            transition: all 0.2s;
        }}
        
        .tab.active {{
            background: rgba(0, 212, 255, 0.2);
            color: #00d4ff;
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        /* Tooltips */
        .tooltip {{
            position: absolute;
            background: #1a1a2e;
            border: 1px solid #333;
            padding: 10px;
            border-radius: 6px;
            font-size: 0.8rem;
            z-index: 1000;
            pointer-events: none;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        }}
        
        footer {{
            text-align: center;
            padding: 30px;
            color: #666;
            font-size: 0.85rem;
        }}
        
        @media (max-width: 768px) {{
            .candidates-grid {{
                grid-template-columns: 1fr;
            }}
            
            .charts-grid {{
                grid-template-columns: 1fr;
            }}
            
            .stats-bar {{
                flex-wrap: wrap;
                gap: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧬 ViroSeek Analysis Dashboard</h1>
            <p class="subtitle">{title}</p>
            <div class="stats-bar">
                <div class="stat">
                    <div class="stat-value">{len(candidates)}</div>
                    <div class="stat-label">Candidates Analyzed</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{sum(len(c.epitopes) for c in candidates)}</div>
                    <div class="stat-label">Epitopes Found</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{len([c for c in candidates if c.overall_score >= 0.8])}</div>
                    <div class="stat-label">High Confidence</div>
                </div>
            </div>
        </header>
        
        <!-- Charts Section -->
        <div class="section">
            <h2 class="section-title">Ranking Overview</h2>
            <div class="charts-grid">
                <div class="chart-container">
                    <canvas id="rankingChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="radarChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Comparison Table -->
        <div class="section">
            <h2 class="section-title">Comparison Table</h2>
            {comparison_table}
        </div>
        
        <!-- Candidate Cards -->
        <div class="section">
            <h2 class="section-title">Detailed Analysis</h2>
            <div class="tabs">
                <button class="tab active" onclick="showTab('cards')">Card View</button>
                <button class="tab" onclick="showTab('list')">List View</button>
            </div>
            <div id="cards" class="tab-content active">
                <div class="candidates-grid">
                    {candidate_cards}
                </div>
            </div>
        </div>
        
        <!-- Export Section -->
        <div class="section">
            <h2 class="section-title">Export Results</h2>
            <div class="export-buttons">
                <button class="export-btn" onclick="exportJSON()">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                        <line x1="16" y1="13" x2="8" y2="13"/>
                        <line x1="16" y1="17" x2="8" y2="17"/>
                    </svg>
                    Export JSON
                </button>
                <button class="export-btn" onclick="exportCSV()">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                        <line x1="16" y1="13" x2="8" y2="13"/>
                        <line x1="16" y1="17" x2="8" y2="17"/>
                    </svg>
                    Export CSV
                </button>
                <button class="export-btn" onclick="exportFASTA()">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                        <polyline points="14 2 14 8 20 8"/>
                    </svg>
                    Export mRNA (FASTA)
                </button>
                <button class="export-btn" onclick="window.print()">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="6 9 6 2 18 2 18 9"/>
                        <path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"/>
                        <rect x="6" y="14" width="12" height="8"/>
                    </svg>
                    Print Report
                </button>
            </div>
        </div>
        
        <footer>
            Generated by ViroSeek | Immunogenic Target Analysis Tool
        </footer>
    </div>
    
    <script>
        // Chart data
        const chartData = {chart_data};
        
        // Ranking Bar Chart
        const rankingCtx = document.getElementById('rankingChart').getContext('2d');
        new Chart(rankingCtx, {{
            type: 'bar',
            data: {{
                labels: chartData.map(d => d.gene),
                datasets: [
                    {{
                        label: 'Overall',
                        data: chartData.map(d => d.overall),
                        backgroundColor: 'rgba(0, 212, 255, 0.8)',
                        borderRadius: 4
                    }},
                    {{
                        label: 'Immunogenicity',
                        data: chartData.map(d => d.immunogenicity),
                        backgroundColor: 'rgba(16, 185, 129, 0.8)',
                        borderRadius: 4
                    }},
                    {{
                        label: 'Stability',
                        data: chartData.map(d => d.stability),
                        backgroundColor: 'rgba(59, 130, 246, 0.8)',
                        borderRadius: 4
                    }},
                    {{
                        label: 'Safety',
                        data: chartData.map(d => d.safety),
                        backgroundColor: 'rgba(139, 92, 246, 0.8)',
                        borderRadius: 4
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Candidate Score Comparison',
                        color: '#eee',
                        font: {{ size: 16 }}
                    }},
                    legend: {{
                        labels: {{ color: '#aaa' }}
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1,
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        ticks: {{ color: '#aaa' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#aaa' }}
                    }}
                }}
            }}
        }});
        
        // Radar Chart for top candidate
        const radarCtx = document.getElementById('radarChart').getContext('2d');
        new Chart(radarCtx, {{
            type: 'radar',
            data: {{
                labels: ['Immunogenicity', 'Stability', 'Safety', 'Epitope Density', 'B-cell', 'T-cell'],
                datasets: chartData.slice(0, 3).map((d, i) => ({{
                    label: d.gene,
                    data: [
                        d.immunogenicity,
                        d.stability,
                        d.safety,
                        Math.min(d.immunogenicity * 0.8, 1),
                        Math.min(d.immunogenicity * 0.9, 1),
                        Math.min(d.immunogenicity * 0.7, 1)
                    ],
                    borderColor: ['rgba(0, 212, 255, 1)', 'rgba(16, 185, 129, 1)', 'rgba(139, 92, 246, 1)'][i],
                    backgroundColor: ['rgba(0, 212, 255, 0.1)', 'rgba(16, 185, 129, 0.1)', 'rgba(139, 92, 246, 0.1)'][i],
                    borderWidth: 2,
                    pointRadius: 4
                }}))
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Top 3 Candidates - Multi-factor Analysis',
                        color: '#eee',
                        font: {{ size: 16 }}
                    }},
                    legend: {{
                        labels: {{ color: '#aaa' }}
                    }}
                }},
                scales: {{
                    r: {{
                        beginAtZero: true,
                        max: 1,
                        grid: {{ color: 'rgba(255,255,255,0.1)' }},
                        angleLines: {{ color: 'rgba(255,255,255,0.1)' }},
                        pointLabels: {{ color: '#aaa' }},
                        ticks: {{ display: false }}
                    }}
                }}
            }}
        }});
        
        // Tab switching
        function showTab(tabId) {{
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }}
        
        // Copy mRNA sequence
        function copyMRNA(id) {{
            const el = document.getElementById(id);
            navigator.clipboard.writeText(el.textContent);
            const btn = event.target;
            btn.textContent = 'Copied!';
            setTimeout(() => btn.textContent = 'Copy', 2000);
        }}
        
        // Export functions
        const fullData = {_generate_json_data(candidates)};
        
        function exportJSON() {{
            const blob = new Blob([JSON.stringify(fullData, null, 2)], {{type: 'application/json'}});
            downloadBlob(blob, 'viroseek_results.json');
        }}
        
        function exportCSV() {{
            let csv = 'Rank,Gene,Protein,Length,Overall,Immunogenicity,Stability,Safety,B-cell,T-cell,Recommendation\\n';
            fullData.candidates.forEach(c => {{
                csv += `${{c.rank}},${{c.gene_name}},"${{c.protein_name}}",${{c.length}},${{c.scores.overall}},${{c.scores.immunogenicity}},${{c.scores.mutation_stability}},${{c.scores.safety}},${{c.epitope_summary.bcell}},${{c.epitope_summary.tcell}},"${{c.recommendation}}"\\n`;
            }});
            const blob = new Blob([csv], {{type: 'text/csv'}});
            downloadBlob(blob, 'viroseek_results.csv');
        }}
        
        function exportFASTA() {{
            let fasta = '';
            fullData.candidates.forEach(c => {{
                if (c.suggested_mrna) {{
                    fasta += `>${{c.gene_name}}_mRNA | ${{c.protein_name}} | Score: ${{c.scores.overall}}\\n`;
                    // Wrap at 70 characters
                    for (let i = 0; i < c.suggested_mrna.length; i += 70) {{
                        fasta += c.suggested_mrna.slice(i, i + 70) + '\\n';
                    }}
                }}
            }});
            const blob = new Blob([fasta], {{type: 'text/plain'}});
            downloadBlob(blob, 'viroseek_mrna.fasta');
        }}
        
        function downloadBlob(blob, filename) {{
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }}
    </script>
</body>
</html>'''
    
    return html


def _generate_candidate_card(candidate: 'VaccineCandidate', rank: int) -> str:
    """Generate HTML card for a single candidate."""
    
    # Determine grade class
    grade = candidate.get_rank_grade()
    grade_class = 'grade-a' if grade in ['A+', 'A'] else \
                  'grade-b' if grade == 'B' else \
                  'grade-c' if grade == 'C' else \
                  'grade-d' if grade == 'D' else 'grade-f'
    
    # Recommendation class
    rec_class = 'caution' if 'CAUTION' in candidate.get_recommendation() or 'NOT' in candidate.get_recommendation() else ''
    
    # Epitope counts
    bcell = len([e for e in candidate.epitopes if e.epitope_type == 'B-cell'])
    tcell = len([e for e in candidate.epitopes if e.epitope_type == 'T-cell'])
    iedb = len([e for e in candidate.epitopes if e.source == 'iedb'])
    
    # Generate protein map markers
    protein_map = _generate_protein_map(candidate)
    
    # mRNA preview
    mrna_preview = candidate.suggested_mrna[:150] + '...' if candidate.suggested_mrna and len(candidate.suggested_mrna) > 150 else (candidate.suggested_mrna or '')
    mrna_id = f"mrna_{rank}"
    
    # GC content calculation
    gc_content = 0
    if candidate.suggested_mrna:
        gc = candidate.suggested_mrna.count('G') + candidate.suggested_mrna.count('C')
        gc_content = round(gc / len(candidate.suggested_mrna) * 100, 1)
    
    return f'''
    <div class="candidate-card">
        <div class="card-header">
            <span class="card-rank">{rank}</span>
            <strong>{candidate.gene_name}</strong> - {candidate.protein_name[:35]}{'...' if len(candidate.protein_name) > 35 else ''}
            <span class="card-grade {grade_class}">{grade}</span>
        </div>
        <div class="card-body">
            <div class="score-bars">
                <div class="score-row">
                    <span class="score-label">Overall Score</span>
                    <div class="score-bar">
                        <div class="score-fill overall" style="width: {candidate.overall_score * 100}%"></div>
                    </div>
                    <span class="score-value">{candidate.overall_score:.3f}</span>
                </div>
                <div class="score-row">
                    <span class="score-label">Immunogenicity</span>
                    <div class="score-bar">
                        <div class="score-fill immuno" style="width: {candidate.immunogenicity_score * 100}%"></div>
                    </div>
                    <span class="score-value">{candidate.immunogenicity_score:.3f}</span>
                </div>
                <div class="score-row">
                    <span class="score-label">Mut. Stability</span>
                    <div class="score-bar">
                        <div class="score-fill stability" style="width: {candidate.mutation_stability * 100}%"></div>
                    </div>
                    <span class="score-value">{candidate.mutation_stability:.3f}</span>
                </div>
                <div class="score-row">
                    <span class="score-label">Safety</span>
                    <div class="score-bar">
                        <div class="score-fill safety" style="width: {candidate.safety_score * 100}%"></div>
                    </div>
                    <span class="score-value">{candidate.safety_score:.3f}</span>
                </div>
            </div>
            
            <div class="recommendation {rec_class}">
                {candidate.get_recommendation()}
            </div>
            
            <h4 style="margin: 15px 0 10px 0; font-size: 0.95rem; color: #00d4ff;">
                Epitope Map ({len(candidate.epitopes)} epitopes)
            </h4>
            {protein_map}
            <div class="map-legend">
                <div class="legend-item">
                    <div class="legend-dot" style="background: rgba(16, 185, 129, 0.7);"></div>
                    B-cell ({bcell})
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background: rgba(239, 68, 68, 0.7);"></div>
                    T-cell ({tcell})
                </div>
                <div class="legend-item">
                    <div class="legend-dot" style="background: rgba(245, 158, 11, 0.7);"></div>
                    IEDB ({iedb})
                </div>
            </div>
            
            <div class="mrna-container">
                <div class="mrna-header">
                    <strong style="font-size: 0.9rem;">Suggested mRNA Sequence</strong>
                    <button class="copy-btn" onclick="copyMRNA('{mrna_id}')">Copy</button>
                </div>
                <div class="mrna-sequence" id="{mrna_id}">{candidate.suggested_mrna or 'N/A'}</div>
                <div class="mrna-stats">
                    <span>Length: {len(candidate.suggested_mrna) if candidate.suggested_mrna else 0} nt</span>
                    <span>GC Content: {gc_content}%</span>
                    <span>Protein: {len(candidate.sequence)} aa</span>
                </div>
            </div>
        </div>
    </div>
    '''


def _generate_protein_map(candidate: 'VaccineCandidate') -> str:
    """Generate SVG protein map with epitope markers."""
    
    if not candidate.epitopes or not candidate.sequence:
        return '<div class="protein-map"><div class="protein-backbone"></div></div>'
    
    seq_len = len(candidate.sequence)
    markers = ""
    
    for epitope in candidate.epitopes:
        # Calculate position as percentage
        start_pct = (epitope.start / seq_len) * 100
        width_pct = max(((epitope.end - epitope.start) / seq_len) * 100, 1)
        
        # Determine class based on type
        if epitope.source == 'iedb':
            cls = 'iedb'
        elif epitope.epitope_type == 'T-cell':
            cls = 'tcell'
        else:
            cls = 'bcell'
        
        markers += f'<div class="epitope-marker {cls}" style="left: calc(10px + {start_pct}% * 0.96); width: {width_pct}%;" title="{epitope.sequence[:20]}... ({epitope.epitope_type})"></div>'
    
    return f'''
    <div class="protein-map">
        <div class="protein-backbone"></div>
        {markers}
    </div>
    '''


def _generate_comparison_table(candidates: List['VaccineCandidate']) -> str:
    """Generate HTML comparison table."""
    
    rows = ""
    for i, c in enumerate(candidates, 1):
        bcell = len([e for e in c.epitopes if e.epitope_type == 'B-cell'])
        tcell = len([e for e in c.epitopes if e.epitope_type == 'T-cell'])
        
        rows += f'''
        <tr>
            <td>{i}</td>
            <td><strong>{c.gene_name}</strong></td>
            <td>{c.protein_name[:30]}{'...' if len(c.protein_name) > 30 else ''}</td>
            <td>{len(c.sequence)} aa</td>
            <td><strong>{c.overall_score:.3f}</strong></td>
            <td>{c.immunogenicity_score:.3f}</td>
            <td>{c.mutation_stability:.3f}</td>
            <td>{c.safety_score:.3f}</td>
            <td>{bcell}</td>
            <td>{tcell}</td>
            <td>{c.get_rank_grade()}</td>
        </tr>
        '''
    
    return f'''
    <table class="comparison-table">
        <thead>
            <tr>
                <th>Rank</th>
                <th>Gene</th>
                <th>Protein</th>
                <th>Length</th>
                <th>Overall</th>
                <th>Immuno.</th>
                <th>Stability</th>
                <th>Safety</th>
                <th>B-cell</th>
                <th>T-cell</th>
                <th>Grade</th>
            </tr>
        </thead>
        <tbody>
            {rows}
        </tbody>
    </table>
    '''


def _generate_json_data(candidates: List['VaccineCandidate']) -> str:
    """Generate JSON data for export functions."""
    
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
                'overall': round(c.overall_score, 4),
                'immunogenicity': round(c.immunogenicity_score, 4),
                'mutation_stability': round(c.mutation_stability, 4),
                'safety': round(c.safety_score, 4)
            },
            'epitope_summary': {
                'total': len(c.epitopes),
                'bcell': len([e for e in c.epitopes if e.epitope_type == 'B-cell']),
                'tcell': len([e for e in c.epitopes if e.epitope_type == 'T-cell']),
                'iedb_matches': len([e for e in c.epitopes if e.source == 'iedb'])
            },
            'suggested_mrna': c.suggested_mrna,
            'mrna_length': len(c.suggested_mrna) if c.suggested_mrna else 0
        })
    
    return json.dumps(data)


def generate_mrna_variants(
    protein_sequence: str,
    optimization: str = "human"
) -> Dict[str, str]:
    """
    Generate multiple mRNA sequence variants with different optimizations.
    
    Args:
        protein_sequence: Input protein sequence
        optimization: Target optimization ('human', 'ecoli', 'yeast', 'cho')
    
    Returns:
        Dictionary with different mRNA variants
    """
    
    # Codon tables for different expression systems
    CODON_TABLES = {
        'human': {
            'A': 'GCC', 'R': 'CGG', 'N': 'AAC', 'D': 'GAC', 'C': 'TGC',
            'Q': 'CAG', 'E': 'GAG', 'G': 'GGC', 'H': 'CAC', 'I': 'ATC',
            'L': 'CTG', 'K': 'AAG', 'M': 'ATG', 'F': 'TTC', 'P': 'CCC',
            'S': 'AGC', 'T': 'ACC', 'W': 'TGG', 'Y': 'TAC', 'V': 'GTG', '*': 'TGA'
        },
        'ecoli': {
            'A': 'GCG', 'R': 'CGT', 'N': 'AAC', 'D': 'GAT', 'C': 'TGC',
            'Q': 'CAG', 'E': 'GAA', 'G': 'GGT', 'H': 'CAT', 'I': 'ATT',
            'L': 'CTG', 'K': 'AAA', 'M': 'ATG', 'F': 'TTT', 'P': 'CCG',
            'S': 'TCT', 'T': 'ACC', 'W': 'TGG', 'Y': 'TAT', 'V': 'GTT', '*': 'TAA'
        },
        'yeast': {
            'A': 'GCT', 'R': 'AGA', 'N': 'AAC', 'D': 'GAT', 'C': 'TGT',
            'Q': 'CAA', 'E': 'GAA', 'G': 'GGT', 'H': 'CAC', 'I': 'ATT',
            'L': 'TTG', 'K': 'AAG', 'M': 'ATG', 'F': 'TTC', 'P': 'CCA',
            'S': 'TCT', 'T': 'ACT', 'W': 'TGG', 'Y': 'TAC', 'V': 'GTT', '*': 'TAA'
        },
        'cho': {  # Chinese Hamster Ovary
            'A': 'GCC', 'R': 'CGC', 'N': 'AAC', 'D': 'GAC', 'C': 'TGC',
            'Q': 'CAG', 'E': 'GAG', 'G': 'GGC', 'H': 'CAC', 'I': 'ATC',
            'L': 'CTG', 'K': 'AAG', 'M': 'ATG', 'F': 'TTC', 'P': 'CCC',
            'S': 'TCC', 'T': 'ACC', 'W': 'TGG', 'Y': 'TAC', 'V': 'GTG', '*': 'TGA'
        }
    }
    
    variants = {}
    
    for system, codon_table in CODON_TABLES.items():
        mrna = []
        for aa in protein_sequence.upper():
            if aa in codon_table:
                mrna.append(codon_table[aa])
            else:
                mrna.append('NNN')
        
        if not protein_sequence.endswith('*'):
            mrna.append(codon_table['*'])
        
        sequence = ''.join(mrna)
        
        # Calculate stats
        gc = sequence.count('G') + sequence.count('C')
        gc_content = round(gc / len(sequence) * 100, 1)
        
        variants[system] = {
            'sequence': sequence,
            'length': len(sequence),
            'gc_content': gc_content,
            'description': {
                'human': 'Optimized for human cell expression',
                'ecoli': 'Optimized for E. coli expression',
                'yeast': 'Optimized for yeast expression',
                'cho': 'Optimized for CHO cell expression'
            }[system]
        }
    
    return variants
