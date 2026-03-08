"""
AEGIS AI - Report Generator
Generates visual scorecards and audit reports.
"""

import os
from typing import Dict, Any
from datetime import datetime


class ReportGenerator:
    """Generate HTML and text audit reports."""
    
    def generate_scorecard_html(self, audit_result: Dict[str, Any]) -> str:
        """Generate an HTML visual scorecard from audit results."""
        
        model_name = audit_result.get('model_name', 'Unknown Model')
        model_type = audit_result.get('model_type', 'Unknown')
        verdict = audit_result.get('overall_verdict', {})
        fairness = audit_result.get('fairness', {})
        metrics = fairness.get('metrics', {})
        recommendations = audit_result.get('recommendations', [])
        performance = audit_result.get('performance', {})
        
        # Build metrics rows
        metric_rows = ""
        for key, result in metrics.items():
            passed = result.get('passed', False)
            status_icon = "✅" if passed else "❌"
            status_class = "pass" if passed else "fail"
            metric_name = result.get('metric', key)
            
            detail = ""
            if 'selection_rates' in result:
                detail = str(result['selection_rates'])
            elif 'true_positive_rates' in result:
                detail = str(result['true_positive_rates'])
            elif 'brier_score' in result:
                detail = f"Brier Score: {result['brier_score']}"
            elif 'score' in result:
                detail = f"Score: {result['score']}/100"
            elif 'sensitive_features_used' in result:
                used = result['sensitive_features_used']
                detail = f"Sensitive features used: {used}" if used else "No sensitive features used"
            
            metric_rows += f"""
            <tr class="{status_class}">
                <td>{status_icon} {metric_name}</td>
                <td>{result.get('attribute', '-')}</td>
                <td>{detail}</td>
                <td class="status-{status_class}">{('PASS' if passed else 'FAIL')}</td>
            </tr>
            """
        
        # Build recommendations
        rec_html = ""
        for rec in recommendations:
            severity = rec.get('severity', 'INFO')
            severity_class = severity.lower()
            rec_html += f"""
            <div class="recommendation {severity_class}">
                <span class="severity-badge">{severity}</span>
                <strong>{rec.get('category', '')}</strong>
                <p><em>Finding:</em> {rec.get('finding', '')}</p>
                <p><em>Recommendation:</em> {rec.get('recommendation', '')}</p>
                <p><em>Impact:</em> {rec.get('impact', '')}</p>
            </div>
            """
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AEGIS AI - Ethical Audit Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    max-width: 900px;
                    margin: 0 auto;
                    padding: 20px;
                    background: #0a0a1a;
                    color: #e0e0e0;
                }}
                h1 {{
                    text-align: center;
                    color: #00d4ff;
                    border-bottom: 2px solid #00d4ff;
                    padding-bottom: 10px;
                }}
                h2 {{ color: #00d4ff; }}
                .header {{
                    text-align: center;
                    padding: 20px;
                    background: linear-gradient(135deg, #0a1628, #1a2744);
                    border-radius: 10px;
                    margin-bottom: 20px;
                    border: 1px solid #00d4ff33;
                }}
                .grade-circle {{
                    width: 120px; height: 120px;
                    border-radius: 50%;
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 48px;
                    font-weight: bold;
                    margin: 10px;
                    border: 4px solid;
                }}
                .grade-green {{ border-color: #22c55e; color: #22c55e; }}
                .grade-yellow {{ border-color: #eab308; color: #eab308; }}
                .grade-orange {{ border-color: #f97316; color: #f97316; }}
                .grade-red {{ border-color: #ef4444; color: #ef4444; }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                    background: #111827;
                    border-radius: 8px;
                    overflow: hidden;
                }}
                th {{
                    background: #1e3a5f;
                    color: #00d4ff;
                    padding: 12px;
                    text-align: left;
                }}
                td {{ padding: 10px 12px; border-bottom: 1px solid #1f2937; }}
                tr:hover {{ background: #1f2937; }}
                .pass {{ }}
                .fail {{ background: #1a0000; }}
                .status-pass {{ color: #22c55e; font-weight: bold; }}
                .status-fail {{ color: #ef4444; font-weight: bold; }}
                .recommendation {{
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 8px;
                    border-left: 4px solid;
                }}
                .critical {{ border-color: #ef4444; background: #1a0000; }}
                .high {{ border-color: #f97316; background: #1a1000; }}
                .info {{ border-color: #22c55e; background: #001a00; }}
                .severity-badge {{
                    padding: 2px 8px;
                    border-radius: 4px;
                    font-size: 12px;
                    font-weight: bold;
                    margin-right: 8px;
                }}
                .critical .severity-badge {{ background: #ef4444; color: white; }}
                .high .severity-badge {{ background: #f97316; color: white; }}
                .info .severity-badge {{ background: #22c55e; color: white; }}
                .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <h1>⚖️ AEGIS AI - Ethical Audit Report</h1>
            
            <div class="header">
                <h2>{model_name}</h2>
                <p>Model Type: {model_type} | Accuracy: {performance.get('accuracy', 'N/A')}</p>
                <div class="grade-circle grade-{verdict.get('color', 'red')}">
                    {verdict.get('grade', 'N/A')}
                </div>
                <p>Fairness Score: <strong>{verdict.get('score', 0)}%</strong> — {verdict.get('label', 'Unknown')}</p>
                <p>Tests Passed: {verdict.get('tests_passed', 0)} / {verdict.get('total_tests', 0)}</p>
            </div>
            
            <h2>📊 Fairness Metrics Scorecard</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Attribute</th>
                    <th>Details</th>
                    <th>Status</th>
                </tr>
                {metric_rows}
            </table>
            
            <h2>💡 Recommendations</h2>
            {rec_html}
            
            <div class="footer">
                <p>Generated by AEGIS AI — Ethical AI Auditing Framework</p>
                <p>Bluebit Hackathon 2026 | Team MISAL PAV</p>
                <p>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def save_report(self, audit_result: Dict[str, Any], output_path: str):
        """Save the HTML report to a file."""
        html = self.generate_scorecard_html(audit_result)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"📄 Report saved → {output_path}")
        return output_path
    
    def generate_text_summary(self, audit_result: Dict[str, Any]) -> str:
        """Generate a plain-text summary of the audit."""
        verdict = audit_result.get('overall_verdict', {})
        fairness = audit_result.get('fairness', {})
        metrics = fairness.get('metrics', {})
        recommendations = audit_result.get('recommendations', [])
        
        lines = [
            "=" * 60,
            "  AEGIS AI — ETHICAL AI AUDIT REPORT",
            "=" * 60,
            f"Model: {audit_result.get('model_name', 'Unknown')}",
            f"Type: {audit_result.get('model_type', 'Unknown')}",
            f"Accuracy: {audit_result.get('performance', {}).get('accuracy', 'N/A')}",
            f"Fairness Score: {verdict.get('score', 0)}% (Grade: {verdict.get('grade', 'N/A')})",
            f"Verdict: {verdict.get('label', 'Unknown')}",
            "",
            "-" * 60,
            "  METRICS SCORECARD",
            "-" * 60,
        ]
        
        for key, result in metrics.items():
            status = "✅ PASS" if result.get('passed') else "❌ FAIL"
            lines.append(f"  {result.get('metric', key)}: {status}")
        
        lines.extend([
            "",
            "-" * 60,
            "  RECOMMENDATIONS",
            "-" * 60,
        ])
        
        for rec in recommendations:
            lines.append(f"  [{rec.get('severity', 'INFO')}] {rec.get('category', '')}")
            lines.append(f"    {rec.get('recommendation', '')}")
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
