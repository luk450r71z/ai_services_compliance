"""
Monitoring script for AI Image Analysis Agent
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

class AgentMonitor:
    def __init__(self, data_dir: str = "envs/data"):
        self.data_dir = Path(data_dir)
    
    def get_all_analyses(self) -> List[Dict[str, Any]]:
        """Get all analysis results"""
        analyses = []
        if not self.data_dir.exists():
            return analyses
        
        for json_file in self.data_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    analyses.append(json.load(f))
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
        
        return analyses
    
    def generate_report(self) -> str:
        """Generate monitoring report"""
        analyses = self.get_all_analyses()
        
        if not analyses:
            return "No analysis data available yet."
        
        report = []
        report.append("=" * 50)
        report.append("AI IMAGE ANALYSIS AGENT - REPORT")
        report.append("=" * 50)
        report.append(f"Total analyses: {len(analyses)}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Status summary
        status_counts = {}
        for analysis in analyses:
            status = analysis.get('content', {}).get('estado_general', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        if status_counts:
            report.append("STATUS DISTRIBUTION:")
            for status, count in status_counts.items():
                report.append(f"  {status}: {count}")
        
        report.append("=" * 50)
        return "\n".join(report)

def main():
    monitor = AgentMonitor()
    report = monitor.generate_report()
    print(report)

if __name__ == "__main__":
    main() 