"""
AI Agent for Image Analysis using LangGraph
"""
import os
import json
import base64
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

class AnalysisState(BaseModel):
    """State for the image analysis workflow"""
    image_path: str = Field(description="Path to the current image being analyzed")
    image_content: Optional[str] = Field(default=None, description="Base64 encoded image content")
    structural_analysis: Optional[Dict[str, Any]] = Field(default=None, description="First stage structural analysis")
    anomaly_analysis: Optional[Dict[str, Any]] = Field(default=None, description="Second stage anomaly analysis")
    error: Optional[str] = Field(default=None, description="Error message if any")
    processed: bool = Field(default=False, description="Whether the image has been processed")

class ImageAnalysisAgent:
    """AI Agent for analyzing dashboard images"""
    
    def __init__(self, api_key: str):
        """Initialize the agent with OpenAI API key"""
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=api_key,
            temperature=0.1
        )
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for image analysis"""
        
        # Define the workflow
        workflow = StateGraph(AnalysisState)
        
        # Add nodes - using unique names that don't conflict with state fields
        workflow.add_node("load_image", self._load_image)
        workflow.add_node("perform_structural_analysis", self._structural_analysis)
        workflow.add_node("perform_anomaly_analysis", self._anomaly_analysis)
        workflow.add_node("save_analysis", self._save_analysis)
        workflow.add_node("move_to_processed", self._move_to_processed)
        
        # Define the flow
        workflow.set_entry_point("load_image")
        workflow.add_edge("load_image", "perform_structural_analysis")
        workflow.add_edge("perform_structural_analysis", "perform_anomaly_analysis")
        workflow.add_edge("perform_anomaly_analysis", "save_analysis")
        workflow.add_edge("save_analysis", "move_to_processed")
        workflow.add_edge("move_to_processed", END)
        
        return workflow.compile()
    
    def _load_image(self, state: AnalysisState) -> AnalysisState:
        """Load and encode the image"""
        try:
            with open(state.image_path, "rb") as image_file:
                image_data = image_file.read()
                state.image_content = base64.b64encode(image_data).decode('utf-8')
            return state
        except Exception as e:
            state.error = f"Error loading image: {str(e)}"
            return state
    
    def _structural_analysis(self, state: AnalysisState) -> AnalysisState:
        """First stage: Chart identification using the unified chart_explorer method"""
        if state.error:
            return state
            
        try:
            # Use the chart_explorer method directly to avoid duplication
            result = self.chart_explorer(state.image_path)
            
            if "error" in result:
                state.error = result["error"]
                return state
            
            state.structural_analysis = result
            return state
            
        except Exception as e:
            state.error = f"Error in structural analysis: {str(e)}"
            return state
    
    def _anomaly_analysis(self, state: AnalysisState) -> AnalysisState:
        """Análisis contextual y ejecutivo de cada subgráfico, con severidad, rango de picos, desvío y picos anómalos."""
        if state.error or not state.structural_analysis:
            return state
            
        try:
            import json
            charts = state.structural_analysis.get("charts", [])
            
            # Process charts in smaller batches to avoid JSON truncation
            all_analysis = []
            batch_size = 2  # Process 2 charts at a time (smaller batches to avoid truncation)
            
            failed_charts = []
            
            for i in range(0, len(charts), batch_size):
                batch = charts[i:i+batch_size]
                
                # Try to process batch with retry
                batch_success = False
                for attempt in range(2):  # Try twice
                    prompt = f'''Analyze the recent period of these {len(batch)} dashboard charts.

CHARTS:
{json.dumps(batch, indent=2, ensure_ascii=False)}

CRITICAL READING RULES:
1. FIRST: Read the Y-axis scale (min and max values shown on left side)
2. SECOND: Look at the rightmost data points only (current period) 
3. THIRD: Estimate current values based on Y-axis scale, NOT invented numbers
4. Example: If Y-axis shows 0-120 and current line is near bottom, value is ~0-10, NOT 800
5. IGNORE historical peaks, focus only on current rightmost values

RESPONSE FORMAT - MUST BE VALID JSON:
{{
  "chart_analysis": [
    {{
      "chart_id": "chart_1",
      "name": "Chart Name",
      "analysis": "Recent values around X",
      "severity": "normal",
      "normal_range": {{"min": 0, "max": 400}},
      "deviation": 10,
      "anormal_peaks": []
    }}
  ]
}}

CRITICAL: Return ONLY complete, valid JSON. Do NOT truncate. Include ALL required fields for EVERY chart.'''
                    
                    message = HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{state.image_content}"
                                }
                            }
                        ]
                    )
                    
                    response = self.llm.invoke([message])
                    batch_analysis = self._parse_json_response(response.content)
                    
                    if batch_analysis and "chart_analysis" in batch_analysis and len(batch_analysis["chart_analysis"]) > 0:
                        all_analysis.extend(batch_analysis["chart_analysis"])
                        batch_success = True
                        break
                
                # If batch failed, collect individual charts for single processing
                if not batch_success:
                    failed_charts.extend(batch)
            
            # Process failed charts individually
            for chart in failed_charts:
                try:
                    prompt = f'''Analyze the recent period of this single dashboard chart.

CHART:
{json.dumps([chart], indent=2, ensure_ascii=False)}

CRITICAL READING RULES:
1. FIRST: Read the Y-axis scale (min and max values shown on left side)
2. SECOND: Look at the rightmost data points only (current period) 
3. THIRD: Estimate current values based on Y-axis scale, NOT invented numbers
4. Example: If Y-axis shows 0-120 and current line is near bottom, value is ~0-10, NOT 800
5. IGNORE historical peaks, focus only on current rightmost values

RESPONSE FORMAT - MUST BE VALID JSON:
{{
  "chart_analysis": [
    {{
      "chart_id": "{chart['chart_id']}",
      "name": "Chart Name",
      "analysis": "Recent values around X",
      "severity": "normal",
      "normal_range": {{"min": 0, "max": 400}},
      "deviation": 10,
      "anormal_peaks": []
    }}
  ]
}}

CRITICAL: Return ONLY complete, valid JSON. Do NOT truncate. Include ALL required fields.'''
                    
                    message = HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{state.image_content}"
                                }
                            }
                        ]
                    )
                    
                    response = self.llm.invoke([message])
                    single_analysis = self._parse_json_response(response.content)
                    
                    if single_analysis and "chart_analysis" in single_analysis:
                        all_analysis.extend(single_analysis["chart_analysis"])
                        
                except Exception as e:
                    print(f"Failed to process chart {chart['chart_id']}: {e}")
            
            state.anomaly_analysis = {"chart_analysis": all_analysis}
            return state
            
        except Exception as e:
            state.error = f"Error in anomaly analysis: {str(e)}"
            return state
    
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON response from GPT-4o"""
        try:
            # Try to find JSON blocks
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                json_str = content[json_start:json_end].strip()
            else:
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                else:
                    raise ValueError("No JSON content found in response")
            
            # Try to parse the JSON
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Fix common JSON formatting issues
                import re
                fixed_json = json_str
                
                # Fix missing quotes around object keys
                fixed_json = re.sub(r'(\w+):', r'"\1":', fixed_json)
                
                # Fix trailing commas
                fixed_json = re.sub(r',(\s*[}\]])', r'\1', fixed_json)
                
                # Fix double braces in normal_range
                fixed_json = re.sub(r'\{\{\s*"min":\s*(\d+),\s*"max":\s*(\d+)\s*\}\}', r'{"min": \1, "max": \2}', fixed_json)
                
                # Complete incomplete structures
                if fixed_json.count('{') > fixed_json.count('}'):
                    fixed_json += '}' * (fixed_json.count('{') - fixed_json.count('}'))
                if fixed_json.count('[') > fixed_json.count(']'):
                    fixed_json += ']' * (fixed_json.count('[') - fixed_json.count(']'))
                
                return json.loads(fixed_json)
                
        except Exception as e:
            print(f"Error parsing JSON response: {e}")
            return self._create_empty_response()
    
    def _create_empty_response(self) -> Dict[str, Any]:
        """Create empty response structure"""
        return {
            "charts": [],
            "chart_analysis": []
        }
    
    def _save_analysis(self, state: AnalysisState) -> AnalysisState:
        """Save the analysis results as separate files"""
        if state.error:
            return state
            
        try:
            # Create output directory if it doesn't exist
            output_dir = Path("envs/data")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique analysis ID
            analysis_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            image_name = Path(state.image_path).stem
            
            # Save chart detection results directly
            if state.structural_analysis:
                chart_file = output_dir / f"charts_{analysis_id}_{image_name}.json"
                with open(chart_file, 'w', encoding='utf-8') as f:
                    json.dump(state.structural_analysis, f, indent=2, ensure_ascii=False)
            
            # Save anomaly analysis results directly  
            if state.anomaly_analysis:
                anomaly_file = output_dir / f"anomalies_{analysis_id}_{image_name}.json"
                with open(anomaly_file, 'w', encoding='utf-8') as f:
                    json.dump(state.anomaly_analysis, f, indent=2, ensure_ascii=False)
            
            return state
            
        except Exception as e:
            state.error = f"Error saving analysis: {str(e)}"
            return state
    
    def _move_to_processed(self, state: AnalysisState) -> AnalysisState:
        """Move the processed image to processed folder"""
        if state.error:
            return state
            
        try:
            # Create processed directory if it doesn't exist
            processed_dir = Path("assets/images/processed")
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Move the image
            source_path = Path(state.image_path)
            dest_path = processed_dir / source_path.name
            
            if source_path.exists():
                import shutil
                shutil.move(str(source_path), str(dest_path))
            
            state.processed = True
            return state
            
        except Exception as e:
            state.error = f"Error moving processed image: {str(e)}"
            return state
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze a single image"""
        state = AnalysisState(image_path=image_path)
        final_state = self.workflow.invoke(state)
        
        # Access the error from the state dict
        if final_state.get('error'):
            return {"error": final_state['error']}
            
        return {
            "charts": final_state.get('structural_analysis', {}).get('charts', []),
            "anomaly_analysis": final_state.get('anomaly_analysis')
        }
    
    def chart_explorer(self, jpg_path: str) -> Dict[str, Any]:
        """
        Analyze a dashboard image to identify and locate charts/graphs.
        
        Args:
            jpg_path: Path to the image file to analyze
            
        Returns:
            Dict containing the list of identified charts with their positions and metadata
        """
        try:
            # Initialize state and load image
            state = AnalysisState(image_path=jpg_path)
            state = self._load_image(state)
            if state.error:
                return {"error": state.error}
            
            # Create system and human messages
            system_message = SystemMessage(content="""You are a precise dashboard analyzer that MUST follow these CRITICAL rules:

1. EXACT TITLE READING:
   - Read and transcribe titles CHARACTER BY CHARACTER
   - Never make assumptions about what a title "should" be
   - Never modify or "correct" what you see
   - If text is unclear, mark as [unclear] - never guess

2. THOROUGH DETECTION:
   - Detect ALL data visualization components
   - Scan the ENTIRE image systematically, quadrant by quadrant
   - Include all charts regardless of size or position
   - Double-check every section to ensure no charts are missed

3. VERIFICATION PROCESS:
   - After identifying each chart, verify its title matches EXACTLY
   - Cross-check that no charts were missed in any section
   - Confirm each identified component is actually a data visualization
   - Do a final verification pass to catch any missed charts

CRITICAL: Your role is to TRANSCRIBE EXACTLY what you see, not interpret it.
Think of yourself as a precise optical character reader - your job is to capture the exact text shown, not to make it "better" or "more logical".""")

            human_message = HumanMessage(content=[
                {
                    "type": "text",
                    "text": """TRANSCRIBE DASHBOARD CHARTS EXACTLY AS SHOWN

TRANSCRIPTION PROCESS:
1. For each chart:
   - Look at the title area
   - Transcribe each character one by one
   - Do not move to the next chart until you've verified each character
   - If any character is unclear, mark entire title as [unclear]

2. VISUAL VERIFICATION:
   - After transcribing each title, look at it again
   - Compare your transcription character-by-character with what you see
   - If there's ANY doubt, start the transcription again

3. POSITION RECORDING:
   - Record exact pixel coordinates
   - Note row and column position

Return the analysis in this EXACT JSON format:

{
  "charts": [
    {
      "chart_id": "chart_1",
      "name": "TRANSCRIBED_TITLE_CHARACTER_BY_CHARACTER",
      "pix_position": {
        "x1": left_coordinate,
        "y1": top_coordinate,
        "x2": right_coordinate,
        "y2": bottom_coordinate
      },
      "row_col_position": {
        "row": row_number,
        "column": column_number
      }
    }
  ]
}

IMPORTANT: Think of yourself as a camera capturing text - your job is to photograph the exact characters, not interpret them."""
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{state.image_content}"
                    }
                }
            ])
            
            # Get the chart analysis
            response = self.llm.invoke([system_message, human_message])
            analysis = self._parse_json_response(response.content)
            
            # Ensure all required fields are present and validate structure
            for i, chart in enumerate(analysis.get("charts", [])):
                # Validate required fields
                if "chart_id" not in chart or not chart["chart_id"]:
                    chart["chart_id"] = f"chart_{i+1}"
                if "name" not in chart:
                    chart["name"] = "Unknown Chart"
                if "pix_position" not in chart:
                    chart["pix_position"] = {"x1": None, "y1": None, "x2": None, "y2": None}
                elif not isinstance(chart["pix_position"], dict):
                    chart["pix_position"] = {"x1": None, "y1": None, "x2": None, "y2": None}
                else:
                    # Ensure all required keys in pix_position
                    for key in ["x1", "y1", "x2", "y2"]:
                        if key not in chart["pix_position"]:
                            chart["pix_position"][key] = None
                if "row_col_position" not in chart:
                    chart["row_col_position"] = None
            
            return analysis
            
        except Exception as e:
            return {"error": str(e)}

 