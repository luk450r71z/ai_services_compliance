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
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

class AnalysisState(BaseModel):
    """State for the image analysis workflow"""
    image_path: str = Field(description="Path to the current image being analyzed")
    image_content: Optional[str] = Field(default=None, description="Base64 encoded image content")
    analysis_result: Optional[Dict[str, Any]] = Field(default=None, description="Analysis result")
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
        
        # Add nodes
        workflow.add_node("load_image", self._load_image)
        workflow.add_node("analyze_image", self._analyze_image)
        workflow.add_node("save_analysis", self._save_analysis)
        workflow.add_node("move_to_processed", self._move_to_processed)
        
        # Define the flow
        workflow.set_entry_point("load_image")
        workflow.add_edge("load_image", "analyze_image")
        workflow.add_edge("analyze_image", "save_analysis")
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
    
    def _analyze_image(self, state: AnalysisState) -> AnalysisState:
        """Analyze the image using GPT-4o Vision"""
        if state.error:
            return state
            
        try:
            # Create the analysis prompt
            system_prompt = """Eres un experto analista de dashboards y sistemas de monitoreo. 
            Tu tarea es analizar la imagen del dashboard proporcionada y detectar:
            
            1. **Estado general del sistema**: ¿El dashboard muestra un estado normal, de advertencia o crítico?
            2. **Variables y métricas importantes**: Identifica las métricas clave que se están mostrando
            3. **Anomalías detectadas**: ¿Hay valores fuera de rango, errores, o comportamientos inusuales?
            4. **Información relevante**: Cualquier otro dato importante que pueda ser útil
            
            Proporciona un análisis detallado pero conciso en formato JSON con la siguiente estructura:
            {
                "estado_general": "normal/advertencia/critico",
                "metricas_principales": ["lista", "de", "metricas"],
                "anomalias_detectadas": ["lista", "de", "anomalias"],
                "informacion_relevante": "texto descriptivo",
                "resumen": "resumen ejecutivo del estado"
            }"""
            
            # Create the message with the image
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": system_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{state.image_content}"
                        }
                    }
                ]
            )
            
            # Get the analysis from GPT-4o
            response = self.llm.invoke([message])
            
            # Parse the response
            try:
                # Try to extract JSON from the response
                content = response.content
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    json_str = content[json_start:json_end].strip()
                else:
                    # If no JSON block, try to find JSON in the text
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                    else:
                        json_str = content
                
                analysis_result = json.loads(json_str)
            except json.JSONDecodeError:
                # If JSON parsing fails, create a structured response
                analysis_result = {
                    "estado_general": "indeterminado",
                    "metricas_principales": [],
                    "anomalias_detectadas": [],
                    "informacion_relevante": content,
                    "resumen": content[:200] + "..." if len(content) > 200 else content
                }
            
            state.analysis_result = analysis_result
            return state
            
        except Exception as e:
            state.error = f"Error analyzing image: {str(e)}"
            return state
    
    def _save_analysis(self, state: AnalysisState) -> AnalysisState:
        """Save the analysis result to JSON file"""
        if state.error or not state.analysis_result:
            return state
            
        try:
            # Create data directory if it doesn't exist
            data_dir = Path("envs/data")
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate unique ID for the analysis
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_id = f"analysis_{timestamp}_{Path(state.image_path).stem}"
            
            # Create analysis record
            analysis_record = {
                "id": file_id,
                "file": Path(state.image_path).name,
                "content": state.analysis_result,
                "datetime": datetime.now().isoformat()
            }
            
            # Save to JSON file
            output_file = data_dir / f"{file_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_record, f, indent=2, ensure_ascii=False)
            
            print(f"Analysis saved to: {output_file}")
            return state
            
        except Exception as e:
            state.error = f"Error saving analysis: {str(e)}"
            return state
    
    def _move_to_processed(self, state: AnalysisState) -> AnalysisState:
        """Move the processed image to the processed folder"""
        if state.error:
            return state
            
        try:
            # Create processed directory if it doesn't exist
            processed_dir = Path("assets/images/processed")
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Move the file
            source_path = Path(state.image_path)
            dest_path = processed_dir / source_path.name
            
            # If file already exists, add timestamp
            if dest_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                name_parts = source_path.stem, source_path.suffix
                dest_path = processed_dir / f"{name_parts[0]}_{timestamp}{name_parts[1]}"
            
            source_path.rename(dest_path)
            print(f"Image moved to: {dest_path}")
            
            state.processed = True
            return state
            
        except Exception as e:
            state.error = f"Error moving file: {str(e)}"
            return state
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze a single image"""
        state = AnalysisState(image_path=image_path)
        result = self.workflow.invoke(state)
        return result
    
    def analyze_all_images(self, images_dir: str = "assets/images") -> List[Dict[str, Any]]:
        """Analyze all images in the specified directory"""
        images_path = Path(images_dir)
        if not images_path.exists():
            print(f"Directory {images_dir} does not exist")
            return []
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        image_files = [
            f for f in images_path.iterdir() 
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            print(f"No image files found in {images_dir}")
            return []
        
        results = []
        for image_file in image_files:
            print(f"\nProcessing: {image_file.name}")
            try:
                result = self.analyze_image(str(image_file))
                results.append(result)
                
                # Handle LangGraph result format (dict instead of AnalysisState)
                if isinstance(result, dict):
                    error = result.get('error')
                    if error:
                        print(f"Error processing {image_file.name}: {error}")
                    else:
                        print(f"Successfully processed: {image_file.name}")
                else:
                    # Fallback for AnalysisState object
                    if hasattr(result, 'error') and result.error:
                        print(f"Error processing {image_file.name}: {result.error}")
                    else:
                        print(f"Successfully processed: {image_file.name}")
                    
            except Exception as e:
                print(f"Unexpected error processing {image_file.name}: {str(e)}")
        
        return results 