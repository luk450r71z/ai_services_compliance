"""
Main execution script for the AI Image Analysis Agent
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent))

from agent import ImageAnalysisAgent

def main():
    """Main execution function"""
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "XXXXXXXX":
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file")
        sys.exit(1)
    
    # Initialize the agent
    print("Initializing AI Image Analysis Agent...")
    agent = ImageAnalysisAgent(api_key)
    
    # Check if images directory exists
    images_dir = "assets/images"
    if not Path(images_dir).exists():
        print(f"Creating images directory: {images_dir}")
        Path(images_dir).mkdir(parents=True, exist_ok=True)
        print(f"Please add images to the {images_dir} directory and run again")
        return
    
    # Analyze all images
    print(f"Starting analysis of images in: {images_dir}")
    results = agent.analyze_all_images(images_dir)
    
    # Summary
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    
    successful = 0
    failed = 0
    
    for result in results:
        # Handle LangGraph result format (dict instead of AnalysisState)
        if isinstance(result, dict):
            if result.get('error'):
                failed += 1
            else:
                successful += 1
        else:
            # Fallback for AnalysisState object
            if hasattr(result, 'error') and result.error:
                failed += 1
            else:
                successful += 1
    
    print(f"Total images processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed analyses:")
        for result in results:
            error = None
            if isinstance(result, dict):
                error = result.get('error')
            elif hasattr(result, 'error'):
                error = result.error
            
            if error:
                # Get filename from result
                filename = "unknown"
                if isinstance(result, dict):
                    filename = Path(result.get('image_path', 'unknown')).name
                elif hasattr(result, 'image_path'):
                    filename = Path(result.image_path).name
                print(f"  - {filename}: {error}")
    
    print(f"\nAnalysis results saved in: envs/data/")
    print(f"Processed images moved to: {images_dir}/processed/")
    print("="*50)

if __name__ == "__main__":
    main() 