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
    
    # Process all images in the directory
    print(f"Starting analysis of images in: {images_dir}")
    results = []
    
    # Get all jpg and jpeg files in the directory (excluding processed folder)
    image_files = []
    for ext in ["*.jpg", "*.jpeg"]:
        image_files.extend([f for f in Path(images_dir).glob(ext) if "processed" not in str(f)])
    
    if not image_files:
        print(f"No JPG/JPEG images found in {images_dir}")
        return
    
    for image_file in image_files:
        print(f"\nProcessing image: {image_file.name}")
        
        try:
            # Use the main workflow that includes both structural analysis and new anomaly analysis
            print("- Running complete analysis workflow...")
            analysis_result = agent.analyze_image(str(image_file))
            
            if "error" in analysis_result:
                print(f"  Error in analysis: {analysis_result['error']}")
                results.append({"image": image_file.name, "error": analysis_result["error"]})
                continue
            
            # Check if we have charts detected
            if not analysis_result.get("charts"):
                print("  No charts identified in the image")
                results.append({"image": image_file.name, "error": "No charts identified"})
                continue
            
            charts_count = len(analysis_result["charts"])
            print(f"  Found {charts_count} charts")
            
            if analysis_result.get("anomaly_analysis"):
                print("  Anomaly analysis completed")
            else:
                print("  No anomaly analysis available")
            
            # Add successful result
            results.append({
                "image": image_file.name,
                "analysis": analysis_result
            })
            
            print("  Image processed successfully")
            
        except Exception as e:
            print(f"  Error processing image: {str(e)}")
            results.append({"image": image_file.name, "error": str(e)})
    
    # Summary
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    
    successful = sum(1 for r in results if "error" not in r)
    failed = sum(1 for r in results if "error" in r)
    
    print(f"Total images processed: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed analyses:")
        for result in results:
            if "error" in result:
                print(f"  - {result['image']}: {result['error']}")
    
    print(f"\nAnalysis results saved in: envs/data/")
    print(f"Processed images moved to: {images_dir}/processed/")
    print("="*50)

if __name__ == "__main__":
    main() 