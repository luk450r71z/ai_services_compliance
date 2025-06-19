"""
Test script to verify the AI Image Analysis Agent setup
"""
import os
import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import langgraph
        print("✓ LangGraph imported successfully")
    except ImportError as e:
        print(f"✗ LangGraph import failed: {e}")
        return False
    
    try:
        import langchain
        print("✓ LangChain imported successfully")
    except ImportError as e:
        print(f"✗ LangChain import failed: {e}")
        return False
    
    try:
        import openai
        print("✓ OpenAI imported successfully")
    except ImportError as e:
        print(f"✗ OpenAI import failed: {e}")
        return False
    
    try:
        from dotenv import load_dotenv
        print("✓ python-dotenv imported successfully")
    except ImportError as e:
        print(f"✗ python-dotenv import failed: {e}")
        return False
    
    return True

def test_project_structure():
    """Test if project structure is correct"""
    print("\nTesting project structure...")
    
    required_dirs = [
        "src",
        "assets/images",
        "assets/images/processed", 
        "envs",
        "envs/data"
    ]
    
    required_files = [
        "src/__init__.py",
        "src/agent.py",
        "src/main.py",
        "envs/run_analysis.sh",
        "envs/monitoring.py",
        "envs/Dockerfile",
        "envs/docker-compose.yml",
        "requirements.txt",
        "README.md"
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ Directory: {dir_path}")
        else:
            print(f"✗ Missing directory: {dir_path}")
            all_good = False
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ File: {file_path}")
        else:
            print(f"✗ Missing file: {file_path}")
            all_good = False
    
    return all_good

def test_env_config():
    """Test environment configuration"""
    print("\nTesting environment configuration...")
    
    # Check if .env exists
    if Path(".env").exists():
        print("✓ .env file exists")
        
        # Load and check API key
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key and api_key != "XXXXXXXX":
            print("✓ OpenAI API key is configured")
            return True
        else:
            print("⚠ OpenAI API key not configured or using placeholder")
            return False
    else:
        print("✗ .env file not found")
        print("  Please create a .env file with your OPENAI_API_KEY")
        return False

def test_agent_import():
    """Test if the agent can be imported"""
    print("\nTesting agent import...")
    
    try:
        # Add src to path
        sys.path.append(str(Path("src")))
        from agent import ImageAnalysisAgent
        print("✓ Agent module imported successfully")
        return True
    except Exception as e:
        print(f"✗ Agent import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("AI IMAGE ANALYSIS AGENT - SETUP TEST")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("Environment Config", test_env_config),
        ("Agent Import", test_agent_import)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The project is ready to use.")
        print("\nNext steps:")
        print("1. Add images to assets/images/")
        print("2. Run: python src/main.py")
        print("3. Or run: ./envs/run_analysis.sh")
    else:
        print("\n⚠ Some tests failed. Please fix the issues above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 