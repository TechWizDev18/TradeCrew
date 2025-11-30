"""
Cache and Configuration Cleanup Script
Run this to clear all OpenAI/LLM related caches
"""
import os
import shutil

def clean_cache():
    """Remove all cached data that might interfere with Gemini"""
    
    cache_dirs = [
        ".crewai_cache",
        "__pycache__",
        ".cache",
        "crewai_cache",
        ".litellm_cache"
    ]
    
    print("🧹 Cleaning cache directories...")
    cleaned = False
    
    # Clean from current directory and subdirectories
    for root, dirs, files in os.walk('.'):
        for cache_dir in cache_dirs:
            cache_path = os.path.join(root, cache_dir)
            if os.path.exists(cache_path):
                try:
                    shutil.rmtree(cache_path)
                    print(f"✓ Removed: {cache_path}")
                    cleaned = True
                except Exception as e:
                    print(f"✗ Could not remove {cache_path}: {e}")
    
    # Clean Python bytecode
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pyc'):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"✓ Removed: {file_path}")
                    cleaned = True
                except Exception as e:
                    print(f"✗ Could not remove {file_path}: {e}")
    
    if not cleaned:
        print("✓ No cache files found. System is clean!")
    else:
        print("\n✅ Cache cleanup complete!")
    
    print("\n📝 Next steps:")
    print("1. Get your FREE Gemini API key from: https://aistudio.google.com/app/apikey")
    print("2. Update MY_GEMINI_KEY in crew.py with your key")
    print("3. Run: python src/crew.py")

if __name__ == "__main__":
    clean_cache()
