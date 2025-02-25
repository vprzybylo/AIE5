import os
import sys
from pathlib import Path

# Add the midterm/src directory to Python path
src_path = Path(__file__).parent / "midterm" / "src"
sys.path.append(str(src_path))

# Import and run the actual application
from ui.app import main

if __name__ == "__main__":
    main()