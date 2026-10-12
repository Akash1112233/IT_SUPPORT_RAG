import os
from pathlib import Path

# Define the root directory name
ROOT_DIR = Path(".")

# Define the directories to create
directories = [
    f"{ROOT_DIR}/app/api",
    f"{ROOT_DIR}/app/core",
    f"{ROOT_DIR}/app/rag",
    f"{ROOT_DIR}/app/services",
    f"{ROOT_DIR}/data",
    f"{ROOT_DIR}/templates",
    f"{ROOT_DIR}/static",
    f"{ROOT_DIR}/uploads",
    f"{ROOT_DIR}/tests",
]

# Define the files to create
files = [
    f"{ROOT_DIR}/app/main.py",
    f"{ROOT_DIR}/requirements.txt",
    f"{ROOT_DIR}/ingest_sample_kb.py",
    f"{ROOT_DIR}/run.py",
    f"{ROOT_DIR}.env",  # Creates the hidden environment file
]

def create_structure():
    # 1. Create all directories
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

    # 2. Create all empty files
    for file_path in files:
        Path(file_path).touch(exist_ok=True)
        print(f"Created file: {file_path}")
        
    print("\n✅ Project structure successfully created!")

if __name__ == "__main__":
    create_structure()