import subprocess
import sys
import os

def install_requirements():
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        return

    print("📦 Installing dependencies...\n")

    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ])

    print("\n✅ All dependencies installed successfully!")

if __name__ == "__main__":
    install_requirements()