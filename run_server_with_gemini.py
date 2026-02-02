"""
Complete test - Start server, test OCR with Gemini, show logs
Run this in one terminal while keeping it open to see logs
"""
import subprocess
import sys
import time
import os

print("="*70)
print("GEMINI OCR API TEST")
print("="*70)

# Verify .env file exists
if not os.path.exists(".env"):
    print("\n✗ ERROR: .env file not found!")
    print("Create a .env file with: GEMINI_API_KEY=your-key-here")
    sys.exit(1)

# Load and verify API key
with open(".env", "r") as f:
    for line in f:
        if line.strip().startswith("GEMINI_API_KEY="):
            key = line.split("=", 1)[1].strip().strip('"').strip("'")
            if key and len(key) > 10:
                print(f"\n✓ GEMINI_API_KEY found: {key[:10]}...{key[-4:]}")
                break
    else:
        print("\n✗ WARNING: GEMINI_API_KEY not found in .env file")

print("\nStarting uvicorn server...")
print("="*70)
print("\nWATCH FOR THESE CONSOLE LOGS:")
print("  [OCR] Processing file: ...")
print("  [OCR] Gemini API key present: True")  
print("  [OCR] Attempting Gemini extraction...")
print("  [OCR] ✓ Gemini extraction successful")
print("\n" + "="*70)
print("\nPress Ctrl+C to stop the server")
print("="*70 + "\n")

try:
    # Start server in foreground so we see all logs
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "api_server:app",
        "--reload",
        "--log-level", "info"
    ])
except KeyboardInterrupt:
    print("\n\nServer stopped.")
