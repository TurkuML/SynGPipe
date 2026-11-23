import subprocess
import sys
import shutil
import os

# ---------------- Settings ----------------
env_name = sys.argv[1] if len(sys.argv) > 1 else None
python_version = "3.10"
requirements_file = "requirements.txt"

# ---------------- Utilities ----------------
def run(command, **kwargs):
    print(f"Running: {' '.join(command)}")
    # Use subprocess.run to show output live
    subprocess.run(command, check=True, **kwargs)

def conda_exists():
    return shutil.which("conda") is not None

# ---------------- Main ----------------
if not env_name:
    print("Error: No environment name provided.")
    print("Usage: python setup_env.py your-env-name")
    sys.exit(1)

if not conda_exists():
    print("Conda is not installed or not found in PATH.")
    print("Please install Anaconda or Miniconda first: https://docs.conda.io/en/latest/miniconda.html")
    sys.exit(1)

print(f"Creating Conda environment: {env_name} (Python {python_version})")
run(["conda", "create", "-n", env_name, f"python={python_version}", "-y"])

# ---------------- Update pip ----------------
print(f"Upgrading pip, setuptools, and wheel in environment '{env_name}'")
run([
    "conda", "run", "-n", env_name,
    "python", "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"
])

print(f"Installing Git inside the environment '{env_name}'")
run(["conda", "install", "-n", env_name, "git", "-y"])

print(f"Installing pip packages from {requirements_file}")
run(["conda", "run", "-n", env_name, "pip", "install", "-r", requirements_file])

print("Registering Jupyter kernel")
run([
    "conda", "run", "-n", env_name,
    "python", "-m", "ipykernel", "install",
    "--user", "--name", env_name,
    "--display-name", f"Python ({env_name})"
])

print("\n Setup complete!")
print(f" To activate the environment, run:")
print(f"  conda activate {env_name}")
