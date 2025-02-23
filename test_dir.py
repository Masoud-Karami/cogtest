import os
import sys
import subprocess

print("Script is running from:", os.getcwd())  # Current working directory
print("Script file location:", os.path.abspath(__file__))  # Full path of the script
print("Python interpreter:", sys.executable)  # Path of the Python binary being used
print("Python version:", sys.version)  # Python version

# Get the output of the "which python" command in the terminal
which_python = subprocess.run(["which", "python"], capture_output=True, text=True)
print("Terminal 'which python' output:", which_python.stdout.strip())
# 