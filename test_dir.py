import os

print("Script is running from:", os.getcwd())  # Current working directory
print("Script file location:", os.path.abspath(
    __file__))  # Full path of the script
