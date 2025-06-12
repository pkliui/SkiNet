import os

# Check if running inside Docker
def is_running_in_docker():
    return os.path.exists("/.dockerenv")  # This file exists inside Docker
