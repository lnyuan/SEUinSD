import os
import sys
import time
import json
import numpy as np
from pathlib import Path
import torch

class TimestampedLogger:
    """Redirects stdout to a file with timestamps."""
    
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "a")

    def write(self, message):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        self.logfile.close()
        sys.stdout = self.terminal

def setup_logging(log_path: str) -> TimestampedLogger:
    """
    Sets up logging to a file with timestamps.

    Args:
        log_path (str): Path to the log file.

    Returns:
        TimestampedLogger: Configured logger instance.
    """
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logger = TimestampedLogger(log_path)
        sys.stdout = logger
        return logger
    except Exception as e:
        print(f"Error setting up logging: {e}")
        raise

def save_error_type_to_file(error_types: list, filename: str = 'error_type.txt'):
    """
    Saves a list of error types to a text file.

    Args:
        error_types (list): List of error types to save.
        filename (str): Output file path.
    """
    try:
        with open(filename, 'w') as f:
            for item in error_types:
                f.write(item + '\n')
        print(f"List saved to {filename}")
    except Exception as e:
        print(f"Error saving error types to {filename}: {e}")

def append_error_type_to_file(error_type: str, filename: str = 'error_types_log.txt'):
    """
    Appends an error type to a text file.

    Args:
        error_type (str): Error type to append.
        filename (str): Output file path.
    """
    try:
        with open(filename, 'a') as f:
            f.write(error_type + '\n')
        print(f"Appended {error_type} to {filename}")
    except Exception as e:
        print(f"Error appending to {filename}: {e}")

def load_error_type_from_file(filename: str = 'error_type.txt') -> list:
    """
    Loads error types from a text file.

    Args:
        filename (str): Input file path.

    Returns:
        list: List of error types.
    """
    try:
        with open(filename, 'r') as f:
            error_types = [line.strip() for line in f]
        print(f"List loaded from {filename}")
        return error_types
    except Exception as e:
        print(f"Error loading error types from {filename}: {e}")
        return []

def append_error_type_to_dict(error_type: str, clip_score: float, filename: str = 'error_types_log.json'):
    """
    Appends an error type and its CLIP score to a JSON file.

    Args:
        error_type (str): Error type identifier.
        clip_score (float): CLIP score for the generated image.
        filename (str): Output JSON file path.
    """
    try:
        error_clipscore_dict = {}
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                error_clipscore_dict = json.load(f)
        
        error_clipscore_dict[error_type] = clip_score
        with open(filename, 'w') as f:
            json.dump(error_clipscore_dict, f, indent=4)
        print(f"Appended {error_type}: {clip_score} to {filename}")
    except Exception as e:
        print(f"Error appending to {filename}: {e}")

def create_image_folder(imagefolder_path: str):
    """
    Creates a folder for storing images if it doesn't exist.

    Args:
        imagefolder_path (str): Path to the image folder.
    """
    try:
        if os.path.exists(imagefolder_path):
            print(f"{imagefolder_path} folder already exists!")
        else:
            os.makedirs(imagefolder_path)
            print(f"{imagefolder_path} folder created!")
    except Exception as e:
        print(f"Error creating folder {imagefolder_path}: {e}")