"""
Environment Variable Loader

Utility functions for loading environment variables from .env file.
"""

import os
from pathlib import Path
from typing import Optional


def load_env_file(env_path: Optional[str] = None) -> dict:
    """
    Load environment variables from a .env file.
    
    Args:
        env_path: Path to .env file. If None, searches for .env in current directory and parent directories.
    
    Returns:
        Dictionary of environment variables
    """
    if env_path is None:
        # Search for .env file starting from current directory and going up
        current_dir = Path.cwd()
        for parent in [current_dir] + list(current_dir.parents):
            env_file = parent / '.env'
            if env_file.exists():
                env_path = str(env_file)
                break
        
        if env_path is None:
            # Try in the project root (assuming we're in llm-ideology-viz)
            project_root = Path(__file__).parent.parent.parent
            env_file = project_root / '.env'
            if env_file.exists():
                env_path = str(env_file)
    
    if env_path is None or not os.path.exists(env_path):
        return {}
    
    env_vars = {}
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse KEY=VALUE format
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                env_vars[key] = value
    
    return env_vars


def get_hf_token(env_path: Optional[str] = None) -> Optional[str]:
    """
    Get HuggingFace token from .env file or environment variable.
    
    Checks in order:
    1. HF_TOKEN environment variable
    2. HUGGINGFACE_TOKEN environment variable
    3. .env file (HF_TOKEN or HUGGINGFACE_TOKEN)
    
    Args:
        env_path: Path to .env file. If None, searches automatically.
    
    Returns:
        HuggingFace token or None if not found
    """
    # First check environment variables
    token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
    if token:
        return token
    
    # Then check .env file
    env_vars = load_env_file(env_path)
    token = env_vars.get('HF_TOKEN') or env_vars.get('HUGGINGFACE_TOKEN')
    
    return token


def load_env_to_os(env_path: Optional[str] = None):
    """
    Load environment variables from .env file into os.environ.
    
    Args:
        env_path: Path to .env file. If None, searches automatically.
    """
    env_vars = load_env_file(env_path)
    for key, value in env_vars.items():
        if key not in os.environ:  # Don't override existing env vars
            os.environ[key] = value

