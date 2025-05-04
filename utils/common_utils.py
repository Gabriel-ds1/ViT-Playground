import os
import re
from datetime import datetime


def create_checkpoint_path(base_path='./checkpoints', folder_prefix='run'):
    """
    Creates a new folder with an incremented number and a timestamp.
    
    Args:
        base_path (str): Directory where the new folder should be created.
        folder_prefix (str): Prefix for the folder name (default 'run').
        
    Returns:
        str: The path to the newly created folder.
    """
    # Ensure base path exists
    os.makedirs(base_path, exist_ok=True)

    # List existing folders that match the pattern 'run_%3d'
    existing_folders = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name)) and re.match(rf'^{re.escape(folder_prefix)}(\d{{3}})', name)]
    # Extract numbers from the matching folder names
    numbers = []
    for folder in existing_folders:
        match = re.search(r'(\d{3})', folder)
        if match:
            numbers.append(int(match.group(1)))

    # Determine next run number
    next_number = max(numbers) + 1 if numbers else 1

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = f"{folder_prefix}{next_number:03d}_{timestamp}"
    ckpt_folder_path = os.path.join(base_path, folder_name)

    os.makedirs(ckpt_folder_path, exist_ok=False)
    return ckpt_folder_path

    
def get_checkpoint_path(base_path='./checkpoints', folder_prefix='run'):
    # Ensure base path exists
    os.makedirs(base_path, exist_ok=True)

    # List existing folders that match the pattern 'run_%3d'
    existing_folders = [name for name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, name)) and re.match(rf'^{re.escape(folder_prefix)}(\d{{3}})', name)]
    if not existing_folders:
        raise Exception("Please create a new run folder first by setting new_run=True")
    ckpt_folder_path = os.path.join(base_path, existing_folders[-1])
    return ckpt_folder_path

    
def vis_log_path(folder_prefix='visualization'):

    last_run_folder = get_checkpoint_path()
    vis_output_folder = os.path.join(last_run_folder, folder_prefix)

    os.makedirs(vis_output_folder, exist_ok=True)

    return vis_output_folder

