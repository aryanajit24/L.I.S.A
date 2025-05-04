import os
import json
from typing import Dict, List, Any
import shutil
from .storage_utils import CloudStorageManager

class DataManager:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        self.processed_dir = os.path.join(base_dir, "processed")
        self.raw_dir = os.path.join(base_dir, "raw")
        self.storage_manager = CloudStorageManager()
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.raw_dir, exist_ok=True)
        
        # Create subdirectories for different data types
        for data_type in ["documents", "images", "videos"]:
            os.makedirs(os.path.join(self.raw_dir, data_type), exist_ok=True)
            os.makedirs(os.path.join(self.processed_dir, data_type), exist_ok=True)
    
    def prepare_training_data(self, metadata: Dict[str, Any]) -> str:
        """Prepare and organize training data with metadata."""
        # Create a training dataset directory
        dataset_dir = os.path.join(self.processed_dir, "training_dataset")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save metadata
        metadata_path = os.path.join(dataset_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return dataset_dir
    
    def upload_training_data(self, dataset_dir: str) -> List[str]:
        """Upload prepared training data to Cloud Storage."""
        return self.storage_manager.upload_directory(
            dataset_dir,
            destination_prefix="training_data"
        )
    
    def clean_processed_data(self):
        """Clean up processed data directory."""
        if os.path.exists(self.processed_dir):
            shutil.rmtree(self.processed_dir)
            os.makedirs(self.processed_dir)
    
    def get_training_files(self) -> Dict[str, List[str]]:
        """Get lists of training files by type."""
        training_files = {
            "documents": [],
            "images": [],
            "videos": []
        }
        
        for data_type in training_files.keys():
            type_dir = os.path.join(self.raw_dir, data_type)
            if os.path.exists(type_dir):
                training_files[data_type] = [
                    os.path.join(type_dir, f) 
                    for f in os.listdir(type_dir)
                    if not f.startswith(".")
                ]
        
        return training_files