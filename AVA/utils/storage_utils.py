from google.cloud import storage
import os
from typing import List, Optional
import logging

class CloudStorageManager:
    def __init__(self, bucket_name: str = "ava-vllm-data"):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
    def upload_file(self, source_file: str, destination_blob: str) -> bool:
        """Upload a file to Google Cloud Storage."""
        try:
            blob = self.bucket.blob(destination_blob)
            blob.upload_from_filename(source_file)
            logging.info(f"File {source_file} uploaded to {destination_blob}")
            return True
        except Exception as e:
            logging.error(f"Failed to upload {source_file}: {str(e)}")
            return False
    
    def upload_directory(self, source_dir: str, destination_prefix: str = "") -> List[str]:
        """Upload an entire directory to Google Cloud Storage."""
        uploaded_files = []
        
        for root, _, files in os.walk(source_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, source_dir)
                blob_path = os.path.join(destination_prefix, relative_path)
                
                if self.upload_file(local_path, blob_path):
                    uploaded_files.append(blob_path)
        
        return uploaded_files
    
    def download_file(self, source_blob: str, destination_file: str) -> bool:
        """Download a file from Google Cloud Storage."""
        try:
            blob = self.bucket.blob(source_blob)
            os.makedirs(os.path.dirname(destination_file), exist_ok=True)
            blob.download_to_filename(destination_file)
            logging.info(f"Downloaded {source_blob} to {destination_file}")
            return True
        except Exception as e:
            logging.error(f"Failed to download {source_blob}: {str(e)}")
            return False
    
    def list_files(self, prefix: Optional[str] = None) -> List[str]:
        """List all files in the bucket with optional prefix."""
        blobs = self.client.list_blobs(self.bucket, prefix=prefix)
        return [blob.name for blob in blobs]