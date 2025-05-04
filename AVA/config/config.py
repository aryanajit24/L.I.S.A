from dataclasses import dataclass

@dataclass
class ModelConfig:
    project_id: str = "dogwood-boulder-458622-t1"
    region: str = "us-central1"
    model_name: str = "models/gemini-2.0-flash"  # Updated to use faster Gemini 2.0 Flash model
    api_key: str = "AIzaSyBTDWNA7KFxsv6yDS3s9OVb8IEUmX00VE4"  # your provided Gemini API key
    api_url: str = "https://generativelanguage.googleapis.com/v1beta/models"
    temperature: float = 0.7
    max_output_tokens: int = 1024
    top_p: float = 0.8
    top_k: int = 40
    dataset_path: str = "data/processed"
    model_output_path: str = "models/trained"