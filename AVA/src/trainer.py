import os
import logging
import vertexai
from vertexai.language_models import TextGenerationModel
from config.config import ModelConfig

class VertexAITrainer:
    def __init__(self, config: ModelConfig):
        self.config = config
        # Ensure credentials are set
        creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if not creds_path or not os.path.exists(creds_path):
            raise RuntimeError(f"Google Cloud credentials not found at {creds_path}")
        try:
            vertexai.init(
                project=config.project_id,
                location=config.region
            )
            self.model = TextGenerationModel.from_pretrained(config.model_name)
            logging.info("Successfully initialized PaLM model")
        except Exception as e:
            logging.error(f"Failed to initialize PaLM model: {str(e)}")
            raise RuntimeError(f"PaLM model initialization failed: {str(e)}")

    def predict(self, text: str) -> str:
        if not self.model:
            raise RuntimeError("Model not initialized")
        try:
            response = self.model.predict(
                text,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
                top_k=self.config.top_k,
                top_p=self.config.top_p
            )
            if hasattr(response, 'text'):
                return response.text
            elif isinstance(response, str):
                return response
            else:
                return str(response)
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")

    @staticmethod
    def test_vertexai_connection(config: ModelConfig):
        """Test connection to Vertex AI and model prediction outside FastAPI."""
        try:
            vertexai.init(project=config.project_id, location=config.region)
            model = TextGenerationModel.from_pretrained(config.model_name)
            response = model.predict("Hello, are you working?", temperature=0.7, max_output_tokens=32)
            print("Vertex AI test response:", getattr(response, 'text', response))
        except Exception as e:
            print("Vertex AI test failed:", str(e))
            raise