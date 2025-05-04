import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import ModelConfig
from src.trainer import VertexAITrainer

config = ModelConfig()
print(config.region, config.model_name)
VertexAITrainer.test_vertexai_connection(config)