import os
from dotenv import load_dotenv
load_dotenv()
from src.model import AVAModel
from config.config import ModelConfig

if __name__ == "__main__":
    try:
        config = ModelConfig()
        ava = AVAModel(config)
        test_message = "Hello, please introduce yourself."
        response = ava.generate_response(test_message)
        print("Test response:", response)
    except Exception as e:
        print("Error during test:", str(e))