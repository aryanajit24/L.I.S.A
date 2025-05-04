import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import Dataset, DataLoader
import json
import torch
from src.model import AVALLM
from config.config import ModelConfig
import logging
import numpy as np
from transformers import BertTokenizer
from typing import List, Dict

class ConversationDataset(Dataset):
    def __init__(self, conversations: List[Dict], tokenizer: BertTokenizer, max_length: int = 512):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        input_text = conversation['input']
        target_text = conversation['response']
        
        # Tokenize input and target
        inputs = self.tokenizer(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            target_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

def prepare_conversation_data():
    """Prepare sample conversation data for training"""
    conversations = [
        {
            "input": "[HUMAN] Hello! How are you?",
            "response": "Hello! I'm doing great, thank you for asking. How can I assist you today?"
        },
        {
            "input": "[HUMAN] What can you help me with?",
            "response": "I can help you with analyzing documents, images, and videos, as well as engage in natural conversations. I can extract text from visual content, understand context, and provide meaningful responses to your questions."
        },
        {
            "input": "[HUMAN] Tell me about yourself.",
            "response": "I am AVA, an AI assistant specializing in understanding both visual and textual information. I'm designed to help analyze various types of content and engage in meaningful conversations. I aim to be helpful, informative, and engaging while providing accurate information."
        },
        {
            "input": "[HUMAN] Can you analyze images?",
            "response": "Yes, I can analyze images in various formats including JPG, PNG, and BMP. I can extract text using OCR, understand the visual content, and provide detailed descriptions and analysis. Would you like to share an image with me?"
        }
    ]
    
    # Add more training examples for better coverage
    additional_conversations = [
        {
            "input": "[HUMAN] What file formats do you support?",
            "response": "I support various file formats including: Images (JPG, PNG, BMP), Documents (PDF, DOC, DOCX), and Videos (MP4, AVI, MOV). I can extract text, analyze content, and provide meaningful insights from these files."
        },
        {
            "input": "[HUMAN] How do you process images?",
            "response": "I process images using computer vision and OCR technology. I can extract text from images, analyze visual content, identify objects and patterns, and provide detailed descriptions. I handle this through my vision model component."
        },
        {
            "input": "[HUMAN] Can you help me understand a document?",
            "response": "Yes, I can help you understand documents. I can extract and analyze text content, identify key information, and provide summaries. I can process PDFs and Microsoft Word documents, helping you understand their content and context."
        }
    ]
    conversations.extend(additional_conversations)
    
    # Save conversations to file
    os.makedirs("data/processed/training", exist_ok=True)
    with open("data/processed/training/conversations.json", "w") as f:
        json.dump(conversations, f, indent=2)
    
    return conversations

def train_model(model: AVALLM, train_dataset: Dataset, config: ModelConfig):
    """Train the model on conversation data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    model.train()
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    # Setup optimizer with gradient clipping
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    try:
        for epoch in range(config.num_epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = model.language_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass with gradient clipping
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Log progress
                if batch_idx % 5 == 0:
                    print(f"Epoch {epoch+1}/{config.num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{config.num_epochs} completed, Average Loss: {avg_loss:.4f}")
            
            # Save checkpoint after each epoch
            checkpoint_dir = os.path.join(config.model_output_path, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        # Try to save the model even if training fails
        emergency_save_path = os.path.join(config.model_output_path, "emergency_save.pt")
        torch.save(model.state_dict(), emergency_save_path)
        print(f"Saved emergency model checkpoint to {emergency_save_path}")
        raise
    
    return model

def main():
    try:
        logging.basicConfig(level=logging.INFO)
        config = ModelConfig()
        
        # Reduce number of epochs for initial training
        config.num_epochs = 5
        config.batch_size = 2  # Smaller batch size for stability
        config.learning_rate = 1e-5  # Lower learning rate
        
        print("Initializing model...")
        model = AVALLM()
        
        print("Preparing training data...")
        conversations = prepare_conversation_data()
        
        print("Creating dataset...")
        dataset = ConversationDataset(conversations, model.tokenizer)
        
        print("Starting training...")
        trained_model = train_model(model, dataset, config)
        
        print("Saving trained model...")
        os.makedirs(config.model_output_path, exist_ok=True)
        model_path = os.path.join(config.model_output_path, "model.pt")
        trained_model.save(model_path)
        print(f"Model saved successfully to {model_path}")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()