import requests
import logging
from typing import List, Optional, Dict, Any
import json
import google.generativeai as genai

class AVAModel:
    def __init__(self, config):
        self.config = config
        self.conversation_history = []
        self.max_history_length = 10
        
        if not self.config.api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set")
            
        # Initialize Gemini with optimized settings for the faster model
        try:
            genai.configure(api_key=self.config.api_key)
            
            # Fix model name format - remove 'models/' prefix if present for GenerativeModel constructor
            model_name = self.config.model_name
            if model_name.startswith("models/"):
                model_name = model_name[7:]  # Remove "models/" prefix for the GenerativeModel constructor
            
            # Configure generation parameters optimized for speed with Gemini 2.0 Flash
            self.generation_config = {
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_output_tokens,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                # Additional parameters specific to Flash models to improve speed
                "response_mime_type": "text/plain",  # Use plain text for faster responses
                "candidate_count": 1,                # Request only one response for speed
            }
                
            self.model = genai.GenerativeModel(model_name)
            self.chat_session = self.model.start_chat(history=[])
            logging.info(f"Successfully initialized Gemini model {model_name}")
            self.use_gemini = True
        except Exception as e:
            logging.error(f"Failed to initialize Gemini: {str(e)}")
            self.model = None
            self.chat_session = None
            self.use_gemini = False
        
    def _format_conversation(self, message: str, role: str = "human") -> str:
        """Format a message for the conversation history"""
        prefix = "Human:" if role == "human" else "Assistant:"
        return f"{prefix} {message}"
    
    def _get_conversation_context(self) -> str:
        """Get the formatted conversation history"""
        history = self.conversation_history[-self.max_history_length:]
        return "\n".join(history)
    
    def generate_response(self, user_message: str, document_data: Dict = None) -> str:
        """Generate a response using Gemini API if enabled, else fallback."""
        try:
            # Special handling for PDF documents processed with Gemini
            if self.use_gemini and document_data and document_data.get("format") == "pdf" and "gemini_matches" in document_data:
                # Compose a prompt using Gemini's extracted matches
                matches = document_data.get("gemini_matches", [])
                context = f"You are an advanced AI assistant analyzing a PDF document. Gemini extracted {len(matches)} key snippets from the PDF.\n\n"
                
                # Include only the most relevant content
                for i, m in enumerate(matches[:5]):
                    if 'text' in m:
                        context += f"Content {i+1} (Page {m.get('page', 'unknown')}): {m.get('text', '')[:200]}\n\n"
                    elif 'table_data' in m and m.get('table_data'):
                        context += f"Table {i+1} (Page {m.get('page', 'unknown')}):\n"
                        for row in m.get('table_data', [])[:3]:
                            context += " | ".join([str(cell) for cell in row]) + "\n"
                        context += "\n"
                    elif 'image_caption' in m and m.get('image_caption'):
                        context += f"Image {i+1} (Page {m.get('page', 'unknown')}): {m.get('image_caption', '')}\n\n"
                
                user_query = user_message or 'Please summarize and interpret the main points of this PDF.'
                
                prompt = (
                    f"{context}\n"
                    f"User's question: {user_query}\n\n"
                    "Provide a concise, professional answer based solely on the content above."
                )
                
                # Use Gemini 2.0 Flash for faster document analysis
                try:
                    response = self.model.generate_content(
                        prompt,
                        generation_config=self.generation_config
                    )
                    text_response = self._clean_response(response.text)
                    return text_response
                except Exception as e:
                    logging.error(f"Gemini API PDF analysis failed: {str(e)}")
                    return self._get_default_response(user_message)
            
            # If document data is provided but not PDF, format it for analysis
            if document_data:
                formatted_message = self._format_document_analysis(user_message, document_data)
            else:
                formatted_message = user_message
            
            # Add the user message to conversation history
            self.conversation_history.append(self._format_conversation(formatted_message))
            
            # Use chat history for context-aware responses if available
            if self.use_gemini and self.model:
                try:
                    # System instruction optimized for faster responses
                    system_instruction = (
                        "You are an AI assistant that provides concise responses. "
                        "Analyze content efficiently and provide direct answers."
                    )
                    
                    # If we have a chat session, use it for better context
                    if self.chat_session:
                        response = self.chat_session.send_message(
                            formatted_message,
                            generation_config=self.generation_config
                        )
                    else:
                        # Fall back to direct generation if chat isn't working
                        content = [system_instruction, formatted_message]
                        response = self.model.generate_content(
                            content, 
                            generation_config=self.generation_config
                        )
                    
                    # Extract and clean the response text
                    text_response = self._clean_response(response.text)
                    
                    # Add response to history
                    formatted_response = self._format_conversation(text_response, role="assistant")
                    self.conversation_history.append(formatted_response)
                    
                    return text_response
                    
                except Exception as e:
                    logging.error(f"Gemini API generation failed: {str(e)}")
                    return self._get_default_response(user_message)
            
            # Fall back to RESTful API if model initialization failed
            try:
                # Prepare request data optimized for Flash model
                data = {
                    "contents": [
                        {
                            "parts": [
                                {"text": "You are a helpful, concise assistant."},
                                {"text": formatted_message}
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": self.config.temperature,
                        "maxOutputTokens": self.config.max_output_tokens,
                        "topP": self.config.top_p,
                        "topK": self.config.top_k,
                        "candidateCount": 1,  # Just one response for speed
                        "responseMimeType": "text/plain"  # Plain text is faster
                    }
                }
                
                # Make API request to Flash model
                headers = {"Content-Type": "application/json"}
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.model_name}:generateContent?key={self.config.api_key}"
                response = requests.post(url, headers=headers, json=data)
                
                if response.status_code != 200:
                    logging.error(f"API request failed: {response.status_code} - {response.text}")
                    return self._get_default_response(user_message)
                    
                # Parse response
                response_data = response.json()
                if not response_data.get("candidates"):
                    logging.error("No response candidates found")
                    return self._get_default_response(user_message)
                    
                text_response = response_data["candidates"][0]["content"]["parts"][0]["text"]
                
                # Clean up response
                text_response = self._clean_response(text_response)
                
                # Add response to history
                formatted_response = self._format_conversation(text_response, role="assistant")
                self.conversation_history.append(formatted_response)
                
                return text_response
                
            except Exception as e:
                logging.error(f"REST API request failed: {str(e)}")
                return self._get_default_response(user_message)
                
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while processing your message. Could you please try again?"
    
    def _format_document_analysis(self, user_message: str, document_data: Dict) -> str:
        """Format document data and user message for analysis"""
        context_parts = []
        
        # Add document type and format
        doc_type = document_data.get('type', 'unknown').title()
        doc_format = document_data.get('format', 'unknown').upper()
        context_parts.extend([
            "You are analyzing a document. Here are the key details:",
            f"\nDocument Type: {doc_type}",
            f"Format: {doc_format}"
        ])

        # --- IMAGE HANDLING ---
        if doc_type.lower() == "image":
            metadata = document_data.get('metadata', {})
            dimensions = metadata.get('dimensions', {})
            context_parts.append(f"\nImage Dimensions: {dimensions.get('width', 0)}x{dimensions.get('height', 0)}")
            
            # Add extracted text from OCR if available
            ocr_text = document_data.get('extracted_text', '')
            if ocr_text and ocr_text.strip():
                context_parts.append(f"\nText Detected in Image (OCR):\n{ocr_text.strip()}")
            
            # Add Vision API analysis if available
            vision_analysis = document_data.get('vision_analysis', {})
            if vision_analysis:
                # Add labels
                if 'labels' in vision_analysis and vision_analysis['labels']:
                    labels = ', '.join(vision_analysis['labels'])
                    context_parts.append(f"\nImage Labels: {labels}")
                
                # Add detected objects
                if 'objects' in vision_analysis and vision_analysis['objects']:
                    objects = ', '.join(vision_analysis['objects'])
                    context_parts.append(f"\nDetected Objects: {objects}")
                
                # Add detected text (from Vision API)
                if 'detected_text' in vision_analysis and vision_analysis['detected_text']:
                    vision_text = vision_analysis['detected_text']
                    if vision_text != ocr_text:  # Avoid duplication with OCR text
                        context_parts.append(f"\nText From Vision API:\n{vision_text}")
                
                # Add face information if available
                if 'faces' in vision_analysis and vision_analysis['faces']:
                    faces = vision_analysis['faces']
                    context_parts.append(f"\nDetected Faces: {faces}")
                    
                    # Add face details if available
                    if 'face_details' in vision_analysis and vision_analysis['face_details']:
                        context_parts.append("\nFace Details:")
                        for i, face in enumerate(vision_analysis['face_details']):
                            emotions = []
                            for emotion, level in face.items():
                                if level not in ['VERY_UNLIKELY', 'UNLIKELY']:
                                    emotions.append(f"{emotion}: {level}")
                            if emotions:
                                context_parts.append(f"Face {i+1}: {', '.join(emotions)}")
                
                # Add landmarks if available
                if 'landmarks' in vision_analysis and vision_analysis['landmarks']:
                    landmarks = ', '.join(vision_analysis['landmarks'])
                    context_parts.append(f"\nLandmarks: {landmarks}")
                
                # Add dominant colors if available
                if 'dominant_colors' in vision_analysis and vision_analysis['dominant_colors']:
                    colors = []
                    for color in vision_analysis['dominant_colors'][:3]:
                        r, g, b = color.get('r', 0), color.get('g', 0), color.get('b', 0)
                        score = color.get('score', 0)
                        colors.append(f"RGB({r},{g},{b}) - score: {score:.2f}")
                    context_parts.append(f"\nDominant Colors: {', '.join(colors)}")
            
            # Add OpenCV analysis if available
            opencv_analysis = document_data.get('opencv_analysis', {})
            if opencv_analysis and 'error' not in opencv_analysis:
                blurriness = opencv_analysis.get('blurriness', 0)
                edge_percentage = opencv_analysis.get('edge_percentage', 0)
                faces_detected = opencv_analysis.get('faces_detected', 0)
                
                context_parts.append(f"\nImage Quality Metrics:")
                context_parts.append(f"- Blurriness: {blurriness:.2f} (lower is sharper)")
                context_parts.append(f"- Edge Percentage: {edge_percentage:.2f} (higher means more detailed)")
                if faces_detected:
                    context_parts.append(f"- Faces Detected by OpenCV: {faces_detected}")
        # --- END IMAGE HANDLING ---

        # --- VIDEO HANDLING ---
        if doc_type.lower() == "video":
            metadata = document_data.get('metadata', {})
            context_parts.append(f"\nVideo Duration: {metadata.get('duration', 0):.1f} seconds")
            context_parts.append(f"Frame Rate: {metadata.get('fps', 0)} fps")
            context_parts.append(f"Resolution: {metadata.get('dimensions', {}).get('width', 0)}x{metadata.get('dimensions', {}).get('height', 0)}")
            
            # Add audio information if available
            audio_analysis = document_data.get('audio_analysis', {})
            if audio_analysis:
                has_audio = audio_analysis.get('has_audio', False)
                context_parts.append(f"Has Audio: {has_audio}")
                if has_audio and 'duration' in audio_analysis:
                    context_parts.append(f"Audio Duration: {audio_analysis['duration']:.1f} seconds")
            
            # Add labels from cloud analysis
            cloud = document_data.get('cloud_analysis', {})
            if 'labels' in cloud and cloud['labels']:
                labels = ', '.join(label['description'] for label in cloud['labels'][:10] if 'description' in label)
                if labels:
                    context_parts.append(f"\nDetected Video Labels: {labels}")
            
            # Add speech transcription
            if 'speech_transcription' in cloud and cloud['speech_transcription']:
                trans = [t['transcript'] for t in cloud['speech_transcription'] if t.get('transcript')]
                if trans:
                    context_parts.append("\nSpeech Transcription (what is said in the video):")
                    context_parts.append('\n'.join(trans[:5]))  # Include up to 5 transcriptions
            
            # Add text annotations
            if 'text_annotations' in cloud and cloud['text_annotations']:
                texts = [t['text'] for t in cloud['text_annotations'] if t.get('text')]
                if texts:
                    context_parts.append("\nText Detected in Video Frames:")
                    context_parts.append('\n'.join(texts[:5]))  # Include up to 5 annotations
            
            # Add scene analysis
            if 'scene_analysis' in document_data:
                scenes = document_data['scene_analysis']
                if scenes:
                    context_parts.append("\nScene Changes Detected:")
                    for scene in scenes[:5]:  # Include up to 5 scenes
                        context_parts.append(f"Scene {scene['scene_number']} ({scene['start_time']:.1f}s - {scene['end_time']:.1f}s)")
            
            # Add OCR text from frames
            if 'extracted_texts' in document_data and document_data['extracted_texts']:
                texts = document_data['extracted_texts']
                if texts:
                    context_parts.append("\nText Detected by OCR in Video Frames:")
                    for entry in texts[:5]:  # Include up to 5 entries
                        context_parts.append(f"[{entry.get('timestamp', 0):.1f}s]: {entry.get('text', '')[:150]}")
            
            # Add information from keyframes
            keyframes = document_data.get('keyframes', [])
            if keyframes:
                # Include vision analysis from keyframes
                for i, kf in enumerate(keyframes[:3]):  # Include up to 3 keyframes
                    if 'vision_analysis' in kf and kf['vision_analysis']:
                        va = kf['vision_analysis']
                        context_parts.append(f"\nKeyframe {i+1} at {kf.get('timestamp', 0):.1f}s:")
                        
                        # Add labels
                        if 'labels' in va and va['labels']:
                            context_parts.append(f"Labels: {', '.join(va['labels'][:5])}")
                        
                        # Add objects
                        if 'objects' in va and va['objects']:
                            context_parts.append(f"Objects: {', '.join(va['objects'][:5])}")
                        
                        # Add text
                        if 'text' in va and va['text']:
                            context_parts.append(f"Text: {va['text'][:150]}")
        # --- END VIDEO HANDLING ---

        # Add document content and structure for text documents
        if doc_type.lower() not in ["image", "video"] and 'extracted_text' in document_data:
            text = document_data['extracted_text']
            if text and text.strip():
                # Limit text length to avoid context overflow
                max_text_length = 5000
                text_preview = text[:max_text_length]
                if len(text) > max_text_length:
                    text_preview += "\n[Text truncated due to length. Full document is longer.]"
                context_parts.append(f"\nDocument Content:\n{text_preview}")
        
        # Add document statistics and analysis
        if 'analysis' in document_data:
            analysis = document_data['analysis']
            stats = analysis.get('statistics', {})
            if stats:
                context_parts.extend([
                    "\nDocument Statistics:",
                    f"- Word Count: {stats.get('word_count', 0)}",
                    f"- Sentence Count: {stats.get('sentence_count', 0)}",
                    f"- Paragraph Count: {stats.get('paragraph_count', 0)}",
                    f"- Average Sentence Length: {stats.get('average_sentence_length', 0):.1f} words"
                ])
            
            if 'key_topics' in analysis and analysis['key_topics']:
                topics = ', '.join(analysis['key_topics'][:10])  # Show up to 10 topics
                context_parts.append(f"\nKey Topics Identified: {topics}")
                
            if 'key_phrases' in analysis and analysis['key_phrases']:
                phrases = ', '.join(analysis['key_phrases'][:10])  # Show up to 10 phrases
                context_parts.append(f"\nKey Phrases: {phrases}")
                
            if 'named_entities' in analysis and analysis['named_entities']:
                entities = [f"{e['text']} ({e['type']})" for e in analysis['named_entities'][:10]]
                if entities:
                    context_parts.append(f"\nNamed Entities: {', '.join(entities)}")

        # Add document structure information
        if doc_type.lower() not in ["image", "video"] and 'document_structure' in document_data:
            structure = document_data['document_structure']
            
            # Add headings/sections
            if 'potential_headings' in structure and structure['potential_headings']:
                headings = [h['text'] for h in structure['potential_headings'][:7]]  # Show up to 7 headings
                context_parts.append("\nDocument Sections/Headings:")
                context_parts.extend([f"- {heading}" for heading in headings])
            
            # Add information about lists
            if 'lists' in structure and structure['lists']:
                context_parts.append(f"\nDocument contains {len(structure['lists'])} lists")
                
            # Add table of contents info
            if structure.get('has_table_of_contents', False):
                context_parts.append("\nDocument contains a table of contents")

        # Add specific analysis instructions based on user's prompt
        context_parts.extend([
            "\nBased on the above content, please:",
            "1. Focus specifically on answering the user's question or addressing their request",
            "2. Provide relevant details from the analysis to support your response",
            "3. For images and videos, describe visual content clearly and concisely",
            "4. For text documents, cite relevant text portions if applicable",
            "5. If the user's request is unclear, analyze the most important aspects of the content"
        ])

        # Add user's specific request
        context_parts.extend([
            "\nUser's Question/Request:",
            user_message or "Please analyze this content and provide key insights."
        ])

        # Join all parts, ensuring we don't exceed token limits (rough estimate)
        full_context = "\n".join(context_parts)
        if len(full_context) > 12000:  # Very rough token estimation
            # Truncate by removing middle portions of the content
            parts = context_parts[:5]  # Keep initial description
            
            if doc_type.lower() == "image":
                # For images, prioritize vision analysis
                for part in context_parts:
                    if "Image Labels:" in part or "Detected Objects:" in part or "Text Detected" in part:
                        parts.append(part)
            elif doc_type.lower() == "video":
                # For videos, prioritize speech and text transcriptions
                for part in context_parts:
                    if "Speech Transcription" in part or "Text Detected" in part:
                        parts.append(part)
            else:
                # For text documents, prioritize topics and key phrases
                for part in context_parts:
                    if "Key Topics" in part or "Key Phrases" in part:
                        parts.append(part)
            
            # Add back the instructions and user request
            parts.extend(context_parts[-4:])
            return "\n".join(parts)
            
        return full_context
    
    def _get_default_response(self, user_message: str) -> str:
        """Generate a default response if the model fails"""
        if 'document' in user_message.lower() or 'analyze' in user_message.lower():
            return ("I apologize, but I couldn't properly analyze the document. " 
                   "This might be due to text extraction issues or document formatting. " 
                   "Could you try uploading the document again or specify what specific information you're looking for?")
        
        message_lower = user_message.lower()
        
        if any(word in message_lower for word in ['hi', 'hello', 'hey']):
            return "Hello! How can I help you today?"
        elif 'how are you' in message_lower:
            return "I'm doing well, thank you for asking! How can I assist you?"
        elif any(word in message_lower for word in ['bye', 'goodbye']):
            return "Goodbye! Have a great day!"
        elif '?' in user_message:
            return "That's an interesting question. Could you provide more details about what you'd like to know?"
        else:
            return "I understand. Please tell me more about how I can help you with that."
    
    def clear_conversation(self):
        """Clear the conversation history"""
        self.conversation_history = []
        if self.use_gemini and self.model:
            try:
                # Reset chat session
                self.chat_session = self.model.start_chat(history=[])
            except Exception as e:
                logging.error(f"Failed to reset chat session: {str(e)}")
    
    def _clean_response(self, response: str) -> str:
        """Clean up the response text by removing unwanted prefixes and formatting"""
        if not response:
            return ""
            
        # Remove common prefixes that Gemini might include
        common_prefixes = [
            "I'd be happy to help analyze",
            "I'll analyze",
            "Based on the provided",
            "Based on the information provided",
            "Based on the content provided",
            "According to the document",
            "From the document",
            "Assistant:",
            "AI:"
        ]
        
        cleaned = response.strip()
        
        for prefix in common_prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                # Remove prefix and any following punctuation and whitespace
                cleaned = cleaned[len(prefix):].strip()
                if cleaned and cleaned[0] in [',', ':', '.', '-']:
                    cleaned = cleaned[1:].strip()
        
        # Remove markdown-style formatting that might be present
        if cleaned.startswith("```") and "```" in cleaned[3:]:
            # Extract content from code blocks
            parts = cleaned.split("```", 2)
            if len(parts) >= 3:
                language_hint = parts[1].strip()
                if not language_hint or language_hint in ['markdown', 'text', 'html', 'css', 'js', 'javascript', 'python']:
                    # It's likely a code block, so extract the content
                    cleaned = parts[2].split("```")[0].strip()
        
        return cleaned