# Week 4: Multimodal AI Applications

## 📚 Lecture Overview

This week we explore the exciting world of multimodal AI, where we work with text, images, audio, and video simultaneously. You'll learn how to build AI systems that can understand and generate content across multiple modalities.

## 🎯 Learning Objectives

By the end of this week, you will:
- Understand multimodal AI architectures and frameworks
- Work with text, image, audio, and video models
- Build cross-modal understanding systems
- Implement multimodal generation capabilities
- Create real-world multimodal applications

## 📖 Lecture Content

### 1. Introduction to Multimodal AI

**What is Multimodal AI?**
- AI systems that process multiple data types simultaneously
- Cross-modal understanding and generation
- Real-world applications: virtual assistants, content creation, healthcare

**Multimodal Tasks:**
- **Vision-Language**: Image captioning, visual question answering
- **Audio-Vision**: Lip reading, audio-visual speech recognition
- **Text-Audio**: Speech synthesis, audio transcription
- **Cross-modal Retrieval**: Finding related content across modalities

**Architecture Patterns:**
```
Input Modalities → Encoders → Fusion → Decoder → Output
     ↓              ↓         ↓        ↓
   Text           Text     Cross-    Text
   Image          Vision   Modal     Image
   Audio          Audio    Fusion    Audio
```

### 2. Vision-Language Models

**CLIP (Contrastive Language-Image Pre-training):**
```python
import torch
from transformers import CLIPProcessor, CLIPModel

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Process inputs
image = Image.open("cat.jpg")
text = ["a photo of a cat", "a photo of a dog"]

inputs = processor(
    text=text,
    images=image,
    return_tensors="pt",
    padding=True
)

# Get embeddings
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
```

**BLIP (Bootstrapping Language-Image Pre-training):**
```python
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Generate caption
image = Image.open("image.jpg")
inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
```

### 3. Audio-Visual Models

**Audio-Visual Speech Recognition:**
```python
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from PIL import Image
import cv2

class AudioVisualASR:
    def __init__(self):
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.audio_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        
    def process_audio(self, audio_path):
        """Process audio input"""
        audio, sr = librosa.load(audio_path, sr=16000)
        inputs = self.audio_processor(audio, sampling_rate=16000, return_tensors="pt")
        return inputs
    
    def process_video(self, video_path):
        """Extract visual features from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames
    
    def transcribe(self, audio_path, video_path=None):
        """Transcribe audio with optional visual context"""
        audio_inputs = self.process_audio(audio_path)
        
        if video_path:
            visual_features = self.process_video(video_path)
            # Combine audio and visual features
            # Implementation depends on specific model architecture
        
        with torch.no_grad():
            logits = self.audio_model(audio_inputs.input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.audio_processor.batch_decode(predicted_ids)
        
        return transcription[0]
```

### 4. Text-to-Speech and Speech-to-Text

**Text-to-Speech with TTS:**
```python
from TTS.api import TTS

# Initialize TTS
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

# Generate speech
tts.tts_to_file(
    text="Hello, this is a test of text-to-speech synthesis.",
    file_path="output.wav"
)
```

**Speech-to-Text with Whisper:**
```python
import whisper

# Load Whisper model
model = whisper.load_model("base")

# Transcribe audio
result = model.transcribe("audio.wav")
print(result["text"])
```

**Real-time Speech Processing:**
```python
import speech_recognition as sr
import pyttsx3

class SpeechAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
    
    def listen(self):
        """Listen for speech input"""
        with sr.Microphone() as source:
            print("Listening...")
            audio = self.recognizer.listen(source)
            try:
                text = self.recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                return "Could not understand audio"
    
    def speak(self, text):
        """Convert text to speech"""
        self.engine.say(text)
        self.engine.runAndWait()
    
    def conversation_loop(self):
        """Main conversation loop"""
        while True:
            user_input = self.listen()
            if user_input.lower() == "quit":
                break
            
            # Process with AI model
            response = self.process_with_ai(user_input)
            self.speak(response)
```

### 5. Multimodal Generation

**Text-to-Image Generation:**
```python
from diffusers import StableDiffusionPipeline
import torch

# Load Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

# Generate image
prompt = "A beautiful sunset over mountains, digital art"
image = pipe(prompt).images[0]
image.save("generated_image.png")
```

**Image-to-Text Generation:**
```python
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image

# Load model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Generate caption
image = Image.open("image.jpg")
pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values

output_ids = model.generate(pixel_values, max_length=50, num_beams=4)
caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

### 6. Cross-Modal Retrieval

**Multimodal Search System:**
```python
import torch
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np

class MultimodalSearch:
    def __init__(self):
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.image_encoder = SentenceTransformer('clip-ViT-B-32')
        
    def encode_text(self, texts):
        """Encode text to embeddings"""
        return self.text_encoder.encode(texts)
    
    def encode_image(self, image_path):
        """Encode image to embeddings"""
        image = Image.open(image_path)
        return self.image_encoder.encode([image])[0]
    
    def search(self, query, database, top_k=5):
        """Search for similar content"""
        if isinstance(query, str):
            query_embedding = self.encode_text([query])[0]
        else:
            query_embedding = self.encode_image(query)
        
        # Calculate similarities
        similarities = []
        for item in database:
            if item['type'] == 'text':
                item_embedding = self.encode_text([item['content']])[0]
            else:
                item_embedding = self.encode_image(item['content'])
            
            similarity = np.dot(query_embedding, item_embedding)
            similarities.append((similarity, item))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:top_k]
```

### 7. Building Multimodal Applications

**Virtual Assistant with Vision:**
```python
class MultimodalAssistant:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.7)
        self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
    def process_input(self, text=None, image=None, audio=None):
        """Process multimodal input"""
        context = {}
        
        if text:
            context['text'] = text
        
        if image:
            # Process image with CLIP
            inputs = self.image_processor(images=image, return_tensors="pt")
            image_features = self.image_model.get_image_features(**inputs)
            context['image_features'] = image_features
        
        if audio:
            # Process audio (simplified)
            context['audio'] = "Audio processed"
        
        return self.generate_response(context)
    
    def generate_response(self, context):
        """Generate multimodal response"""
        prompt = self.build_prompt(context)
        response = self.llm.predict(prompt)
        return response
    
    def build_prompt(self, context):
        """Build prompt from multimodal context"""
        prompt = "You are a helpful multimodal assistant. "
        
        if 'text' in context:
            prompt += f"User said: {context['text']} "
        
        if 'image_features' in context:
            prompt += "The user also shared an image. "
        
        if 'audio' in context:
            prompt += "The user also shared audio. "
        
        prompt += "Please provide a helpful response."
        return prompt
```

**Content Creation Pipeline:**
```python
class ContentCreator:
    def __init__(self):
        self.text_generator = pipeline("text-generation")
        self.image_generator = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
        
    def create_multimodal_content(self, topic):
        """Create text, image, and audio content"""
        # Generate text
        text = self.text_generator(f"Write about {topic}")[0]['generated_text']
        
        # Generate image
        image = self.image_generator(f"Create an image about {topic}").images[0]
        
        # Generate audio
        self.tts.tts_to_file(text=text[:200], file_path=f"{topic}_audio.wav")
        
        return {
            'text': text,
            'image': image,
            'audio_path': f"{topic}_audio.wav"
        }
```

## 🛠️ Workshop Preparation

**Pre-workshop Tasks:**
1. Install multimodal libraries: `pip install transformers diffusers TTS opencv-python`
2. Set up GPU environment for image generation
3. Prepare sample images, audio, and text data
4. Review CLIP and Stable Diffusion documentation

**Workshop Goals:**
- Build a multimodal search system
- Create a text-to-image generation pipeline
- Implement audio-visual processing
- Deploy a multimodal web application

## 📚 Additional Resources

**Reading Materials:**
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [BLIP Paper](https://arxiv.org/abs/2201.12086)
- [Stable Diffusion Paper](https://arxiv.org/abs/2112.10752)
- [Multimodal AI Survey](https://arxiv.org/abs/2306.02066)

**Videos:**
- [Multimodal AI Tutorial](https://www.youtube.com/watch?v=YQpV7WVTJXE)
- [CLIP Explained](https://www.youtube.com/watch?v=2xK74MqYvdg)
- [Stable Diffusion Guide](https://www.youtube.com/watch?v=1pedAIvTWXk)

**Tools & Platforms:**
- [Hugging Face Spaces](https://huggingface.co/spaces) - Deploy multimodal apps
- [Gradio](https://gradio.app/) - Create demos
- [Streamlit](https://streamlit.io/) - Build web apps

## 📝 Assignment

**Due Date**: End of Week 4

**Tasks:**
1. Build a multimodal AI application
2. Implement cross-modal understanding
3. Create a content generation pipeline
4. Deploy your application

**Submission Requirements:**
- Working multimodal application
- Documentation of architecture and features
- Demo or screenshots
- Performance analysis

## 🎯 Next Week Preview

**Week 5: API Development & Deployment**
- RESTful API design principles
- FastAPI and Flask frameworks
- Model serving and optimization
- Containerization with Docker

## 💡 Advanced Concepts

**Multimodal Fusion Strategies:**
- Early fusion vs late fusion
- Attention mechanisms
- Cross-modal transformers
- Hierarchical fusion

**Real-time Multimodal Processing:**
- Streaming architectures
- Latency optimization
- Edge computing considerations
- Real-time inference pipelines

## 🚀 Best Practices

**Data Management:**
- Efficient data loading and preprocessing
- Caching strategies
- Data versioning
- Quality assurance

**Performance Optimization:**
- Model quantization and compression
- Batch processing
- GPU utilization
- Memory management

**User Experience:**
- Intuitive multimodal interfaces
- Accessibility considerations
- Error handling and fallbacks
- Progressive enhancement

## 📞 Support

- **Element Support**: `#support`
- **Office Hours**: TBD
- **Email**: course-instructor@example.com

---

*Remember: Multimodal AI opens up incredible possibilities. Think creatively about how different modalities can work together!* 