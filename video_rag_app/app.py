import streamlit as st
import pickle
import faiss
import numpy as np
import os
import time
from datetime import timedelta
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Video RAG System",
    page_icon="🎬",
    layout="wide"
)

# App title and description
st.title("Video RAG System")
st.markdown("""
This application allows you to ask questions about a video and retrieves the most relevant segments.
The system uses a multimodal fusion approach combining text and visual information.
""")

# Sidebar information
with st.sidebar:
    st.header("About")
    st.info("This Video RAG system uses multimodal fusion to retrieve relevant video segments based on your questions.")
    st.markdown("---")
    st.subheader("Retrieval Methods")
    st.markdown("""
    - **Semantic Text**: Retrieves transcript segments using semantic search
    - **Semantic Image**: Retrieves relevant video frames 
    - **Multimodal Fusion**: Combines text and image similarity
    """)
    st.markdown("---")
    
    # Show confidence threshold slider
    st.subheader("Settings")
    threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5,
        help="Set the confidence threshold for determining if an answer is present in the video"
    )
    
    fusion_alpha = st.slider(
        "Text-Image Balance (alpha)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5,
        help="Balance between text (higher values) and image (lower values) in fusion"
    )

# Function to format timestamp
def format_timestamp(seconds):
    """Format seconds to HH:MM:SS."""
    return str(timedelta(seconds=round(seconds)))

# Function to convert HH:MM:SS to seconds
def timestamp_to_seconds(timestamp_str):
    """Convert HH:MM:SS to seconds."""
    h, m, s = map(int, timestamp_str.split(':'))
    return h * 3600 + m * 60 + s

# Function to load models and data
@st.cache_resource
def load_models_and_data():
    """Load all necessary components for the fusion model."""
    
    data = {}
    
    # Check if model files exist
    if not os.path.exists("fusion_model/transcript_segments.pkl"):
        st.error("Model files not found. Please place the app.py file in the same directory as the fusion_model folder.")
        st.stop()
    
    # Load transcript segments
    with open('fusion_model/transcript_segments.pkl', 'rb') as f:
        data["transcript_segments"] = pickle.load(f)
    
    # Load frames data
    with open('fusion_model/frames_data.pkl', 'rb') as f:
        data["frames_data"] = pickle.load(f)
    
    # Load text FAISS index
    data["text_faiss_index"] = faiss.read_index('fusion_model/text_faiss_index.bin')
    
    # Load image FAISS index
    data["image_faiss_index"] = faiss.read_index('fusion_model/image_faiss_index.bin')
    
    # Load TF-IDF vectorizer and matrix
    with open('fusion_model/tfidf_vectorizer.pkl', 'rb') as f:
        data["tfidf_vectorizer"] = pickle.load(f)
    
    with open('fusion_model/tfidf_matrix.pkl', 'rb') as f:
        data["tfidf_matrix"] = pickle.load(f)
    
    # Load BM25 model
    with open('fusion_model/bm25_model.pkl', 'rb') as f:
        data["bm25_model"] = pickle.load(f)
    
    # Load video path
    with open('fusion_model/video_path.txt', 'r') as f:
        data["video_path"] = f.read().strip()
    
    # Load models
    data["text_model"] = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data["clip_model"] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    data["clip_processor"] = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    data["device"] = device
    
    return data

# Function to perform semantic text retrieval
def semantic_text_retrieval(query, top_k=5, data=None):
    """Retrieve most relevant transcript segments using semantic search."""
    
    # Encode query
    query_embedding = data["text_model"].encode([query])[0]
    
    # Normalize embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    
    # Search
    distances, indices = data["text_faiss_index"].search(query_embedding, top_k)
    
    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(data["transcript_segments"]):
            segment = data["transcript_segments"][idx]
            results.append({
                "segment_id": idx,
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "timestamp": segment["timestamp"],
                "text": segment["text"],
                "score": float(dist),
                "rank": i + 1
            })
    
    return results

# Function to perform semantic image retrieval
def semantic_image_retrieval(query, top_k=5, data=None):
    """Retrieve most relevant video frames using semantic search."""
    
    # Initialize CLIP
    model = data["clip_model"]
    processor = data["clip_processor"]
    device = data["device"]
    
    # Process text
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    
    # Generate embedding
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    
    # Normalize embedding
    query_embedding = text_features.cpu().numpy()[0]
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    query_embedding = query_embedding.reshape(1, -1).astype('float32')
    
    # Search
    distances, indices = data["image_faiss_index"].search(query_embedding, top_k)
    
    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(data["frames_data"]):
            frame = data["frames_data"][idx]
            results.append({
                "frame_id": frame["frame_id"],
                "timestamp": frame["timestamp"],
                "timestamp_formatted": frame["timestamp_formatted"],
                "frame_path": frame["frame_path"],
                "score": float(dist),
                "rank": i + 1
            })
    
    return results

# Function to perform multimodal fusion retrieval
def multimodal_fusion_retrieval(query, top_k=5, alpha=0.5, data=None):
    """Combine text and image retrieval scores for multimodal retrieval."""
    
    # Get text and image results
    text_results = semantic_text_retrieval(query, top_k=top_k*2, data=data)
    image_results = semantic_image_retrieval(query, top_k=top_k*2, data=data)
    
    # Create a mapping of timestamp to results
    fusion_results = {}
    
    # Process text results
    for result in text_results:
        timestamp = result["start_time"]
        
        if timestamp not in fusion_results:
            fusion_results[timestamp] = {
                "timestamp": timestamp,
                "timestamp_formatted": format_timestamp(timestamp),
                "text": result["text"],
                "text_score": result["score"],
                "image_score": 0,
                "combined_score": alpha * result["score"]
            }
        else:
            fusion_results[timestamp]["text_score"] = result["score"]
            fusion_results[timestamp]["combined_score"] += alpha * result["score"]
    
    # Process image results - match to nearest text segment
    for result in image_results:
        img_timestamp = result["timestamp"]
        
        # Find closest text segment
        closest_timestamp = min(
            [seg["start_time"] for seg in data["transcript_segments"]],
            key=lambda x: abs(x - img_timestamp)
        )
        
        if closest_timestamp not in fusion_results:
            # Find the corresponding text segment
            closest_segment = next(seg for seg in data["transcript_segments"] if seg["start_time"] == closest_timestamp)
            
            fusion_results[closest_timestamp] = {
                "timestamp": closest_timestamp,
                "timestamp_formatted": format_timestamp(closest_timestamp),
                "text": closest_segment["text"],
                "text_score": 0,
                "image_score": result["score"],
                "combined_score": (1 - alpha) * result["score"]
            }
        else:
            fusion_results[closest_timestamp]["image_score"] = result["score"]
            fusion_results[closest_timestamp]["combined_score"] += (1 - alpha) * result["score"]
    
    # Sort by combined score and take top k
    sorted_results = sorted(
        fusion_results.values(),
        key=lambda x: x["combined_score"],
        reverse=True
    )[:top_k]
    
    # Add rank
    for i, result in enumerate(sorted_results):
        result["rank"] = i + 1
    
    return sorted_results

# Function to create HTML video player with timestamp support
def create_video_player(video_path, start_time=0):
    """Create HTML video player with time control."""
    # Ensure the video path is used correctly - account for relative paths
    if not os.path.isabs(video_path):
        # If the video_path is a relative path from the fusion_model directory
        if video_path.startswith('./') or video_path.startswith('../'):
            # Convert to absolute path based on current directory
            base_dir = os.path.dirname(os.path.abspath(__file__))
            video_path = os.path.normpath(os.path.join(base_dir, 'fusion_model', os.path.normpath(video_path)))
        else:
            # If just a filename, assume it's in the current directory
            video_path = os.path.abspath(video_path)
    
    # Check if the video file exists
    if not os.path.exists(video_path):
        st.warning(f"Video file not found at: {video_path}")
        st.info("Please ensure the video file is properly placed in your project directory.")
        return None
    
    # Create a unique key for the video player
    video_key = f"video_player_{int(time.time())}"
    
    # Use the time_start parameter to set the video's starting position
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    
    return video_bytes, start_time

# Load models and data
try:
    with st.spinner("Loading models and data..."):
        data = load_models_and_data()
    st.success("Models and data loaded successfully!")
except Exception as e:
    st.error(f"Error loading models and data: {e}")
    st.stop()

# Initialize session state for video player
if "current_video_time" not in st.session_state:
    st.session_state.current_video_time = 0

# Display the video player at the top of the app (before tabs)
video_container = st.container()
with video_container:
    video_path = data["video_path"]
    
    try:
        st.subheader("Video Player")
        video_data, start_time = create_video_player(video_path, st.session_state.current_video_time)
        
        if video_data:
            st.video(video_data, start_time=start_time)
            st.markdown(f"Current position: {format_timestamp(st.session_state.current_video_time)}")
        else:
            st.warning("Please ensure the video file path is correctly specified in video_path.txt")
    except Exception as e:
        st.error(f"Error loading video: {e}")
        st.info("Please check that the video file exists and is in a supported format (mp4, webm, ogg).")

# Create tabs for different views
tab1, tab2 = st.tabs(["Search", "Chat History"])

with tab1:
    # Chat container
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Display frames and timestamps if available
            if "results" in message and message["results"]:
                for result in message["results"]:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    # Display frame if path exists
                    if "frame_path" in result and os.path.exists(result["frame_path"]):
                        with col1:
                            st.image(result["frame_path"], caption=f"Frame at {result['timestamp_formatted']}")
                    
                    # Display text and timestamp
                    with col2:
                        st.markdown(f"**Time:** {result['timestamp_formatted']}")
                        st.markdown(f"**Text:** {result['text']}")
                        st.markdown(f"**Confidence:** {result['combined_score']:.4f}")
                    
                    # Add button to jump to this timestamp in the video
                    with col3:
                        if st.button(f"Play from {result['timestamp_formatted']}", key=f"play_{message['role']}_{result['timestamp']}"):
                            st.session_state.current_video_time = result['timestamp']
                            st.experimental_rerun()
                    
                    st.markdown("---")

    # Input for new query
    query = st.chat_input("Ask a question about the video...")
    
    if query:
        # Display user message
        with st.chat_message("user"):
            st.write(query)
        
        # Add message to history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Perform retrieval using fusion model
        with st.spinner("Searching for relevant video segments..."):
            fusion_results = multimodal_fusion_retrieval(
                query, 
                top_k=3, 
                alpha=fusion_alpha,
                data=data
            )
        
        # Prepare assistant response
        if fusion_results and fusion_results[0]["combined_score"] >= threshold:
            # Get frames for top results
            frames = []
            for result in fusion_results:
                # Find closest frame to the result timestamp
                closest_frame = min(
                    data["frames_data"],
                    key=lambda x: abs(x["timestamp"] - result["timestamp"])
                )
                result["frame_path"] = closest_frame["frame_path"]
                frames.append(result)
            
            response = f"I found relevant information in the video:"
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.write(response)
                
                # Display results
                for result in frames:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    # Display frame
                    if os.path.exists(result["frame_path"]):
                        with col1:
                            st.image(result["frame_path"], caption=f"Frame at {result['timestamp_formatted']}")
                    
                    # Display text and timestamp
                    with col2:
                        st.markdown(f"**Time:** {result['timestamp_formatted']}")
                        st.markdown(f"**Text:** {result['text']}")
                        st.markdown(f"**Confidence:** {result['combined_score']:.4f}")
                    
                    # Add button to jump to this timestamp in the video
                    with col3:
                        if st.button(f"Play from {result['timestamp_formatted']}", key=f"play_result_{result['timestamp']}"):
                            st.session_state.current_video_time = result['timestamp']
                            st.experimental_rerun()
                    
                    st.markdown("---")
                
                # Auto-play the first result
                if frames:
                    st.session_state.current_video_time = frames[0]['timestamp']
                    st.info(f"Video player automatically set to timestamp: {frames[0]['timestamp_formatted']}")
                    st.experimental_rerun()
            
            # Add to history with results
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "results": frames
            })
        else:
            response = "I couldn't find relevant information about that in the video. Please try another question."
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.write(response)
            
            # Add to history without results
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "results": []
            })

with tab2:
    # Display chat history in a table format
    if st.session_state.messages:
        st.subheader("Chat History")
        
        history_df = []
        for i in range(0, len(st.session_state.messages), 2):
            if i+1 < len(st.session_state.messages):
                entry = {
                    "Question": st.session_state.messages[i]["content"],
                    "Answer": st.session_state.messages[i+1]["content"],
                    "Results Found": "Yes" if "results" in st.session_state.messages[i+1] and st.session_state.messages[i+1]["results"] else "No"
                }
                history_df.append(entry)
        
        if history_df:
            st.table(pd.DataFrame(history_df))
    else:
        st.info("No chat history yet. Ask a question to get started!")

# Add a download button for sample questions
st.sidebar.markdown("---")
st.sidebar.subheader("Sample Questions")
sample_questions = """
- What is 15-puzzle?
- What is a reconfiguration step in 15-puzzle?
- What is the Graph Coloring Problem?
- What is the token jumping rule?
- What is token sliding?
- What is PSPACE?
- How can you parametrize token jumping and token sliding?
- What is the Lemma on FPT on {C3, C4} free graphs?
"""
st.sidebar.markdown(sample_questions)