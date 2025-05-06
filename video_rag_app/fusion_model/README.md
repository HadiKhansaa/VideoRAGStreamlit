# Video RAG Fusion Model

This directory contains the saved components needed for the Video RAG streamlit application.

## Files
- `transcript_segments.pkl`: Transcript segments with timestamps
- `frames_data.pkl`: Video frame data with timestamps
- `text_faiss_index.bin`: FAISS index for text embeddings
- `image_faiss_index.bin`: FAISS index for image embeddings
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer
- `tfidf_matrix.pkl`: TF-IDF matrix
- `bm25_model.pkl`: BM25 model
- `video_path.txt`: Path to the video file

## Usage
1. Download this entire folder
2. Run the streamlit app with `streamlit run app.py`
