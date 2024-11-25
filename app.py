import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

# Set the page title
st.title("AI-Powered Redirect Mapping Tool - Version 2.0")

st.markdown("""
⚡ **What It Is**  
This tool automates redirect mappings during site migrations by matching URLs from an old site to a new site based on content similarity and custom fallback rules for unmatched URLs.

⚡ **How to Use It:**  
1. Upload `origin.csv` and `destination.csv` files. Ensure that your file contains the URL, Title, Meta Description, and H1.
2. Click **"Let's Go!"** to initiate the matching process.
3. Download the resulting `output.csv` file containing matched URLs with similarity scores or fallback rules.
""")

# Step 1: Upload Files
st.header("Upload Your Files")
uploaded_origin = st.file_uploader("Upload origin.csv", type="csv")
uploaded_destination = st.file_uploader("Upload destination.csv", type="csv")

if uploaded_origin and uploaded_destination:
    st.success("Files uploaded successfully!")
    
    # Step 2: Load Data with Encoding Handling
    try:
        origin_df = pd.read_csv(uploaded_origin, encoding="ISO-8859-1")
        destination_df = pd.read_csv(uploaded_destination, encoding="ISO-8859-1")
    except UnicodeDecodeError:
        st.error("Error reading CSV files. Please ensure they are saved in a supported encoding (UTF-8 or ISO-8859-1).")
        st.stop()

    # Combine all columns for similarity matching
    origin_df['combined_text'] = origin_df.fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1)
    destination_df['combined_text'] = destination_df.fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1)

    # Step 3: Button to Process Matching
    if st.button("Let's Go!"):
        st.info("Processing data... This may take a while.")

        # Use a pre-trained model for embedding
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Vectorize the combined text
        origin_embeddings = model.encode(origin_df['combined_text'].tolist(), show_progress_bar=True)
        destination_embeddings = model.encode(destination_df['combined_text'].tolist(), show_progress_bar=True)

        # Create a FAISS index
        dimension = origin_embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(destination_embeddings.astype('float32'))

        # Perform the search for the nearest neighbors
        D, I = faiss_index.search(origin_embeddings.astype('float32'), k=1)

        # Calculate similarity scores
        similarity_scores = 1 - (D / np.max(D))

        # Create the output DataFrame with similarity scores
        matches_df = pd.DataFrame({
            'origin_url': origin_df.iloc[:, 0],
            'matched_url': destination_df.iloc[:, 0].iloc[I.flatten()].values,
            'similarity_score': np.round(similarity_scores.flatten(), 4),
            'fallback_applied': ['No'] * len(origin_df)  # Default to 'No' for fallback
        })

        # Step 4: Apply Fallbacks for Low Scores
        fallback_threshold = 0.6
        destination_urls = destination_df['combined_text'].tolist()  # Convert destination URLs to a list for fallback function
        for idx, score in enumerate(matches_df['similarity_score']):
            if isinstance(score, float) and score < fallback_threshold:
                origin_url = matches_df.at[idx, 'origin_url']
                fallback_url = apply_fallback_rule(origin_url, destination_urls)  # Function to determine fallback based on URL category
                matches_df.at[idx, 'matched_url'] = fallback_url
                matches_df.at[idx, 'similarity_score'] = 'Fallback'
                matches_df.at[idx, 'fallback_applied'] = 'Yes'

        # Step 5: Display and Download Results
        st.success("Matching complete! Download your results below.")
        st.write(matches_df)

        st.download_button(
            label="Download Results as CSV",
            data=matches_df.to_csv(index=False),
            file_name="redirect_mapping_output_v2.csv",
            mime="text/csv",
        )

def apply_fallback_rule(origin_url, destination_urls):
    # Implement fallback logic here based on provided categories
    if "about" in origin_url:
        for dest_url in destination_urls:
            if "about-us" in dest_url:
                return dest_url
    elif "agent" in origin_url or "staff" in origin_url:
        for dest_url in destination_urls:
            if "team" in dest_url:
                return dest_url
    elif "properties" in origin_url:
        for dest_url in destination_urls:
            if "properties" in dest_url:
                return dest_url
    elif "blog" in origin_url:
        for dest_url in destination_urls:
            if "blog" in dest_url:
                return dest_url
    # Add more fallback rules based on the PDF logic provided
    return "/"  # Default to homepage if no match is found
