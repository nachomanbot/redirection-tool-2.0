import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials

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

    # Step 3: Connect to Google Sheets for Rules
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/spreadsheets",
             "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
    client = gspread.authorize(creds)

    # Load the rules sheet
    try:
        sheet = client.open_by_key("1xzm76zzYCDeFMZRejF1onxJVcph6s7wosBKT5mJfQAo").sheet1
        rules_df = pd.DataFrame(sheet.get_all_records())
        st.success("Rules loaded successfully from Google Sheets!")
    except Exception as e:
        st.error(f"Error loading rules from Google Sheets: {e}")
        st.stop()

    # Step 4: Button to Process Matching
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
        similarity_scores = 1 - (D / (np.max(D) + 1e-10))  # Add small value to avoid division by zero

        # Create the output DataFrame with similarity scores
        matches_df = pd.DataFrame({
            'origin_url': origin_df.iloc[:, 0],
            'matched_url': destination_df.iloc[:, 0].iloc[I.flatten()].apply(lambda x: x.split()[0]).values,  # Ensure only the URL is added
            'similarity_score': np.round(similarity_scores.flatten(), 4),
            'fallback_applied': ['No'] * len(origin_df)  # Default to 'No' for fallback
        })

        # Step 5: Apply Fallbacks for Low Scores
        fallback_threshold = 0.6
        destination_urls = destination_df['combined_text'].tolist()  # Convert destination URLs to a list for fallback function
        for idx, score in enumerate(matches_df['similarity_score']):
            if isinstance(score, (float, int)) and score < fallback_threshold:
                origin_url = matches_df.at[idx, 'origin_url']
                fallback_url = "/"  # Default fallback to homepage

                # Apply Google Sheet rules
                applicable_rules = rules_df.sort_values(by='Priority')  # Sort rules by priority
                for _, rule in applicable_rules.iterrows():
                    if rule['Keyword'] in origin_url:
                        fallback_url = rule['Destination URL Pattern']
                        break

                # Update the DataFrame with the fallback URL
                matches_df.at[idx, 'matched_url'] = fallback_url
                matches_df.at[idx, 'similarity_score'] = 'Fallback'
                matches_df.at[idx, 'fallback_applied'] = 'Yes'

        # Step 6: Display and Download Results
        st.success("Matching complete! Download your results below.")
        st.write(matches_df)

        st.download_button(
            label="Download Results as CSV",
            data=matches_df.to_csv(index=False),
            file_name="redirect_mapping_output_v2.csv",
            mime="text/csv",
        )
