import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import re

# Set the page title
st.title("AI-Powered Redirect Mapping Tool - Version 2.0")

st.markdown("""
⚡ **What It Is**  
This tool automates redirect mappings during site migrations by matching URLs from an old site to a new site based on content similarity and custom fallback rules for unmatched URLs.

⚡ **How to Use It:**  
1. Upload `origin.csv` and `destination.csv` files. Ensure that your files are correctly formatted.
2. The `rules.csv` will be automatically loaded from the backend.
3. Click **"Let's Go!"** to initiate the matching process.
4. Download the resulting `output.csv` file containing matched URLs with similarity scores or fallback rules.
""")

# Step 1: Upload Files
st.header("Upload Your Files")
uploaded_origin = st.file_uploader("Upload origin.csv", type="csv")
uploaded_destination = st.file_uploader("Upload destination.csv", type="csv")

# Load rules.csv from the backend
rules_path = 'rules.csv'  # Path to the rules CSV on the backend
us_cities_path = 'us_cities.csv'  # Path to the US cities CSV on the backend

if os.path.exists(rules_path):
    rules_df = pd.read_csv(rules_path, encoding="ISO-8859-1")
else:
    st.error("Rules file not found on the backend.")
    st.stop()

if os.path.exists(us_cities_path):
    us_cities_df = pd.read_csv(us_cities_path, encoding="ISO-8859-1")
    city_names = us_cities_df['CITY'].str.lower().str.strip().tolist()
else:
    st.error("US cities file not found on the backend.")
    st.stop()

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
        similarity_scores = 1 - (D / (np.max(D) + 1e-10))  # Add small value to avoid division by zero

        # Create the output DataFrame with similarity scores
        matches_df = pd.DataFrame({
            'origin_url': origin_df.iloc[:, 0],
            'matched_url': destination_df.iloc[:, 0].iloc[I.flatten()].apply(lambda x: x.split()[0]).values,  # Ensure only the URL is added
            'similarity_score': np.round(similarity_scores.flatten(), 4),
            'fallback_applied': ['No'] * len(origin_df)  # Default to 'No' for fallback
        })

        # Step 4: Apply Fallbacks for Low Scores
         0.65
        for idx, score in enumerate(matches_df['similarity_score']):
            if isinstance(score, (float, int)) and score < fallback_threshold:
                origin_url = matches_df.at[idx, 'origin_url']
                fallback_url = "/"  # Default fallback to homepage

                # Normalize the origin URL
                origin_url_normalized = origin_url.lower().strip().rstrip('/')

                # Apply CSV rules
                applicable_rules = rules_df.sort_values(by='Priority')  # Sort rules by priority
                for _, rule in applicable_rules.iterrows():
                    keyword_normalized = rule['Keyword'].lower().strip().rstrip('/')
                    if keyword_normalized in origin_url_normalized:
                        fallback_url = rule['Destination URL Pattern']
                        break

                # Address Redirection Rule - Check if the origin URL looks like an address and set fallback to properties/sale
                if fallback_url == '/' and re.search(r'\d{1,5}-[a-z0-9-]+', origin_url_normalized):
                    fallback_url = '/properties/sale'

                # Neighborhood Redirection Rule - Apply only if certain conditions are met
                if (fallback_url == '/'
                    and origin_url_normalized not in ['/', '']  # More robust check for root paths
                    and not re.search(r'\.html$', origin_url_normalized)  # Skip URLs ending with .html
                    and not re.search(r'/go/', origin_url_normalized)  # Skip URLs with '/go/' pattern
                    and not re.search(r'(test|careers|agent|page|about|contact|blog|faq|help)', origin_url_normalized)  # Skip specific keywords
                    and any(city_name.replace('-', ' ').lower().strip() in origin_url_normalized.replace('-', ' ') for city_name in city_names)):
                    fallback_url = '/neighborhoods'

                # Update the DataFrame with the fallback URL
                matches_df.at[idx, 'matched_url'] = fallback_url
                matches_df.at[idx, 'similarity_score'] = 'Fallback'
                matches_df.at[idx, 'fallback_applied'] = 'Yes'

        # Step 5: Final Check for Homepage Redirection
        for idx, matched_url in enumerate(matches_df['matched_url']):
            origin_url = matches_df.at[idx, 'origin_url']
            origin_url_normalized = re.sub(r'^https?://', '', origin_url.lower().strip().rstrip('/'))  # Remove protocol and trailing slash
            if origin_url_normalized in ['www.danadamsteam.com', '', 'index.html']:  # Match both absolute and relative homepages, including index.html
                matches_df.at[idx, 'matched_url'] = '/'
                matches_df.at[idx, 'similarity_score'] = 'Homepage'
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
