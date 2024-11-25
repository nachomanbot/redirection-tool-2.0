import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import re
from fuzzywuzzy import fuzz

# Set the page title
st.title("AI-Powered Redirect Mapping Tool - Version 2.0")

st.markdown("""

Relevancy Script made by Daniel Emery

Everything else by: NDA

⚡ **What It Is:**  
This tool automates redirect mappings during site migrations by matching URLs from an old site to a new site based on content similarity and custom fallback rules for unmatched URLs.

⚡ **How to Use It:**  
1. Upload `origin.csv` and `destination.csv` files. Ensure that your files have the following headers: Address,Title 1,Meta Description 1,H1-1.
2. Ensure that you remove any duplicates and the http status of all URLs is 200.
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
    origin_df['combined_text'] = origin_df.apply(lambda x: (' '.join([x[0]] * 2) + ' ' + ' '.join(x.astype(str))), axis=1)  # Increase weight of URL
    destination_df['combined_text'] = destination_df.apply(lambda x: (' '.join([x[0]] * 2) + ' ' + ' '.join(x.astype(str))), axis=1)  # Increase weight of URL

    # Step 3: Button to Process Matching
    if st.button("Let's Go!"):
        st.info("Processing data... This may take a while.")

        # Step 4: Custom String Matching Before Embedding
        matches = []
        for origin_url in origin_df['Address']:
            best_match = None
            highest_score = 0
            origin_parts = origin_url.lower().split('/')
            for destination_url in destination_df['Address']:
                destination_parts = destination_url.lower().split('/')
                common_parts = set(origin_parts) & set(destination_parts)
                score = len(common_parts) / max(len(origin_parts), len(destination_parts)) * 100

                # Use fuzzywuzzy for partial matching as a secondary check if no exact parts match
                fuzzy_score = fuzz.partial_ratio(origin_url.lower(), destination_url.lower())
                score = max(score, fuzzy_score)  # Use the higher score between common parts and fuzzy matching

                if score > highest_score:
                    highest_score = score
                    best_match = destination_url

            # If the match score is high enough, consider it a match
            if highest_score >= 70:  # Lowered threshold for partial matching
                matches.append((origin_url, best_match, highest_score / 100, 'Yes'))
            else:
                matches.append((origin_url, None, None, 'No'))

        # Convert matches to DataFrame
        matches_df = pd.DataFrame(matches, columns=['origin_url', 'matched_url', 'similarity_score', 'fallback_applied'])

        # Separate matched and unmatched URLs
        unmatched_df = matches_df[matches_df['matched_url'].isna()].copy()
        matched_df = matches_df[~matches_df['matched_url'].isna()].copy()

        # Proceed with embedding for unmatched URLs
        unmatched_origin_df = origin_df[origin_df['Address'].isin(unmatched_df['origin_url'])]

        # Use a pre-trained model for embedding
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Vectorize the combined text
        origin_embeddings = model.encode(unmatched_origin_df['combined_text'].tolist(), show_progress_bar=True)
        destination_embeddings = model.encode(destination_df['combined_text'].tolist(), show_progress_bar=True)

        # Create a FAISS index
        dimension = origin_embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(destination_embeddings.astype('float32'))

        # Perform the search for the nearest neighbors
        D, I = faiss_index.search(origin_embeddings.astype('float32'), k=1)

        # Calculate similarity scores
        similarity_scores = 1 - (D / (np.max(D) + 1e-10))  # Add small value to avoid division by zero

        # Create the output DataFrame with similarity scores for unmatched URLs
        unmatched_results_df = pd.DataFrame({
            'origin_url': unmatched_origin_df.iloc[:, 0],
            'matched_url': destination_df.iloc[:, 0].iloc[I.flatten()].apply(lambda x: x.split()[0]).values,  # Ensure only the URL is added
            'similarity_score': np.round(similarity_scores.flatten(), 4),
            'fallback_applied': ['No'] * len(unmatched_origin_df)  # Default to 'No' for fallback
        })

        # Combine matched and unmatched results after embedding matching
        interim_results_df = pd.concat([matched_df, unmatched_results_df], ignore_index=True)

        # Step 5: Apply Fallbacks for Low Scores
        fallback_threshold = 0.65
        for idx, score in enumerate(interim_results_df['similarity_score']):
            if isinstance(score, (float, int)) and score < fallback_threshold:
                origin_url = interim_results_df.at[idx, 'origin_url']
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
                interim_results_df.at[idx, 'matched_url'] = fallback_url
                interim_results_df.at[idx, 'similarity_score'] = 'Fallback'
                interim_results_df.at[idx, 'fallback_applied'] = 'Yes'

        # Step 6: Final Check for Homepage Redirection
        for idx, matched_url in enumerate(interim_results_df['matched_url']):
            origin_url = interim_results_df.at[idx, 'origin_url']
            origin_url_normalized = re.sub(r'^https?://', '', origin_url.lower().strip().rstrip('/'))  # Remove protocol and trailing slash
            if origin_url_normalized in ['www.danadamsteam.com', '', 'index.html', 'index.php', 'index.asp']:  # Match both absolute and relative homepages, including index.html, index.php, index.asp
                interim_results_df.at[idx, 'matched_url'] = '/'
                interim_results_df.at[idx, 'similarity_score'] = 'Homepage'
                interim_results_df.at[idx, 'fallback_applied'] = 'Yes'

        # Step 7: Display and Download Results
        st.success("Matching complete! Download your results below.")
        st.write(interim_results_df)

        st.download_button(
            label="Download Results as CSV",
            data=interim_results_df.to_csv(index=False),
            file_name="redirect_mapping_output_v2.csv",
            mime="text/csv",
        )
