import streamlit as st
import requests
from PIL import Image
import os

API_URL = os.environ.get("API_URL", "http://localhost:5000")

st.set_page_config(page_title="Banknote Classifier", page_icon="")

st.title("Banknote Classification")

uploaded_file = st.file_uploader("Upload Banknote Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)
    
    if st.button("Classify"):
        with st.spinner("Classifying..."):
            uploaded_file.seek(0)
            
            try:
                files = {"image": uploaded_file}
                response = requests.post(f"{API_URL}/predict", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("Classification Complete!")
                    
                    st.markdown("---")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Currency", result['predicted_class'])
                    with col2:
                        st.metric("Confidence", result['confidence_percentage'])
                    
                    st.markdown("---")
                    
                    st.subheader("All Probabilities")
                    
                    sorted_probs = sorted(
                        result['probabilities'].items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )
                    
                    for currency, prob in sorted_probs:
                        percentage = prob * 100
                        st.write(f"**{currency}**")
                        st.progress(prob)
                        st.caption(f"{percentage:.2f}%")
                        st.write("")
                else:
                    st.error(f"Error: {response.json().get('error', 'Unknown error')}")
            
            except Exception as e:
                st.error(f"Failed to connect to API: {str(e)}")