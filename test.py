import google.generativeai as genai

# Set up API key (replace with your actual key)
genai.configure(api_key="AIzaSyAdKv_6J7-yjBZSd8LynHEc4B90h4SWGZI")


try:
    # Correct model name with prefix
    response = genai.embed_content(model="models/embedding-001", content="Test text for embedding.")
    
    # Check response
    if "embedding" in response:
        print("Embedding access confirmed:", response["embedding"][:5])  # Print first 5 values
    else:
        print("Embedding model responded, but no embeddings returned:", response)

except Exception as e:
    print("Error:", e)
