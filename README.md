# Universal Document Intelligence Chatbot  

A simple, private, and powerful chatbot that can answer your questions using both your own documents and the web.  

## What it can do  
- **Runs locally**: Uses Hugging Face Transformers, so your data stays private  
- **Quick search**: Finds answers fast with Sentence Transformers  
- **Smart choice**: Decides when to pull from your documents or from the web  
- **Handles multiple formats**: Upload PDFs and ask questions directly  
- **Stay up to date**: Can use web search for the latest information  
- **No setup hassle**: Downloads models automatically the first time you use them  

## Getting Started  

### Setup  
```bash
# Create a virtual environment
python -m venv venv  

# Activate it
venv\Scripts\activate.bat  

# Install dependencies
pip install -r requirements.txt  

# Launch the app
streamlit run app.py  
