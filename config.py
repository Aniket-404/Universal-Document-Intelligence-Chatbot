# Local Hugging Face Model Settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast embedding model
CHAT_MODEL = "google/flan-t5-base"  # Better for summarization and QA tasks

# Alternative chat models you can use (just change CHAT_MODEL):
# "google/flan-t5-small" (faster, smaller - 250MB)
# "google/flan-t5-base" (good balance - 990MB) - RECOMMENDED
# "google/flan-t5-large" (better quality, slower - 3GB)
# "facebook/bart-large-cnn" (excellent for summarization but larger)
# "t5-small" (good for summarization, 240MB)

# Model Settings
MODEL_MAX_LENGTH = 1000  # Maximum tokens for generation
TEMPERATURE = 0.7  # Creativity (0.0 = deterministic, 1.0 = very creative)
USE_CUDA = True  # Set to False if you don't have GPU
DEVICE = "cpu"  # "auto", "cuda", "cpu"
MODEL_CACHE_DIR = "./models"  # Local directory to cache downloaded models

# Document Processing Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Vector Store Settings
SIMILARITY_THRESHOLD = 0.1
MAX_SEARCH_RESULTS = 5

# Web Search Settings
WEB_SEARCH_RESULTS = 5
WEB_SEARCH_TIMEOUT = 10

# Query Routing Settings
WEB_SEARCH_CONFIDENCE_THRESHOLD = 0.6
DOCUMENT_SEARCH_CONFIDENCE_THRESHOLD = 0.7
HYBRID_THRESHOLD = 0.3

# Fallback Settings (if local OpenAI models are not available)
USE_SENTENCE_TRANSFORMERS_FALLBACK = True
FALLBACK_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence Transformers model

# UI Settings
PAGE_TITLE = "Universal Document Intelligence Chatbot"
LAYOUT = "wide"

# File Settings
SUPPORTED_FILE_TYPES = ['pdf']
MAX_FILE_SIZE_MB = 50

# Response Settings
MAX_RESPONSE_LENGTH = 2000
MAX_SOURCES_DISPLAYED = 3