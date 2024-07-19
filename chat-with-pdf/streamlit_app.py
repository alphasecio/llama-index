import os, tempfile, streamlit as st
from llama_index.core import VectorStoreIndex
from llama_parse import LlamaParse

# Streamlit app config
st.subheader("Chat with PDF")
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API key", type="password")
    llama_cloud_api_key = st.text_input("LlamaCloud API key", type="password")
    source_doc = st.file_uploader("Source document", type="pdf")
col1, col2 = st.columns([4,1])
query = col1.text_input("Query", label_visibility="collapsed")

# Session state initialization for documents and retrievers
if  "loaded_doc" not in st.session_state or "query_engine" not in st.session_state:
    st.session_state.loaded_doc = None
    st.session_state.query_engine = None

submit = col2.button("Submit")

# If the "Submit" button is clicked
if submit:
    if not openai_api_key.strip() or not llama_cloud_api_key.strip() or not query.strip():
        st.error("Please provide the missing fields.")
    elif not source_doc:
        st.error("Please upload the source document.")
    else:
        with st.spinner("Please wait..."):
            # Set API key environment variables
            os.environ["OPENAI_API_KEY"] = openai_api_key
            os.environ["LLAMA_CLOUD_API_KEY"] = llama_cloud_api_key
            
            # Check if document has already been uploaded
            if st.session_state.loaded_doc != source_doc:
                try:
                    # Initialize parser with markdown output (alternative: text)
                    parser = LlamaParse(language="en", result_type="markdown")
                    
                    # Save uploaded file temporarily to disk, parse uploaded file, delete temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(source_doc.read())

                    documents = parser.load_data(tmp_file.name)
                    os.remove(tmp_file.name)
                    
                    # Create a vector store index for uploaded file
                    index = VectorStoreIndex.from_documents(documents)
                    st.session_state.query_engine = index.as_query_engine()
                    
                    # Store the uploaded file in session state to prevent reloading
                    st.session_state.loaded_doc = source_doc
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            try:
                response = st.session_state.query_engine.query(query)
                st.success(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
