import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# 1. Configuration
st.set_page_config(page_title="Central Test AI", page_icon="🤖")

google_api_key = st.secrets.get("GOOGLE_API_KEY")
pinecone_api_key = st.secrets.get("PINECONE_API_KEY")

st.title("🤖 Central Test AI Assistant (Google V4)")
st.markdown("---")

# Use your new index name here
index_name = "centralai" 

# 2. Model Loading
@st.cache_resource
def load_models():
    # HIGH QUALITY: Google text-embedding-004 (768 dimensions)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=google_api_key
    )
    pc = Pinecone(api_key=pinecone_api_key)
    return embeddings, pc

embeddings, pc = load_models()

# 3. Sidebar - Sync Data
with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("🔄 Sync Data (Google V4)"):
        if not os.path.exists("Data"):
            st.error("'Data' folder not found!")
        else:
            with st.spinner("Uploading to Pinecone with 768 Dimensions..."):
                try:
                    loader = DirectoryLoader("Data", glob="./*.docx", loader_cls=Docx2txtLoader)
                    docs = loader.load()
                    # Optimized chunking for higher quality
                    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                    texts = splitter.split_documents(docs)
                    
                    # Create new vectorstore
                    PineconeVectorStore.from_documents(
                        texts, embeddings, index_name=index_name, pinecone_api_key=pinecone_api_key
                    )
                    st.success(f"Success! {len(texts)} chunks synced using Google Embeddings.")
                except Exception as e:
                    st.error(f"Sync Error: {e}")

# 4. Chat Interface
query = st.text_input("Ask a question:", placeholder="Search Central Test documents...")

if query:
    with st.spinner("AI is thinking..."):
        try:
            vectorstore = PineconeVectorStore(
                index_name=index_name, embedding=embeddings, pinecone_api_key=pinecone_api_key
            )
            # Retrieve top 7 relevant pieces
            search_results = vectorstore.similarity_search(query, k=7)
            context = "\n\n".join([doc.page_content for doc in search_results])
            
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key, temperature=0.1)
            
            prompt = f"""
            Та бол Central Test компанийн албан ёсны AI туслах байна. 
            Доорх мэдээлэлд тулгуурлан асуултанд монгол хэлээр маш тодорхой хариулна уу.
            
            Мэдээлэл: {context}
            Асуулт: {query}
            
            Хэрэв мэдээлэлд хариулт байхгүй бол 'Мэдээлэл алга' гэж хэлээрэй.
            """
            
            response = llm.invoke(prompt)
            st.markdown("### 🤖 AI Response:")
            st.write(response.content)
            
            with st.expander("Show Context (DEBUG)"):
                st.info(context)
        except Exception as e:
            st.error(f"Error: {e}")
