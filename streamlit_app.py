import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# 1. Configuration and Secrets
st.set_page_config(page_title="Central Test AI Assistant", page_icon="🤖")

google_api_key = st.secrets.get("GOOGLE_API_KEY")
pinecone_api_key = st.secrets.get("PINECONE_API_KEY")

st.title("🤖 Central Test AI Assistant")
st.markdown("---")

index_name = "centralai"

# 2. Model Loading (Cached for Performance)
@st.cache_resource
def load_models():
    # Embedding model: 384 dimensions for all-MiniLM-L6-v2
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Pinecone initialization
    pc = Pinecone(api_key=pinecone_api_key)
    return embeddings, pc

embeddings, pc = load_models()

# 3. Sidebar - Data Management
with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("🔄 Sync Data to Cloud"):
        # Verifying the existence of the Data folder
        if not os.path.exists("Data"):
            st.error("'Data' folder not found! Please check your directory structure.")
        else:
            with st.spinner("Processing documents and updating Pinecone..."):
                try:
                    # 1. Load Documents
                    loader = DirectoryLoader("Data", glob="./*.docx", loader_cls=Docx2txtLoader)
                    docs = loader.load()
                    
                    # 2. Optimized Text Splitting (Chunking)
                    # Smaller chunks with higher overlap improve retrieval accuracy
                    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                    texts = splitter.split_documents(docs)
                    
                    # 3. Upsert to Pinecone
                    PineconeVectorStore.from_documents(
                        texts, 
                        embeddings, 
                        index_name=index_name,
                        pinecone_api_key=pinecone_api_key
                    )
                    st.success(f"Successfully synced {len(texts)} text blocks to Pinecone!")
                except Exception as e:
                    st.error(f"Sync failed: {e}")

# 4. Chat Interface
query = st.text_input("Ask a question:", placeholder="Search the Central Test knowledge base...")

if query:
    if not google_api_key or not pinecone_api_key:
        st.warning("API keys are missing. Please verify your Streamlit Secrets.")
    else:
        with st.spinner("Analyzing data and generating response..."):
            try:
                # 1. Semantic Search in Pinecone
                vectorstore = PineconeVectorStore(
                    index_name=index_name, 
                    embedding=embeddings,
                    pinecone_api_key=pinecone_api_key
                )
                
                # Retrieving top 7 most relevant context segments
                search_results = vectorstore.similarity_search(query, k=7)
                context = "\n\n".join([doc.page_content for doc in search_results])
                
                # 2. LLM Configuration
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.5-flash-lite", 
                    google_api_key=google_api_key,
                    temperature=0.1
                )
                
                # Mongolian prompt ensures the AI responds in the correct language and context
                prompt = f"""
                Та бол Central Test компанийн албан ёсны AI туслах байна. 
                Доорх 'Мэдээлэл' хэсэгт байгаа текст дээр тулгуурлан хэрэглэгчийн асуултанд маш дэлгэрэнгүй, эелдэг хариулна уу.
                
                Мэдээлэл:
                {context}
                
                Асуулт: {query}
                
                ХАРИУЛАХ ЗААВАР:
                1. Зөвхөн өгөгдсөн 'Мэдээлэл' доторх текстийг ашигла.
                2. Хэрэв мэдээлэл дотор хариулт байвал түүнийг логиктой, эмх цэгцтэй (магадгүй жагсаалтаар) тайлбарла.
                3. Хэрэв асуултанд хариулах мэдээлэл огт байхгүй бол "Уучлаарай, миний мэдээллийн санд энэ талаарх мэдээлэл алга байна. Та асуултаа арай өөрөөр асууж үзнэ үү?" гэж хариулаарай.
                4. Хэзээ ч мэдээллийн санд байхгүй зүйлийг өөрөө зохиож хариулж болохгүй.
                """
                
                response = llm.invoke(prompt)
                
                # Output results to the user
                st.markdown("### 🤖 AI Response:")
                st.write(response.content)
                
                # Transparency section
                with st.expander("Show retrieved data (Source context)"):
                    st.info(context)
                    
            except Exception as e:
                st.error(f"System error: {e}")
