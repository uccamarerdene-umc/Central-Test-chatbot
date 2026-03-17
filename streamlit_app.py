import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# 1. Configuration and Secrets
st.set_page_config(page_title="Central Test AI", page_icon="🤖")

google_api_key = st.secrets.get("GOOGLE_API_KEY")
pinecone_api_key = st.secrets.get("PINECONE_API_KEY")

st.title("🤖 Central Test AI Assistant")
st.markdown("---")

index_name = "centralai"

# 2. Model Loading (Cached)
@st.cache_resource
def load_models():
    # Embedding model (Dimensions: 384)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Pinecone connection
    pc = Pinecone(api_key=pinecone_api_key)
    return embeddings, pc

embeddings, pc = load_models()

# 3. Sidebar - Data Synchronization
with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("🔄 Sync Data to Cloud"):
        # Check if Data folder exists
        if not os.path.exists("Data"):
            st.error("'Data' folder not found! Please ensure it exists in your repository.")
        else:
            with st.spinner("Reading documents and uploading to Pinecone..."):
                try:
                    # 1. Load Files
                    loader = DirectoryLoader("Data", glob="./*.docx", loader_cls=Docx2txtLoader)
                    docs = loader.load()
                    
                    # 2. Split Text - Optimized for better context retention
                    # chunk_size-ийг 500 болгож, overlap-ийг 100 болгож нэмлээ
                    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                    texts = splitter.split_documents(docs)
                    
                    # 3. Save to Pinecone
                    PineconeVectorStore.from_documents(
                        texts, 
                        embeddings, 
                        index_name=index_name,
                        pinecone_api_key=pinecone_api_key
                    )
                    st.success(f"Successfully uploaded {len(texts)} chunks to Pinecone!")
                except Exception as e:
                    st.error(f"Error during data sync: {e}")

# 4. Chat Interface
query = st.text_input("Ask a question:", placeholder="Ask about Central Test...")

if query:
    if not google_api_key or not pinecone_api_key:
        st.warning("API keys are missing. Please check your Streamlit Secrets.")
    else:
        with st.spinner("AI is thinking..."):
            try:
                # 1. Search in Pinecone
                vectorstore = PineconeVectorStore(
                    index_name=index_name, 
                    embedding=embeddings,
                    pinecone_api_key=pinecone_api_key
                )
                
                # Хайлтын үр дүнг 3 байсныг 7 болгож нэмсэн (илүү их мэдээлэл AI-д очно)
                search_results = vectorstore.similarity_search(query, k=7)
                context = "\n\n".join([doc.page_content for doc in search_results])
                
                # 2. Generate Answer with Gemini
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash", # Тогтвортой ажиллагааны үүднээс 1.5-flash ашиглахыг зөвлөж байна
                    google_api_key=google_api_key,
                    temperature=0.1 # Илүү бодитой хариулт өгүүлэх
                )
                
                # Сайжруулсан Промпт (Зааварчилгаа)
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
                
                # Display Results
                st.markdown("### 🤖 AI Response:")
                st.write(response.content)
                
                # Sources for transparency
                with st.expander("View Source Context (DEBUG)"):
                    st.info(context)
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
