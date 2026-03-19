import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# 1. Configuration and Secrets
st.set_page_config(page_title="Central Test AI Assistant", page_icon="🤖")

google_api_key = st.secrets.get("GOOGLE_API_KEY")
pinecone_api_key = st.secrets.get("PINECONE_API_KEY")
index_name = "centralai"

# 2. Model Loading (Cached for Performance)
@st.cache_resource
def load_models():
    # 404 алдаанаас сэргийлж хамгийн тогтвортой 'embedding-001' моделийг ашиглав
    # Энэ нь 768 хэмжээстэй тул таны Pinecone-той яг таарна.
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=google_api_key
    )
    pc = Pinecone(api_key=pinecone_api_key)
    return embeddings, pc

if not google_api_key or not pinecone_api_key:
    st.error("API keys are missing! Please check Streamlit Secrets.")
    st.stop()

embeddings, pc = load_models()

# 3. Sidebar - Data Management (Sync)
with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("🔄 Sync Data to Cloud"):
        if not os.path.exists("Data"):
            st.error("'Data' folder not found! Please ensure it exists in your GitHub repo.")
        else:
            with st.spinner("Processing documents and updating Pinecone..."):
                try:
                    loader = DirectoryLoader("Data", glob="./*.docx", loader_cls=Docx2txtLoader)
                    docs = loader.load()

                    if not docs:
                        st.warning("No .docx files found in the 'Data' folder.")
                    else:
                        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
                        texts = splitter.split_documents(docs)

                        # Хуучин өгөгдлийг цэвэрлэх эсвэл шууд нэмэх
                        PineconeVectorStore.from_documents(
                            texts, 
                            embeddings, 
                            index_name=index_name,
                            pinecone_api_key=pinecone_api_key
                        )
                        st.success(f"Successfully synced {len(texts)} chunks to Pinecone!")
                except Exception as e:
                    st.error(f"Sync failed: {e}")

# 4. Chat Interface
st.title("🤖 Central Test AI Assistant")
query = st.text_input("Ask a question:", placeholder="Search the Central Test knowledge base...")

if query:
    with st.spinner("Analyzing data and generating response..."):
        try:
            vectorstore = PineconeVectorStore(
                index_name=index_name, 
                embedding=embeddings,
                pinecone_api_key=pinecone_api_key
            )

            search_results = vectorstore.similarity_search(query, k=5)
            context = "\n\n".join([doc.page_content for doc in search_results])

            # ЗАСРУУЛГА: Моделийн нэр 'gemini-1.5-flash' байх ёстой
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                google_api_key=google_api_key,
                temperature=0.2
            )

            prompt = f"""
            Та бол Central Test компанийн албан ёсны AI туслах байна. 
            Доорх 'Мэдээлэл' хэсэгт байгаа текст дээр тулгуурлан хэрэглэгчийн асуултанд маш дэлгэрэнгүй, эелдэг хариулна уу.

            Мэдээлэл:
            {context}

            Асуулт: {query}
            """

            response = llm.invoke(prompt)

            st.markdown("---")
            st.markdown("### 🤖 AI Response:")
            st.write(response.content)

            with st.expander("Show retrieved context (Source)"):
                st.info(context)

        except Exception as e:
            st.error(f"An error occurred: {e}")
