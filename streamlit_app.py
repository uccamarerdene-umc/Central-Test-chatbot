import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# 1. Тохиргоо
st.set_page_config(page_title="Central Test AI Assistant", page_icon="🤖")

google_api_key = st.secrets.get("GOOGLE_API_KEY")
pinecone_api_key = st.secrets.get("PINECONE_API_KEY")
index_name = "centralai"

# 2. Модель ачааллах (768 Dimension тааруулах)
@st.cache_resource
def load_models():
    # 'models/' угтварыг нэмж өгснөөр API-д илүү ойлгомжтой болно
    # text-embedding-004 нь яг 768 хэмжээстэй вектор үүсгэдэг
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        google_api_key=google_api_key
    )
    pc = Pinecone(api_key=pinecone_api_key)
    return embeddings, pc

if not google_api_key or not pinecone_api_key:
    st.error("API түлхүүрүүд тохируулагдаагүй байна! Streamlit Secrets-ээ шалгана уу.")
    st.stop()

embeddings, pc = load_models()

# 3. Sidebar - Sync Data
with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("🔄 Sync Data to Cloud"):
        if not os.path.exists("Data"):
            st.error("'Data' хавтас олдсонгүй!")
        else:
            with st.spinner("Pinecone-руу илгээж байна..."):
                try:
                    loader = DirectoryLoader("Data", glob="./*.docx", loader_cls=Docx2txtLoader)
                    docs = loader.load()

                    if not docs:
                        st.warning(".docx файл олдсонгүй.")
                    else:
                        # Chunking
                        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
                        texts = splitter.split_documents(docs)

                        # Хадгалах
                        PineconeVectorStore.from_documents(
                            texts, 
                            embeddings, 
                            index_name=index_name,
                            pinecone_api_key=pinecone_api_key
                        )
                        st.success(f"Амжилттай! {len(texts)} хэсэг өгөгдөл хадгалагдлаа.")
                except Exception as e:
                    st.error(f"Sync failed: {e}")

# 4. Chat
st.title("🤖 Central Test AI Assistant")
query = st.text_input("Асуултаа бичнэ үү:")

if query:
    with st.spinner("AI хариулт боловсруулж байна..."):
        try:
            vectorstore = PineconeVectorStore(
                index_name=index_name, 
                embedding=embeddings,
                pinecone_api_key=pinecone_api_key
            )

            search_results = vectorstore.similarity_search(query, k=5)
            context = "\n\n".join([doc.page_content for doc in search_results])

            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                google_api_key=google_api_key,
                temperature=0.2
            )

            prompt = f"Мэдээлэл: {context}\n\nАсуулт: {query}"
            response = llm.invoke(prompt)

            st.markdown("### 🤖 AI Хариулт:")
            st.write(response.content)

        except Exception as e:
            st.error(f"Алдаа гарлаа: {e}")
