import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Тохиргоо болон Нууц түлхүүрүүд (Streamlit Secrets-ээс уншина)
st.set_page_config(page_title="Central Test AI", page_icon="🤖")

# API Key-үүдийг Secrets-ээс авах (Кодон дотор ил бичиж болохгүй!)
google_api_key = st.secrets["GOOGLE_API_KEY"]
pinecone_api_key = st.secrets["PINECONE_API_KEY"]
index_name = "centralai"

st.title("🤖 Central Test AI Assistant")
st.markdown("---")

# 2. Модел ачаалах (Caching) - Google-ийн хамгийн сүүлийн 004 модел
@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", # 768 dimensions
        google_api_key=google_api_key
    )

embeddings = get_embeddings()

# 3. Sidebar - Датагаа Pinecone руу илгээх (Sync хийх хэсэг заавал хэрэгтэй)
with st.sidebar:
    st.header("⚙️ Удирдлага")
    if st.button("🔄 Датаг Pinecone-руу синхрончлох"):
        if not os.path.exists("Data"):
            st.error("'Data' хавтас олдсонгүй!")
        else:
            with st.spinner("Файлуудыг уншиж байна..."):
                try:
                    # Файлуудыг унших
                    loader = DirectoryLoader("Data", glob="./*.docx", loader_cls=Docx2txtLoader)
                    docs = loader.load()
                    
                    # Текстийг жижиглэх
                    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                    texts = splitter.split_documents(docs)
                    
                    # Pinecone-руу хадгалах
                    PineconeVectorStore.from_documents(
                        texts, 
                        embeddings, 
                        index_name=index_name, 
                        pinecone_api_key=pinecone_api_key
                    )
                    st.success(f"Амжилттай! {len(texts)} хэсэг өгөгдөл хадгалагдлаа.")
                except Exception as e:
                    st.error(f"Алдаа гарлаа: {e}")

# 4. Чат болон Хайлт хийх хэсэг
query = st.text_input("Асуултаа энд бичнэ үү:", placeholder="Жишээ нь: Central Test гэж юу вэ?")

if query:
    with st.spinner("AI хариулт бэлдэж байна..."):
        try:
            # Pinecone-оос хайх
            vectorstore = PineconeVectorStore(
                index_name=index_name, 
                embedding=embeddings, 
                pinecone_api_key=pinecone_api_key
            )
            
            # Хамгийн ойр 5 илэрцийг хайх
            docs = vectorstore.similarity_search(query, k=5)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Gemini моделтой холбогдох
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                google_api_key=google_api_key, 
                temperature=0.1
            )
            
            # Зааварчилгаа (Prompt)
            prompt = f"""
            Та бол Central Test компанийн албан ёсны AI туслах байна. 
            Доорх 'Мэдээлэл' хэсэгт байгаа текст дээр тулгуурлан асуултад маш тодорхой хариул.

            Мэдээлэл:
            {context}

            Асуулт:
            {query}

            Хэрэв өгөгдсөн мэдээлэл дотор хариулт байхгүй бол:
            "Уучлаарай, миний мэдээллийн санд энэ талаарх мэдээлэл алга байна." гэж хариул.
            """
            
            response = llm.invoke(prompt)
            
            # Хариултыг харуулах
            st.markdown("### 🤖 AI Хариулт:")
            st.info(response.content)
            
            # Эх сурвалжийг харах (Нууц байдлаар)
            with st.expander("Ашигласан мэдээлэл (Source Context)"):
                st.write(context)
                
        except Exception as e:
            st.error(f"Алдаа гарлаа: {e}")
