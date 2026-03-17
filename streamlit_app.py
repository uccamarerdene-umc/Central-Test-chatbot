import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# .env файлаас нууц түлхүүрүүдийг унших
load_dotenv()

# Вэб хуудасны гарчиг ба тохиргоо
st.set_page_config(page_title="Central Test AI", page_icon="🤖")
st.title("🤖 Central Test AI Туслах")
st.markdown("---")

# Pinecone холболт (Таны үүсгэсэн 'quickstart' индекс)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "centralai" 

# Текстүүдийг вектор болгох модел (Dimensions: 384)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Sidebar - Датаг шинэчлэх хэсэг
with st.sidebar:
    st.header("⚙️ Тохиргоо")
    if st.button("🔄 Датаг онлайн руу шинэчлэх"):
        if not os.path.exists("Data"):
            st.error("'Data' хавтас олдсонгүй!")
        else:
            with st.spinner("Баримтуудыг уншиж, Pinecone-руу илгээж байна..."):
                # 1. Файлуудыг унших
                loader = DirectoryLoader("Data", glob="./*.docx", loader_cls=Docx2txtLoader)
                docs = loader.load()
                # 2. Текстүүдийг жижиглэх
                splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
                texts = splitter.split_documents(docs)
                # 3. Pinecone-руу хадгалах
                PineconeVectorStore.from_documents(texts, embeddings, index_name=index_name)
                st.success("Дата онлайн санд амжилттай хадгалагдлаа!")

# Асуулт асуух хэсэг
query = st.text_input("Асуултаа бичнэ үү:", placeholder="Central test-ийн талаар асуух...")

if query:
    with st.spinner("AI хариулт бэлдэж байна..."):
        try:
            # 1. Pinecone-оос хайх
            vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
            search_results = vectorstore.similarity_search(query, k=3)
            context = "\n".join([doc.page_content for doc in search_results])
            
            # 2. Gemini-ээр хариулуулах
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash", 
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            
            prompt = f"""
            Мэдээлэл: {context}
            
            Асуулт: {query}
            
            Дээрх мэдээлэлд үндэслэн асуултанд маш тодорхой, монгол хэлээр хариулна уу. 
            Хэрэв мэдээлэл дотор хариулт байхгүй бол 'Мэдээлэл олдсонгүй' гэж хэлээрэй.
            """
            
            response = llm.invoke(prompt)
            
            # Хариултыг харуулах
            st.markdown("### 🤖 AI-ийн хариулт:")
            st.write(response.content)
            
        except Exception as e:
            st.error(f"Алдаа гарлаа: {e}")
