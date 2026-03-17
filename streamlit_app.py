import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# 1. Тохиргоо болон Нууц түлхүүрүүд (Streamlit Secrets-ээс унших)
st.set_page_config(page_title="Central Test AI", page_icon="🤖")

# Streamlit Cloud дээр Secrets-ээс түлхүүрүүдийг авна
google_api_key = st.secrets.get("GOOGLE_API_KEY")
pinecone_api_key = st.secrets.get("PINECONE_API_KEY")

st.title("🤖 Central Test AI Туслах")
st.markdown("---")

# Индексийн нэрийг таны Pinecone дээрх нэртэй яг адилхан (centralai) болголоо
index_name = "centralai"

# 2. Моделүүдийг бэлдэх (Нэг удаа ачаална)
@st.cache_resource
def load_models():
    # Текстүүдийг вектор болгох модел (Dimensions: 384)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Pinecone холболт
    pc = Pinecone(api_key=pinecone_api_key)
    return embeddings, pc

embeddings, pc = load_models()

# 3. Sidebar - Датаг шинэчлэх хэсэг
with st.sidebar:
    st.header("⚙️ Тохиргоо")
    if st.button("🔄 Датаг онлайн руу шинэчлэх"):
        if not os.path.exists("Data"):
            st.error("'Data' хавтас олдсонгүй! GitHub-дээ Data хавтсаа оруулсан эсэхээ шалгаарай.")
        else:
            with st.spinner("Баримтуудыг уншиж, Pinecone-руу илгээж байна..."):
                try:
                    # 1. Файлуудыг унших
                    loader = DirectoryLoader("Data", glob="./*.docx", loader_cls=Docx2txtLoader)
                    docs = loader.load()
                    
                    # 2. Текстүүдийг жижиглэх
                    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
                    texts = splitter.split_documents(docs)
                    
                    # 3. Pinecone-руу хадгалах
                    PineconeVectorStore.from_documents(
                        texts, 
                        embeddings, 
                        index_name=index_name,
                        pinecone_api_key=pinecone_api_key
                    )
                    st.success(f"Нийт {len(texts)} хэсэг дата амжилттай хадгалагдлаа!")
                except Exception as e:
                    st.error(f"Дата хуулахад алдаа гарлаа: {e}")

# 4. Асуулт асуух хэсэг
query = st.text_input("Асуултаа бичнэ үү:", placeholder="Central test-ийн талаар асуух...")

if query:
    if not google_api_key or not pinecone_api_key:
        st.warning("API түлхүүрүүд тохируулагдаагүй байна. Settings -> Secrets хэсгийг шалгана уу.")
    else:
        with st.spinner("AI хариулт бэлдэж байна..."):
            try:
                # 1. Pinecone-оос хайх
                vectorstore = PineconeVectorStore(
                    index_name=index_name, 
                    embedding=embeddings,
                    pinecone_api_key=pinecone_api_key
                )
                
                # Хамгийн ойр 3 хэсэг мэдээллийг хайж олох
                search_results = vectorstore.similarity_search(query, k=3)
                context = "\n\n".join([doc.page_content for doc in search_results])
                
                # 2. Gemini-ээр хариулуулах (Моделийн нэрийг зөв болгож зассан)
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash", 
                    google_api_key=google_api_key,
                    temperature=0.1 # Хариултыг илүү бодитой байлгах үүднээс
                )
                
                prompt = f"""
                Та бол Central Test компанийн албан ёсны туслах AI байна. 
                Доорх мэдээлэлд үндэслэн асуултанд монгол хэлээр маш тодорхой хариулна уу.
                
                Мэдээлэл:
                {context}
                
                Асуулт: {query}
                
                Хэрэв өгөгдсөн мэдээлэл дотор хариулт байхгүй бол "Уучлаарай, миний мэдээллийн санд энэ талаар мэдээлэл алга байна." гэж хариулаарай.
                """
                
                response = llm.invoke(prompt)
                
                # Хариултыг харуулах
                st.markdown("### 🤖 AI-ийн хариулт:")
                st.write(response.content)
                
                # Эх сурвалжийг харуулах (Хэрэгцээтэй бол нээж үзэх)
                with st.expander("Ашигласан мэдээллийн хэсэг"):
                    st.info(context)
                    
            except Exception as e:
                st.error(f"Алдаа гарлаа: {e}")
