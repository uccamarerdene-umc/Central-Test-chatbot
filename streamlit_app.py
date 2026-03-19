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

# 2. Модель ачааллах (Кэш ашиглах)
@st.cache_resource
def load_models():
    # 404 NOT_FOUND алдаанаас сэргийлж 'models/' угтваргүй бичив.
    # Энэ модель 768 хэмжээстэй тул таны Pinecone index-тэй яг таарна.
    embeddings = GoogleGenerativeAIEmbeddings(
        model="embedding-001", 
        google_api_key=google_api_key
    )
    pc = Pinecone(api_key=pinecone_api_key)
    return embeddings, pc

if not google_api_key or not pinecone_api_key:
    st.error("API keys are missing! Please check Streamlit Secrets.")
    st.stop()

embeddings, pc = load_models()

# 3. Sidebar - Өгөгдөл синхрончлох (Sync)
with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("🔄 Sync Data to Cloud"):
        if not os.path.exists("Data"):
            st.error("'Data' хавтас олдсонгүй! GitHub репозитортоо 'Data' хавтас үүсгэж, .docx файлуудаа хийнэ үү.")
        else:
            with st.spinner("Баримтуудыг боловсруулж, Pinecone-руу илгээж байна..."):
                try:
                    # 1. Файл унших
                    loader = DirectoryLoader("Data", glob="./*.docx", loader_cls=Docx2txtLoader)
                    docs = loader.load()

                    if not docs:
                        st.warning("Data хавтсанд .docx файл олдсонгүй.")
                    else:
                        # 2. Текстийг хэсэгчлэн хуваах
                        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
                        texts = splitter.split_documents(docs)

                        # 3. Pinecone-руу хадгалах
                        PineconeVectorStore.from_documents(
                            texts, 
                            embeddings, 
                            index_name=index_name,
                            pinecone_api_key=pinecone_api_key
                        )
                        st.success(f"Амжилттай! {len(texts)} хэсэг өгөгдлийг Pinecone-руу хадгаллаа.")
                except Exception as e:
                    st.error(f"Sync failed: {e}")

# 4. Чатлах хэсэг
st.title("🤖 Central Test AI Assistant")
query = st.text_input("Асуултаа бичнэ үү:", placeholder="Central Test-ийн талаар юу мэдэхийг хүсэж байна?")

if query:
    with st.spinner("AI хариулт боловсруулж байна..."):
        try:
            vectorstore = PineconeVectorStore(
                index_name=index_name, 
                embedding=embeddings,
                pinecone_api_key=pinecone_api_key
            )

            # Хамгийн хамааралтай 5 хэсгийг хайх
            search_results = vectorstore.similarity_search(query, k=5)
            context = "\n\n".join([doc.page_content for doc in search_results])

            # Gemini 1.5-flash ашиглах (моделийн нэрийг зөв бичих)
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
            st.markdown("### 🤖 AI Хариулт:")
            st.write(response.content)

            with st.expander("Эх сурвалж (Source Context)"):
                st.info(context)

        except Exception as e:
            st.error(f"Алдаа гарлаа: {e}")
