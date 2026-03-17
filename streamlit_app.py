import streamlit as st
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# ----------------------------------
# 1. PAGE CONFIG
# ----------------------------------
st.set_page_config(page_title="Central Test AI", page_icon="🤖")

st.title("🤖 Central Test AI Туслах")
st.markdown("---")

# ----------------------------------
# 2. API KEYS (Streamlit Secrets)
# ----------------------------------
google_api_key = st.secrets.get("GOOGLE_API_KEY")
pinecone_api_key = st.secrets.get("PINECONE_API_KEY")

index_name = "centralai"

# ----------------------------------
# 3. LOAD MODELS (CACHE)
# ----------------------------------
@st.cache_resource
def load_models():
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    pc = Pinecone(api_key=pinecone_api_key)
    return embeddings, pc

embeddings, pc = load_models()

# ----------------------------------
# 4. SIDEBAR (DATA UPDATE)
# ----------------------------------
with st.sidebar:
    st.header("⚙️ Тохиргоо")

    if st.button("🔄 Датаг Pinecone руу шинэчлэх"):

        if not os.path.exists("Data"):
            st.error("❌ 'Data' хавтас олдсонгүй!")
        else:
            with st.spinner("📄 Баримт уншиж байна..."):
                try:
                    # Load DOCX
                    loader = DirectoryLoader(
                        "Data",
                        glob="*.docx",
                        loader_cls=Docx2txtLoader
                    )
                    docs = loader.load()

                    # Split text
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=600,
                        chunk_overlap=50
                    )
                    texts = splitter.split_documents(docs)

                    # Upload to Pinecone
                    PineconeVectorStore.from_documents(
                        texts,
                        embeddings,
                        index_name=index_name,
                        pinecone_api_key=pinecone_api_key
                    )

                    st.success(f"✅ {len(texts)} хэсэг дата амжилттай хадгалагдлаа!")

                except Exception as e:
                    st.error(f"❌ Алдаа: {e}")

# ----------------------------------
# 5. CHAT INPUT
# ----------------------------------
query = st.text_input(
    "Асуултаа бичнэ үү:",
    placeholder="Central Test-ийн талаар асуугаарай..."
)

# ----------------------------------
# 6. QA SYSTEM
# ----------------------------------
if query:

    if not google_api_key or not pinecone_api_key:
        st.warning("⚠️ API key-үүдээ secrets дээр тохируулна уу!")
    else:
        with st.spinner("🤖 AI бодож байна..."):

            try:
                # Pinecone search
                vectorstore = PineconeVectorStore(
                    index_name=index_name,
                    embedding=embeddings,
                    pinecone_api_key=pinecone_api_key
                )

                results = vectorstore.similarity_search(query, k=3)

                if not results:
                    st.warning("Мэдээлэл олдсонгүй!")
                else:
                    context = "\n\n".join(
                        [doc.page_content for doc in results]
                    )

                    # ✅ FIXED MODEL NAME
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-flash-latest",
                        google_api_key=google_api_key,
                        temperature=0.1
                    )

                    # ✅ BETTER PROMPT
                    prompt = f"""
Та бол Central Test компанийн албан ёсны AI туслах.

Дүрэм:
- Зөвхөн доорх мэдээлэлд үндэслэж хариул
- Таамаглах, нэмэлт зүйл зохиохыг хориглоно
- Хариултыг товч, тодорхой өг

Мэдээлэл:
{context}

Асуулт:
{query}

Хэрэв мэдээлэл байхгүй бол:
"Уучлаарай, миний мэдээллийн санд энэ талаар мэдээлэл алга байна." гэж хариул.
"""

                    response = llm.invoke(prompt)

                    # OUTPUT
                    st.markdown("### 🤖 AI хариулт")
                    st.write(response.content)

                    # SOURCES
                    with st.expander("📚 Ашигласан мэдээлэл"):
                        st.info(context)

            except Exception as e:
                st.error(f"❌ Алдаа гарлаа: {e}")
