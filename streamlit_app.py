import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

st.set_page_config(page_title="Central Test AI", page_icon="🤖")

google_api_key = st.secrets["AIzaSyCpcv6TOO4E-pe7BgZYMA_BJeOw6y-zRdI"]
pinecone_api_key = st.secrets["pcsk_65ZC2g_4k2eyNb9EAaAQ4g3rfFdHFKqbTDmKRGMxfgTV5NLLjaTYBFiK154icTn4ggGXaM"]

index_name = "centralai"

st.title("🤖 Central Test AI Assistant")
st.markdown("---")

@st.cache_resource
def load_vectorstore():

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )

    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key
    )

    return vectorstore

vectorstore = load_vectorstore()

query = st.text_input("Асуулт асууна уу:")

if query:

    with st.spinner("AI бодож байна..."):

        docs = vectorstore.similarity_search(query, k=5)

        context = "\n\n".join([doc.page_content for doc in docs])

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=google_api_key,
            temperature=0.1
        )

        prompt = f"""
Та бол Central Test компанийн албан ёсны AI туслах.

Доорх мэдээлэлд тулгуурлан асуултад монгол хэлээр тодорхой хариул.

Мэдээлэл:
{context}

Асуулт:
{query}

Хэрэв мэдээлэл байхгүй бол:
"Мэдээлэл алга" гэж хариул.
"""

        response = llm.invoke(prompt)

        st.markdown("### 🤖 Хариулт")
        st.write(response.content)

        with st.expander("Context"):

            for doc in docs:
                st.write(doc.page_content[:500])
