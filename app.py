import tempfile
import streamlit as st

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)


# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="LegalSmart",
    page_icon="⚖️",
    layout="wide"
)


# =====================================================
# CSS
# =====================================================

def load_css():
    with open("style.css", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()


# =====================================================
# FUNÇÕES IA
# =====================================================

def carregar_pdf(uploaded_file):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        tmp.write(uploaded_file.read())
        caminho_pdf = tmp.name

    loader = PyPDFLoader(caminho_pdf)

    return loader.load()


def dividir_texto(documentos):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(documentos)


def criar_base_vetorial(chunks, api_key):

    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

    return FAISS.from_documents(
        chunks,
        embeddings
    )


# =====================================================
# HERO
# =====================================================

st.markdown("""
<div class="hero">
    <h1>⚖️ LegalSmart</h1>
    <p>
        Seu Paralegal inteligente para análise contratual
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# API
# =====================================================

groq_key = st.secrets["GROQ_API_KEY"]
# =====================================================
# UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload do contrato",
    type=["pdf"]
)


# =====================================================
# PROCESSAR
# =====================================================

if uploaded_file and groq_key:

    with st.spinner("Paralegal analisando..."):

        docs = carregar_pdf(uploaded_file)

        chunks = dividir_texto(docs)

        vectorstore = criar_base_vetorial(
            chunks,
            groq_key
        )

        st.session_state.vectorstore = vectorstore

    st.success("Contrato processado.")



# =====================================================
# CHAT
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:

    classe = (
        "message-user"
        if msg["role"] == "user"
        else "message-ai"
    )

    st.markdown(
        f'<div class="{classe}">{msg["content"]}</div>',
        unsafe_allow_html=True
    )


pergunta = st.chat_input(
    "Pergunte algo ao Paralegal..."
)


if pergunta and "vectorstore" in st.session_state:

    st.session_state.messages.append({
        "role": "user",
        "content": pergunta
    })

    llm = ChatGroq(
        groq_api_key=groq_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    docs = st.session_state.vectorstore.similarity_search(
        pergunta,
        k=4
    )

    contexto = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
    Você é um Paralegal jurídico profissional.

    Responda de forma objetiva, moderna e clara.

    NÃO:
    - assine mensagens
    - diga "atenciosamente"
    - diga "prezado advogado"
    - invente nomes
    - finalize como e-mail

    Responda apenas com a informação jurídica.

    Use apenas o contexto abaixo.


    Contexto:
    {contexto}

    Pergunta:
    {pergunta}
    """

    resposta = llm.invoke(prompt).content

    st.session_state.messages.append({
        "role": "assistant",
        "content": resposta
    })

    st.rerun()