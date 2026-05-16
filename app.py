# =====================================================
# IMPORTS
# =====================================================

import tempfile
import streamlit as st

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_groq import ChatGroq

from langchain_community.vectorstores import (
    FAISS
)

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import (
    getSampleStyleSheet
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

    with open(
        "style.css",
        encoding="utf-8"
    ) as f:

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

        tmp.write(
            uploaded_file.read()
        )

        caminho_pdf = tmp.name

    loader = PyPDFLoader(
        caminho_pdf
    )

    return loader.load()


def dividir_texto(documentos):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(
        documentos
    )


def criar_base_vetorial(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(
        chunks,
        embeddings
    )


# =====================================================
# PDF
# =====================================================

def gerar_pdf(resumo):

    caminho_pdf = "resumo_contrato.pdf"

    doc = SimpleDocTemplate(
        caminho_pdf
    )

    styles = getSampleStyleSheet()

    elementos = []

    titulo = Paragraph(
        "Resumo Jurídico - LegalSmart",
        styles["Title"]
    )

    texto = Paragraph(
        resumo,
        styles["BodyText"]
    )

    elementos.append(titulo)

    elementos.append(
        Spacer(1, 20)
    )

    elementos.append(texto)

    doc.build(elementos)

    return caminho_pdf

# =====================================================
# HERO
# =====================================================

hero_html = """
<div class="hero">

    <h1>⚖️ LegalSmart</h1>

    <p class="hero-text">
        Seu Paralegal inteligente
        para análise contratual
    </p>

</div>
"""

st.markdown(
    hero_html,
    unsafe_allow_html=True
)


# =====================================================
# API
# =====================================================

groq_key = st.secrets.get(
    "GROQ_API_KEY",
    ""
)


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

    if "vectorstore" not in st.session_state:

        with st.spinner(
            "Paralegal analisando..."
        ):

            docs = carregar_pdf(
                uploaded_file
            )

            chunks = dividir_texto(
                docs
            )

            vectorstore = criar_base_vetorial(
                chunks
            )

            st.session_state.vectorstore = (
                vectorstore
            )

        st.success(
            "Contrato processado."
        )


# =====================================================
# RESUMO PDF
# =====================================================

if "vectorstore" in st.session_state:

    gerar_resumo = st.button(
        "📄 Gerar Resumo PDF",
        use_container_width=True
    )

    if gerar_resumo:

        llm = ChatGroq(
            groq_api_key=groq_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )

        docs = (
            st.session_state.vectorstore
            .similarity_search(
                "Faça um resumo jurídico completo deste contrato",
                k=8
            )
        )

        contexto = "\n\n".join([
            doc.page_content
            for doc in docs
        ])

        prompt = f"""
        Gere um resumo jurídico
        profissional e objetivo.

        NÃO:
        - invente informações
        - assine mensagens
        - escreva como e-mail

        Contexto:
        {contexto}
        """

        resumo = llm.invoke(
            prompt
        ).content

        pdf_path = gerar_pdf(
            resumo
        )

        st.success(
            "Resumo gerado com sucesso!"
        )

        with open(
            pdf_path,
            "rb"
        ) as file:

            st.download_button(
                label="⬇️ Download PDF",
                data=file,
                file_name="resumo_contrato.pdf",
                mime="application/pdf",
                use_container_width=True
            )


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
        f"""
        <div class="{classe}">
            {msg["content"]}
        </div>
        """,
        unsafe_allow_html=True
    )


pergunta = st.chat_input(
    "Pergunte algo ao Paralegal..."
)


# =====================================================
# RESPOSTA IA
# =====================================================

if (
    pergunta
    and "vectorstore" in st.session_state
):

    st.session_state.messages.append({

        "role": "user",
        "content": pergunta

    })

    llm = ChatGroq(
        groq_api_key=groq_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    docs = (
        st.session_state.vectorstore
        .similarity_search(
            pergunta,
            k=4
        )
    )

    contexto = "\n\n".join([
        doc.page_content
        for doc in docs
    ])

    prompt = f"""
    Você é um Paralegal jurídico.

    Responda de forma:
    - objetiva
    - moderna
    - clara

    NÃO:
    - assine mensagens
    - diga "atenciosamente"
    - invente nomes
    - escreva como e-mail

    Use apenas o contexto abaixo.

    Contexto:
    {contexto}

    Pergunta:
    {pergunta}
    """

    resposta = llm.invoke(
        prompt
    ).content

    st.session_state.messages.append({

        "role": "assistant",
        "content": resposta

    })

    st.rerun()
