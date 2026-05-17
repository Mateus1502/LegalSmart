# =====================================================
# IMPORTS
# =====================================================

import tempfile
import streamlit as st
import re
import pdfplumber

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
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

    texto = ""

    with pdfplumber.open(uploaded_file) as pdf:

        for pagina in pdf.pages:

            conteudo = pagina.extract_text()

            if conteudo:

                texto += conteudo + "\n"
                texto = texto[:50000]

    return texto


def dividir_texto(texto):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100
    )

    chunks = splitter.create_documents([texto])

    return chunks


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

    texto_limpo = limpar_texto_pdf(
        resumo
    )

    texto = Paragraph(
        texto_limpo,
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
# LIMPEZA PDF
# =====================================================

def limpar_texto_pdf(texto):

    texto = re.sub(r"\*\*", "", texto)

    texto = re.sub(r"\*", "", texto)

    texto = texto.replace(
        "\n",
        "<br/><br/>"
    )

    return texto

# =====================================================
# HERO
# =====================================================

st.markdown(
    """
    <div class="hero">
        <h1>⚖️ LegalSmart</h1>
        <p class="hero-text">
            Seu Paralegal inteligente para análise contratual
        </p>
    </div>
    """,
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

if uploaded_file is None:

    st.session_state.pop(
        "vectorstore",
        None
    )

    st.session_state.pop(
        "arquivo_atual",
        None
    )


# =====================================================
# PROCESSAR
# =====================================================

if uploaded_file and groq_key:

    if (
        "arquivo_atual" not in st.session_state
        or st.session_state.arquivo_atual != uploaded_file.name
    ):

        with st.spinner(
            "Paralegal analisando..."
        ):

            texto = carregar_pdf(
                uploaded_file
            )

            chunks = dividir_texto(
                texto
            )

            vectorstore = criar_base_vetorial(
                chunks
            )

            st.session_state.vectorstore = (
                vectorstore
            )

            st.session_state.arquivo_atual = (
                uploaded_file.name
            )

        st.success(
            "Contrato processado."
        )

# =====================================================
# RESUMO PDF
# =====================================================

if "vectorstore" in st.session_state:

    st.markdown(
        """
        <div class="section">
            <h2 class="section-title">
                Ferramentas
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    gerar_resumo = st.button(
        "Gerar Resumo PDF",
        use_container_width=True
    )

    if gerar_resumo:

        llm = ChatGroq(
            groq_api_key=groq_key,
            model_name="llama-3.1-8b-instant",
            temperature=0
        )

        docs = (
            st.session_state.vectorstore
            .similarity_search(
                "Faça um resumo jurídico completo deste contrato",
                k=4
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
                label="Download PDF",
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
        model_name="llama-3.1-8b-instant",
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
    Você é um assistente jurídico especializado
    em análise contratual.

    RESPONDA APENAS COM BASE NO CONTEXTO.

    Se a informação não estiver no contrato:
    diga que não foi encontrada.

    NÃO:
    - invente cláusulas
    - invente informações
    - faça suposições
    - escreva como e-mail
    - assine respostas

    Responda de forma:
    - objetiva
    - profissional
    - clara

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
