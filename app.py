# =====================================================
# IMPORTS
# =====================================================

import streamlit as st
import re
import pdfplumber
from gtts import gTTS

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


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
# FUNÇÕES
# =====================================================

def carregar_pdf(uploaded_file):
    texto = ""

    with pdfplumber.open(uploaded_file) as pdf:
        for pagina in pdf.pages:
            conteudo = pagina.extract_text()

            if conteudo:
                texto += conteudo + "\n"

    return texto[:50000]


def dividir_texto(texto):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1250,
        chunk_overlap=300
    )

    return splitter.create_documents([texto])


def criar_base_vetorial(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embeddings)


def limpar_texto_pdf(texto):
    texto = re.sub(r"\*\*", "", texto)
    texto = re.sub(r"\*", "", texto)

    texto = texto.replace(
        "\n",
        "<br/><br/>"
    )

    return texto


def gerar_pdf(resumo):
    caminho_pdf = "resumo_contrato.pdf"

    doc = SimpleDocTemplate(caminho_pdf)
    styles = getSampleStyleSheet()

    elementos = []

    titulo = Paragraph(
        "Resumo Jurídico - LegalSmart",
        styles["Title"]
    )

    texto_limpo = limpar_texto_pdf(resumo)

    texto = Paragraph(
        texto_limpo,
        styles["BodyText"]
    )

    elementos.append(titulo)
    elementos.append(Spacer(1, 20))
    elementos.append(texto)

    doc.build(elementos)

    return caminho_pdf


def gerar_audio(resumo):
    caminho_audio = "resumo_contrato.mp3"

    tts = gTTS(
        text=resumo,
        lang="pt-br"
    )

    tts.save(caminho_audio)

    return caminho_audio


def gerar_resumo_contrato(modo="texto"):
    llm = ChatGroq(
        groq_api_key=groq_key,
        model_name="llama-3.1-8b-instant",
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

    if modo == "audio":
        prompt = f"""
        Gere um resumo jurídico profissional, objetivo e claro.

        O texto será convertido em áudio, então escreva de forma natural para ser ouvido.

        NÃO:
        - invente informações
        - assine mensagens
        - escreva como e-mail

        Contexto:
        {contexto}
        """
    else:
        prompt = f"""
        Gere um resumo jurídico profissional e objetivo.

        NÃO:
        - invente informações
        - assine mensagens
        - escreva como e-mail

        Contexto:
        {contexto}
        """

    return llm.invoke(prompt).content


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
    st.session_state.pop("vectorstore", None)
    st.session_state.pop("arquivo_atual", None)


# =====================================================
# PROCESSAR PDF
# =====================================================

if uploaded_file and groq_key:

    if (
        "arquivo_atual" not in st.session_state
        or st.session_state.arquivo_atual != uploaded_file.name
    ):

        with st.spinner("Paralegal analisando..."):

            texto = carregar_pdf(uploaded_file)
            chunks = dividir_texto(texto)
            vectorstore = criar_base_vetorial(chunks)

            st.session_state.vectorstore = vectorstore
            st.session_state.arquivo_atual = uploaded_file.name

        st.success("Contrato processado.")


# =====================================================
# FERRAMENTAS
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

    col1, col2 = st.columns(2)

    with col1:
        gerar_resumo_pdf = st.button(
            "Gerar Resumo PDF",
            use_container_width=True
        )

    with col2:
        gerar_resumo_audio = st.button(
            "Gerar Resumo em Áudio",
            use_container_width=True
        )

    if gerar_resumo_pdf:

        with st.spinner("Gerando resumo em PDF..."):

            resumo = gerar_resumo_contrato(
                modo="texto"
            )

            pdf_path = gerar_pdf(resumo)

        st.success("Resumo em PDF gerado com sucesso!")

        with open(pdf_path, "rb") as file:
            st.download_button(
                label="Download PDF",
                data=file,
                file_name="resumo_contrato.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    if gerar_resumo_audio:

        with st.spinner("Gerando resumo em áudio..."):

            resumo_audio = gerar_resumo_contrato(
                modo="audio"
            )

            audio_path = gerar_audio(resumo_audio)

        st.success("Resumo em áudio gerado com sucesso!")

        st.audio(
            audio_path,
            format="audio/mp3"
        )

        with open(audio_path, "rb") as file:
            st.download_button(
                label="Download Áudio MP3",
                data=file,
                file_name="resumo_contrato.mp3",
                mime="audio/mpeg",
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

if pergunta and "vectorstore" in st.session_state:

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

    resposta = llm.invoke(prompt).content

    st.session_state.messages.append({
        "role": "assistant",
        "content": resposta
    })

    st.rerun()
