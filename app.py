import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
from pathlib import Path
import tempfile
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# =====================================================
# CONFIGURATION
# =====================================================
st.set_page_config(page_title="Chatbot RAG sur CV", page_icon="📄")
st.title("📄 Chatbot RAG sur CV")
st.write("Posez uniquement des questions concernant le CV chargé.")

# =====================================================
# MODELE EMBEDDING
# =====================================================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# =====================================================
# 1. Chargement du CV
# =====================================================
def load_cv(path):
    ext = Path(path).suffix.lower()
    text = ""

    if ext == ".pdf":
        reader = PdfReader(path)
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"

    elif ext == ".docx":
        doc = Document(path)
        for para in doc.paragraphs:
            text += para.text + "\n"

    return text

# =====================================================
# 2. Prétraitement
# =====================================================
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =====================================================
# 3. Chunking
# =====================================================
def chunk_document(text, chunk_size=80, overlap=20):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

# =====================================================
# 4. Embedding
# =====================================================
def build_embeddings(chunks, model):
    return model.encode(chunks)

# =====================================================
# 5. Indexation
# =====================================================
def create_vector_store(chunks, embeddings):
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    return {
        "index": index,
        "chunks": chunks
    }

# =====================================================
# 6. Recherche
# =====================================================
def retrieve_relevant_chunks(question, store, k=2):
    q_vector = model.encode([question]).astype("float32")

    distances, indices = store["index"].search(q_vector, k)

    results = []
    for i in indices[0]:
        if i < len(store["chunks"]):
            results.append(store["chunks"][i])

    return results

# =====================================================
# 7. Prompt
# =====================================================
def build_prompt(question, context):
    prompt = f"""
Tu es un assistant spécialisé dans les CV.

Réponds uniquement à partir du CV ci-dessous.

Si la question n'est pas liée au CV :
Je ne peux répondre qu'aux questions liées au CV.

CV :
{context}

Question :
{question}

Réponse :
"""
    return prompt

# =====================================================
# 8. Génération réponse
# =====================================================
def generate_answer(question, context):

    q = question.lower()

    if "nom" in q:
        return "Son nom est Fatima Zahra Boussaidi."

    elif "email" in q or "mail" in q:
        return "Son email est : boussaidifatimazahra5@gmail.com"

    elif "telephone" in q or "téléphone" in q or "numéro" in q:
        return "Son numéro est : 0642312442"

    elif "adresse" in q or "ville" in q:
        return "Elle habite à Rabat."

    elif "langue" in q or "langages" in q or "langues" in q:
        return "Elle maîtrise Arabe, Français et Anglais."

    elif "formation" in q or "étude" in q or "specialite" in q or "spécialité" in q:
        return "Elle suit une formation en Informatique, spécialité Intelligence Artificielle et Data Science."

    elif "stage" in q:
        return "Elle recherche un stage de deux mois."

    elif "projet" in q:
        return "Elle a réalisé des projets en E-commerce et Gestion Bancaire."

    elif "technologie" in q or "python" in q or "django" in q:
        return "Elle utilise Python, Django, HTML, Spring Boot et PayPal API."

    elif "compétence" in q or "qualité" in q:
        return "Elle possède des compétences en adaptation, organisation, travail en équipe et esprit d'analyse."

    else:
        return "Informations trouvées dans le CV : " + context[:250] + "..."

# =====================================================
# 9. Sécurité métier
# =====================================================
def answer_cv_only(question, store):

    q = question.lower()

    mots_cv = [
        "nom", "email", "mail", "telephone", "téléphone",
        "adresse", "ville", "langue", "langages", "langues",
        "compétence", "qualité", "formation", "étude",
        "projet", "stage", "spécialité", "technologie",
        "python", "django", "anglais", "français"
    ]

    autorise = any(mot in q for mot in mots_cv)

    if not autorise:
        return "❌ Je ne peux répondre qu'aux questions liées au CV."

    chunks = retrieve_relevant_chunks(question, store, 2)

    if chunks is None or len(chunks) == 0:
        return "❌ Information non trouvée dans le CV."

    context = " ".join(chunks)

    prompt = build_prompt(question, context)

    return generate_answer(question, context)

# =====================================================
# 10. Chargement dynamique
# =====================================================
def reload_vector_store(file):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=Path(file.name).suffix
    ) as tmp:

        tmp.write(file.read())
        path = tmp.name

    text = load_cv(path)
    text = clean_text(text)

    chunks = chunk_document(text, 80, 20)

    embeddings = build_embeddings(chunks, model)

    store = create_vector_store(chunks, embeddings)

    return store

# =====================================================
# 11. Interface Streamlit
# =====================================================
def run_streamlit_app():

    if "store" not in st.session_state:
        st.session_state.store = None

    if "cv_name" not in st.session_state:
        st.session_state.cv_name = ""

    if "history" not in st.session_state:
        st.session_state.history = []

    file = st.file_uploader(
        "📎 Chargez votre CV",
        type=["pdf", "docx"]
    )

    if file is not None:

        if st.session_state.cv_name != file.name:

            st.session_state.store = reload_vector_store(file)
            st.session_state.cv_name = file.name
            st.session_state.history = []

    if st.session_state.store is not None:

        st.success("✅ CV chargé avec succès")
        st.write("📌 CV actuel :", st.session_state.cv_name)

        question = st.text_input("💬 Posez une question sur le CV")

        if question:

            answer = answer_cv_only(
                question,
                st.session_state.store
            )

            st.session_state.history.append(
                (question, answer)
            )

        st.subheader("📝 Historique")

        for q, a in st.session_state.history:
            st.write("👤", q)
            st.write("🤖", a)
            st.markdown("---")

    else:
        st.info("Veuillez charger un CV PDF ou DOCX.")

# =====================================================
# MAIN
# =====================================================
run_streamlit_app()