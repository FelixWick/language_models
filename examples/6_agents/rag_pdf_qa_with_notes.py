import os
import sys
import json
import tempfile
from typing import List, Dict, Tuple
import numpy as np
import pdfplumber
import re
import faiss
from sentence_transformers import SentenceTransformer

from pdf2image import convert_from_path
from PIL import ImageDraw
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

PDF_PATH = "data/Grundlagen der Elektrotechnik_Skript.pdf"

EMBED_MODEL_NAME = "BAAI/bge-m3"
embed_model = SentenceTransformer(EMBED_MODEL_NAME)

DEFAULT_CHUNK_SIZE = 800
DEFAULT_MIN_CHUNK_LEN = 200
TOP_K = 5

FAISS_INDEX_PATH = "./faiss_index_ollama.bin"
METADATA_PATH = "./chunks_metadata_ollama.json"

try:
    import ollama
    HAS_OLLAMA_PY = True
except Exception:
    HAS_OLLAMA_PY = False
    import requests

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:12b")
OLLAMA_BASE = os.environ.get("OLLAMA_BASE", "http://localhost:11434")


# -----------------------------------------
# PDF → Seiten extrahieren
# -----------------------------------------
def pdf_to_pages(pdf_path: str) -> List[str]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)
    return pages


# -----------------------------------------
# Cleaning
# -----------------------------------------
def clean_pdf_text(text: str) -> str:
    if not text or not text.strip():
        return ""

    text = re.sub(r"-\s*\n\s*", "", text)
    text = text.replace("\r", "")

    lines = text.split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()

        if re.match(r"^\d{1,3}$", stripped):
            continue

        cleaned.append(stripped)

    final = []
    last_empty = False
    for l in cleaned:
        if l == "":
            if not last_empty:
                final.append("")
            last_empty = True
        else:
            final.append(l)
            last_empty = False

    text = "\n".join(final).strip()
    text = re.sub(r"\s{2,}", " ", text)
    return text


# -----------------------------------------
# Chunking
# -----------------------------------------
def chunk_text_with_meta(pages: List[str],
                         target_chunk_size: int = DEFAULT_CHUNK_SIZE,
                         min_chunk_len: int = DEFAULT_MIN_CHUNK_LEN) -> List[Dict]:
    chunks = []

    for p_idx, page_text in enumerate(pages, start=1):
        cleaned = clean_pdf_text(page_text)
        if not cleaned:
            continue

        paragraphs = cleaned.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            continue

        current_chunk = []
        current_len = 0

        def flush(force=False):
            nonlocal chunks, current_chunk, current_len, p_idx
            if not current_chunk:
                return
            text = "\n".join(current_chunk).strip()
            if len(text) < min_chunk_len and not force:
                return
            chunks.append({
                "page": p_idx,
                "text": text,
            })
            current_chunk = []
            current_len = 0

        for para in paragraphs:
            if current_len + len(para) > target_chunk_size:
                flush(force=True)
            current_chunk.append(para)
            current_len += len(para)
            if current_len >= min_chunk_len:
                flush()

        flush(force=True)

    return chunks


# -----------------------------------------
# Embeddings
# -----------------------------------------
def embed_texts(texts: List[str], is_query=False) -> np.ndarray:
    formatted = [("query: " + t) if is_query else t for t in texts]
    embs = embed_model.encode(
        formatted,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=64
    )
    return embs.astype("float32")


# -----------------------------------------
# FAISS
# -----------------------------------------
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index

def save_faiss(index, path):
    faiss.write_index(index, path)

def load_faiss(path):
    return faiss.read_index(path)


# -----------------------------------------
# Index → laden oder neu erstellen
# -----------------------------------------
def build_or_load_index(force_rebuild=False):
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH) and not force_rebuild:
        print("Load existing FAISS index and meta data ...")
        index = load_faiss(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return index, metadata

    print("Extracting text from PDF ...")
    pages = pdf_to_pages(PDF_PATH)
    print(f"{len(pages)} pages extracted.")
    print("Generating chunks ...")

    chunks = chunk_text_with_meta(pages)
    texts = [c["text"] for c in chunks]

    print(f"{len(texts)} chunks generated. Generate embeddings ...")
    embeddings = embed_texts(texts)
    index = build_faiss_index(embeddings)

    print("Store index and meta data ...")
    save_faiss(index, FAISS_INDEX_PATH)

    meta = [
        {
            "page": c["page"],
            "text": c["text"][:300]
        }
        for c in chunks
    ]

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return index, meta


# -----------------------------------------
# Retrieval
# -----------------------------------------
def retrieve(query: str, index: faiss.IndexFlatIP, metadata: List[Dict], top_k: int=TOP_K):
    q_emb = embed_texts([query], is_query=True)
    D, I = index.search(q_emb, top_k)
    results = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        results.append((metadata[idx], float(score)))
    return results


# =============================================================================
#                               TOOL: annotate_pdf
# =============================================================================

def annotate_pdf(pdf_path: str, page: int, note: str) -> str:
    """
    Tool-Funktion: Erzeugt eine annotierte Kopie des PDFs.
    Auf Seite 'page' wird oben eine gelbe Notiz eingefügt.
    """
    images = convert_from_path(pdf_path, dpi=150)

    if page < 1 or page > len(images):
        raise ValueError(f"Seite {page} existiert nicht.")

    # Bild der Zielseite
    img = images[page-1].convert("RGBA")
    draw = ImageDraw.Draw(img)

    # gelbe Box
    box_h = 80
    draw.rectangle([(0, 0), (img.width, box_h)], fill=(255, 255, 0, 180))

    # Text
    draw.text((10, 10), note, fill=(0, 0, 0))

    # speichern als temp
    modified_img_path = tempfile.mktemp(suffix=".png")
    img.save(modified_img_path)

    # neue PDF erstellen
    out_pdf = pdf_path.replace(".pdf", "_annotated.pdf")
    c = canvas.Canvas(out_pdf, pagesize=letter)

    for i, original_img in enumerate(images, start=1):
        if i == page:
            tmp_path = modified_img_path
        else:
            tmp_path = tempfile.mktemp(suffix=".png")
            original_img.save(tmp_path)

        c.drawImage(ImageReader(tmp_path), 0, 0, width=letter[0], height=letter[1])
        c.showPage()

    c.save()
    return out_pdf


# =============================================================================
#                       Agentische Tool-Erkennung (LLM)
# =============================================================================

def check_for_tool_call(user_input: str):
    m = re.search(r"<<tool:annotate_pdf>>(.*?)<<endtool>>", user_input, re.S)
    if not m:
        return None

    try:
        data = json.loads(m.group(1))
        # defensive patch
        if "page" in data:
            try:
                data["page"] = int(data["page"])
            except:
                pass
        if "note" in data:
            data["note"] = str(data["note"])
        return ("annotate_pdf", data)
    except Exception:
        return None


# -----------------------------------------
# Ollama: Generator
# -----------------------------------------
def generate_via_ollama_sdk(prompt: str, model: str = OLLAMA_MODEL):
    if not HAS_OLLAMA_PY:
        raise RuntimeError("Python package 'ollama' not installed.")
    resp = ollama.generate(model=model, prompt=prompt)
    if isinstance(resp, dict) and "response" in resp:
        return resp["response"]
    return str(resp.response) if hasattr(resp, "response") else str(resp)

def generate_via_ollama_rest(prompt: str, model: str = OLLAMA_MODEL):
    url = f"{OLLAMA_BASE}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "response" in data:
        return data["response"]
    return json.dumps(data, ensure_ascii=False)


# -----------------------------------------
# Prompt
# -----------------------------------------
def build_prompt(query: str, retrieved: List[Tuple[Dict, float]]):
    context_blocks = []
    for meta, score in retrieved:
        block = f"[Seite {meta['page']}] {meta.get('text','')}"
        context_blocks.append(block)
    context = "\n\n".join(context_blocks)

    ### Agentische Anweisung (NEU)
    tool_instruction = (
        "AGENT-INSTRUKTIONEN:\n"
        "Du verfügst über EIN Werkzeug:\n"
        "- annotate_pdf(page, note): Fügt eine Notiz in das PDF ein.\n\n"
        "WANN DARFST DU DAS TOOL BENUTZEN?\n"
        "- Nur wenn der Benutzer EXPLIZIT sagt, dass er eine Notiz schreiben, hinzufügen, einfügen,\n"
        "  markieren oder kommentieren möchte.\n"
        "- Wenn KEINE solche Anweisung erteilt wird, darfst du das Tool NICHT benutzen.\n"
        "- Beantworte normale Fragen ausschließlich mit einer normalen Antwort.\n\n"
        "WIE NUTZT DU DAS TOOL?\n"
        "Gib als Antwort ausschließlich folgenden Block zurück:\n"
        "<<tool:annotate_pdf>>{\"page\": SEITE, \"note\": \"TEXT\"}<<endtool>>\n\n"
    )

    prompt = (
        tool_instruction +
        "Du bist ein präziser Tutor für Elektrotechnik. "
        "Benutze nur den untenstehenden Dokumentkontext, "
        "antworte knapp und nenne die Seiten in eckigen Klammern. "
        "Wenn die Antwort nicht im Kontext steht, sag, dass du es nicht weißt.\n\n"
        f"KONTEXT:\n{context}\n\nFRAGE: {query}\n\nANTWORT:"
    )
    return prompt


# -----------------------------------------
# Antwortgenerator
# -----------------------------------------
def generate_answer_with_ollama(query: str, index, metadata):
    # 1. TOOL-CALL direkt vom User
    tool = check_for_tool_call(query)
    if tool:
        tool_name, params = tool
        if tool_name == "annotate_pdf":
            page = int(params["page"])
            note = str(params["note"])
            out = annotate_pdf(PDF_PATH, page, note)
            return f"Notiz gespeichert in: {out}", [params["page"]]

    # 2. Normales RAG
    retrieved = retrieve(query, index, metadata, top_k=TOP_K)
    if not retrieved:
        return "No relevant text passages found.", []

    prompt = build_prompt(query, retrieved)

    # 3. Ask LLM
    try:
        if HAS_OLLAMA_PY:
            ans = generate_via_ollama_sdk(prompt, OLLAMA_MODEL)
        else:
            ans = generate_via_ollama_rest(prompt, OLLAMA_MODEL)
    except Exception as e:
        return f"Problem with Ollama: {e}", []

    # 4. TOOL-CALL aus LLM ausführen
    # Werkzeugnutzung nur erlauben, wenn die Nutzerfrage es eindeutig verlangt
    if "notiz" in query.lower() or "note" in query.lower() or "kommentar" in query.lower():
        tool = check_for_tool_call(ans)
        if tool:
            tool_name, params = tool
            if tool_name == "annotate_pdf":
                page = int(params["page"])
                note = str(params["note"])
                out = annotate_pdf(PDF_PATH, page, note)
                return f"Notiz gespeichert in: {out}", [page]
    else:
        # Falls das LLM fälschlich einen Tool-Call produziert → ignorieren, normal antworten
        # (Tool-Blöcke aus der Antwort entfernen)
        ans = re.sub(r"<<tool:annotate_pdf>>.*?<<endtool>>", "", ans, flags=re.S)

    # 5. Rückgabe
    pages = sorted({r[0]['page'] for r in retrieved})
    return ans, pages


# -----------------------------------------
# Interaktiver Loop
# -----------------------------------------
def interactive_loop(index, metadata):
    print("\nRAG + Agent für PDF-Notizen bereit. 'exit' zum Beenden.")
    while True:
        q = input("\nFrage: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        ans, pages = generate_answer_with_ollama(q, index, metadata)
        print("\n--- Antwort ---\n")
        print(ans)
        if pages:
            print(f"\n(Quellen: Seiten {', '.join(map(str,pages))} aus '{os.path.basename(PDF_PATH)}')")


# -----------------------------------------
# MAIN
# -----------------------------------------
if __name__ == "__main__":
    force_rebuild = "--rebuild" in sys.argv
    index, metadata = build_or_load_index(force_rebuild)
    interactive_loop(index, metadata)