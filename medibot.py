import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Optional: .env support
from dotenv import load_dotenv
load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"

# ---------- Helpers ----------
def _extract_page(meta: dict):
    """Best-effort page number extraction across common loaders."""
    if not meta:
        return None
    # Common keys seen in PyPDF/Unstructured/Splade loaders, etc.
    return (
        meta.get("page")
        or meta.get("page_number")
        or (meta.get("loc", {}) or {}).get("page")
        or meta.get("pdf_page")
        or None
    )

def _extract_source(meta: dict):
    """Best-effort source/path extraction."""
    if not meta:
        return None
    return (
        meta.get("source")
        or meta.get("file_path")
        or meta.get("path")
        or meta.get("document_id")
        or meta.get("id")
        or None
    )

@st.cache_resource(show_spinner=False)
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not os.path.exists(DB_FAISS_PATH):
        return None
    # allow_dangerous_deserialization=True is needed when saving/loading with older LC versions
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# ---------- UI ----------
def main():
    st.set_page_config(page_title="Medibot", page_icon="💬")
    st.title("Ask Medibot!")

    # Sidebar controls
    with st.sidebar:
        st.markdown("### Settings")
        top_k = st.slider("Top-K Documents", 1, 10, 3)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.1)
        max_tokens = st.number_input("Max tokens", min_value=128, max_value=4096, value=512, step=64)
        show_sources_by_default = st.checkbox("Expand sources by default", value=False)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            # If a past assistant message had sources, render its expander too
            if m.get("sources"):
                with st.expander("View sources", expanded=False):
                    for i, s in enumerate(m["sources"], start=1):
                        st.markdown(f"**[{i}]** {s['label']}")
                        if s["preview"]:
                            st.caption(s["preview"])

    prompt = st.chat_input("Pass your prompt here")
    if not prompt:
        return

    # Show user message immediately
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error(f"Couldn't find a FAISS index at `{DB_FAISS_PATH}`. Build or place one, then reload.")
            return

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY not set. Add it to your environment or .env file.")
            return

        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

        # Prompt & chains
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
        retriever = vectorstore.as_retriever(search_kwargs={"k": int(top_k)})
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

        # Run
        response = rag_chain.invoke({"input": prompt})

        # Newer LC returns docs under "context"; older patterns may use "source_documents"
        docs = response.get("context") or response.get("source_documents") or []

        answer_text = response.get("answer", "").strip()
        # Render assistant message with a collapsible "Sources" section
        with st.chat_message("assistant"):
            st.markdown(answer_text if answer_text else "_(No answer text returned)_")

            # Build a compact, robust list of source dicts for session storage & display
            sources_for_session = []
            if docs:
                with st.expander("View sources", expanded=show_sources_by_default):
                    for idx, d in enumerate(docs, start=1):
                        meta = getattr(d, "metadata", {}) or {}
                        page = _extract_page(meta)
                        src = _extract_source(meta) or "Unknown source"
                        preview = (getattr(d, "page_content", None) or "")[:400].strip()

                        label_bits = [src]
                        if page is not None:
                            label_bits.append(f"page {page}")
                        label = " — ".join(label_bits)

                        st.markdown(f"**[{idx}]** {label}")
                        if preview:
                            st.caption(preview)

                        sources_for_session.append(
                            {"label": label, "preview": preview}
                        )
            else:
                # Still store an empty list so we don't KeyError later
                sources_for_session = []

        # Persist assistant turn, including sources so they re-render on rerun
        st.session_state.messages.append(
            {"role": "assistant", "content": answer_text, "sources": sources_for_session}
        )

    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
