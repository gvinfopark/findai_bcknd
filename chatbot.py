import os
import gc
import re
import logging
import warnings
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict, Literal
from flask import session
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from toc import TOC_ENTRIES

# --------------------------
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")
os.environ["USER_AGENT"] = "Mozilla/5.0"

Section = Literal["beginning", "middle", "end"]

# --------------------------
class GraphState(TypedDict):
    question: str
    context: List[Document]
    answer: str
    history: List[str]

class Search(TypedDict):
    query: str
    section: Section

session_user_profiles = {}

# --------------------------
class MistralChatbot:
    def __init__(self):
        self.model_name = "mistral-small"
        self.embeddings_model = MistralAIEmbeddings()
        self.llm = init_chat_model(self.model_name, model_provider="mistralai")
        self.vector_store_path = "vector_db_faiss"
        
        # Use LangGraph's built-in MemorySaver instead of custom checkpointer
        self.checkpointer = MemorySaver()

        self.vector_store = None
        self.setup_rag()

        # ✅ Compile LangGraph
        self.compiled_graph = self.compile_graph()

    # --------------------------
    def setup_rag(self):
        page_to_unit = {}
        for i, entry in enumerate(TOC_ENTRIES):
            start = entry["page"]
            end = TOC_ENTRIES[i+1]["page"] if i+1 < len(TOC_ENTRIES) else 9999
            for p in range(start, end):
                page_to_unit[p] = {"unit": entry["unit"], "lesson_title": entry["lesson_title"]}

        if os.path.exists(os.path.join(self.vector_store_path, "index.faiss")):
            logger.info("✅ Loading FAISS vector store from disk...")
            self.vector_store = FAISS.load_local(
                folder_path=self.vector_store_path,
                embeddings=self.embeddings_model,
                allow_dangerous_deserialization=True
            )
        else:
            logger.info("📚 Creating FAISS vector store from PDFs...")
            loader = DirectoryLoader("pdf_books", glob="*.pdf", loader_cls=PyPDFLoader)
            raw_docs = loader.load()
            for i, doc in enumerate(raw_docs):
                doc.metadata["page_number"] = doc.metadata.get("page", i+1)
                source = doc.metadata.get("source", "unknown.pdf")
                doc.metadata["source"] = os.path.basename(source)
                page_num = doc.metadata["page_number"]
                unit_info = page_to_unit.get(page_num, {})
                doc.metadata["unit"] = unit_info.get("unit", "")
                doc.metadata["lesson_title"] = unit_info.get("lesson_title", "")

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            all_splits = splitter.split_documents(raw_docs)
            self.vector_store = FAISS.from_documents(all_splits, self.embeddings_model)
            os.makedirs(self.vector_store_path, exist_ok=True)
            self.vector_store.save_local(self.vector_store_path)
            logger.info("✅ Vector store created and saved.")

    # --------------------------
    def analyze_query(self, state: GraphState) -> GraphState:
        try:
            structured_llm = self.llm.with_structured_output(Search)
            result = structured_llm.invoke(state["question"])
            state["history"].append(f"Analyzed query: {result}")
            return state
        except Exception as e:
            state["history"].append(f"Query fallback: {e}")
            return state

    # --------------------------
    def extract_metadata(self, question: str) -> Dict[str, Any]:
        page_match = re.search(r'page\s+(\d+)', question, re.IGNORECASE)
        book_match = re.search(r'(\d+)(?:st|nd|rd|th)?\s*book', question, re.IGNORECASE)
        unit_match = re.search(r'(?:unit|lesson)[\s:-]*(\d+)', question, re.IGNORECASE)
        ordinal_unit = re.search(
            r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+(unit|lesson)',
            question, re.IGNORECASE
        )

        unit_number = None
        if unit_match:
            unit_number = int(unit_match.group(1))
        elif ordinal_unit:
            ordinals = {'first':1,'second':2,'third':3,'fourth':4,'fifth':5,
                        'sixth':6,'seventh':7,'eighth':8,'ninth':9,'tenth':10}
            unit_number = ordinals.get(ordinal_unit.group(1).lower())
        return {"page_number": int(page_match.group(1)) if page_match else None,
                "book_hint": book_match.group(1) if book_match else None,
                "unit_number": unit_number}

    # --------------------------
    def retrieve(self, state: GraphState) -> GraphState:
        meta = self.extract_metadata(state["question"])
        all_docs = self.vector_store.similarity_search_with_score(state["question"], k=25)
        matched_docs = [doc for doc, _ in all_docs[:5]]
        state["context"] = matched_docs
        state["history"].append(f"Retrieved {len(matched_docs)} docs")
        return state

    # --------------------------
    def truncate_text(self, text: str, max_chars: int = 4000) -> str:
        return text[-max_chars:] if len(text) > max_chars else text

    # --------------------------
    def generate(self, state: GraphState) -> GraphState:
        with open("prompt.txt", encoding="utf-8") as f:
            prompt = f.read()
        prompt = ChatPromptTemplate.from_template(prompt)

        docs_content = self.truncate_text("\n\n".join(doc.page_content for doc in state["context"]))
        history_text = self.truncate_text("\n".join(state["history"]))
        user_profile = session_user_profiles.get(session.get("session_id"), "")

        prompt_values = prompt.invoke({
            "question": state["question"],
            "context": docs_content,
            "history": history_text,
            "user_profile": user_profile,
        })

        response = self.llm.invoke(prompt_values.to_messages())
        state["answer"] = response.content
        state["history"].append(f"Assistant: {response.content}")
        return state

    # --------------------------
    def compile_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("analyze_query", self.analyze_query)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("generate", self.generate)
        graph.add_edge(START, "analyze_query")
        graph.add_edge("analyze_query", "retrieve")
        graph.add_edge("retrieve", "generate")
        return graph.compile(checkpointer=self.checkpointer)

    # --------------------------
    def cleanup_memory(self):
        gc.collect()