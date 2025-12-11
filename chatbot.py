import os
import gc
import re
import logging
import warnings
import sqlite3
from typing import List, Dict, Any
from typing_extensions import TypedDict, Annotated, Literal
from flask import session
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langchain_core.prompts import ChatPromptTemplate
import re
from collections import defaultdict
from toc import TOC_ENTRIES

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")
os.environ["USER_AGENT"] = "Mozilla/5.0"

Section = Literal["beginning", "middle", "end"]

class GraphState(TypedDict):
    question: str
    context: List[Document]
    answer: str
    history: List[str]

class Search(TypedDict):
    query: Annotated[str, ...]
    section: Annotated[Section, ...]

session_user_profiles = {}

class MistralChatbot:
    def __init__(self):
        self.model_name = "mistral-small"
        self.embeddings_model = MistralAIEmbeddings()
        self.llm = init_chat_model(self.model_name, model_provider="mistralai")
        self.vector_store_path = "vector_db_faiss"
        self.conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
        self.checkpointer = SqliteSaver(self.conn)
        self.vector_store = None
        self.setup_rag()
        self.compiled_graph = self.compile_graph()

    def setup_rag(self):
        page_to_unit = {}
        
        for i, entry in enumerate(TOC_ENTRIES):
            start = entry["page"]
            end = TOC_ENTRIES[i + 1]["page"] if i + 1 < len(TOC_ENTRIES) else 9999
            for p in range(start, end):
                page_to_unit[p] = {
                    "unit": entry["unit"],
                    "lesson_title": entry["lesson_title"]
                }
                
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
            logger.info(f"📄 Loaded {len(raw_docs)} pages from PDFs")

            for i, doc in enumerate(raw_docs):
                doc.metadata["page_number"] = doc.metadata.get("page", i + 1)
                source = doc.metadata.get("source", "unknown.pdf")
                doc.metadata["source"] = os.path.basename(source)

                match = re.search(r'(\d+)(?:st|nd|rd|th)?', source)
                if match:
                    doc.metadata["book_class"] = match.group(1)

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

    def analyze_query(self, state: GraphState) -> Dict[str, Any]:
        try:
            structured_llm = self.llm.with_structured_output(Search)
            result = structured_llm.invoke(state["question"])
            return {"query": result}
        except Exception as e:
            logger.warning(f"Fallback query parse: {e}")
            return {"query": {"query": state["question"], "section": "middle"}}

    def extract_metadata(self, question: str) -> Dict[str, Any]:
        page_match = re.search(r'page\s+(\d+)', question, re.IGNORECASE)
        book_match = re.search(r'(\d+)(?:st|nd|rd|th)?\s*book', question, re.IGNORECASE)

        # Extract unit number from various formats
        unit_match = re.search(r'(?:unit|lesson)[\s:-]*(\d+)', question, re.IGNORECASE)
        ordinal_unit = re.search(r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+(unit|lesson)', question, re.IGNORECASE)

        unit_number = None
        if unit_match:
            unit_number = int(unit_match.group(1))
        elif ordinal_unit:
            ordinals = {
                'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
                'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10
            }
            unit_number = ordinals.get(ordinal_unit.group(1).lower())

        unit_title_match = re.search(r'(?:on|about|regarding|in)\s+([A-Za-z ]+)', question, re.IGNORECASE)
        unit_title = unit_title_match.group(1).strip() if unit_title_match else None

        marks_type = None
        if re.search(r"\b2[-\s]?mark", question, re.IGNORECASE):
            marks_type = "2 mark"
        elif re.search(r"\b5[-\s]?mark", question, re.IGNORECASE):
            marks_type = "5 mark"
        elif re.search(r"\b1[06][-]?\s?mark", question, re.IGNORECASE):
            marks_type = "10 mark"

        return {
            "page_number": int(page_match.group(1)) if page_match else None,
            "book_hint": book_match.group(1) if book_match else None,
            "unit_number": unit_number,
            "unit_title": unit_title,
            "marks_type": marks_type
        }

    def retrieve(self, state: GraphState) -> Dict[str, Any]:
        meta = self.extract_metadata(state["question"])
        query_text = state["question"]
        book_hint = meta.get("book_hint")
        unit_number = meta.get("unit_number")
        unit_title = meta.get("unit_title")

        logger.info(f"📘 Extracted unit: {unit_number}, title: {unit_title}")

        # Identify unit start page from TOC
        unit_page = None
        TOC_ENTRIES = [
            {"unit": f"Unit {i+1}", "lesson_title": title, "page": page}
            for i, (title, page) in enumerate([
                ("Measurement", 1), ("Force and Pressure", 12), ("Light", 22), ("Heat", 35),
                ("Electricity", 46), ("Sound", 60), ("Magnetism", 72), ("Universe and Space Science", 83),
                ("Matter around us", 94), ("Changes around us", 104), ("Air", 114), ("Atomic Structure", 124),
                ("Water", 139), ("Acids and Bases", 155), ("Chemistry in Everyday Life", 166),
                ("Microorganisms", 180), ("Plant Kingdom", 192), ("Organisation of Life", 205),
                ("Movements in Animals", 218), ("Reaching the Age of Adolescence", 232),
                ("Crop Production and Management", 244), ("Conservation of Plants and Animals", 260),
                ("Libre Office Calc", 276)
            ])
        ]

        for i, entry in enumerate(TOC_ENTRIES):
            if unit_number and entry["unit"] == f"Unit {unit_number}":
                unit_page = entry["page"]
                break
            if unit_title and unit_title.lower() in entry["lesson_title"].lower():
                unit_page = entry["page"]
                break

        all_docs = self.vector_store.similarity_search_with_score(query_text, k=25)
        matched_docs = []

        if unit_page:
            # Find page range of that unit
            end_page = 9999
            for i, entry in enumerate(TOC_ENTRIES):
                if entry["page"] == unit_page and i + 1 < len(TOC_ENTRIES):
                    end_page = TOC_ENTRIES[i + 1]["page"]
                    break

            for doc, _ in all_docs:
                page = doc.metadata.get("page_number", 0)
                if unit_page <= page < end_page:
                    matched_docs.append(doc)

        if matched_docs:
            logger.info(f"✅ Found {len(matched_docs)} matched documents from unit page range.")
            return {
                "context": matched_docs,
                "question": query_text,
                "history": state.get("history", []),
                "answer": ""
            }

        logger.info("⚠️ No unit match — using fallback top results")
        fallback_docs = [doc for doc, _ in all_docs[:5]]
        return {
            "context": fallback_docs,
            "question": query_text,
            "history": state.get("history", []),
            "answer": ""
        }


    def truncate_text(self, text: str, max_chars: int = 4000) -> str:
        return text[-max_chars:] if len(text) > max_chars else text

    def generate(self, state: GraphState) -> Dict[str, Any]:
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
        updated_history = state.get("history", []) + [
            f"User: {state['question']}",
            f"Assistant: {response.content}"
        ]

        return {
            "answer": response.content,
            "question": state["question"],
            "context": state["context"],
            "history": updated_history
        }

    def compile_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("analyze_query", self.analyze_query)
        graph.add_node("retrieve", self.retrieve)
        graph.add_node("generate", self.generate)
        graph.add_edge(START, "analyze_query")
        graph.add_edge("analyze_query", "retrieve")
        graph.add_edge("retrieve", "generate")
        return graph.compile(checkpointer=self.checkpointer)

    def cleanup_memory(self):
        gc.collect()
