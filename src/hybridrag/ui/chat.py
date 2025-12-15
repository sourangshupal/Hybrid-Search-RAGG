"""
Chainlit chat handlers for HybridRAG.

Features:
- PDF/TXT/MD file upload and ingestion
- Chat with documents using hybrid search
- Slash commands for mode switching
- Streaming responses
"""

import os
import sys
import tempfile
from typing import Optional

import chainlit as cl

# Add project paths
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_project_root, "src"))

from hybridrag import create_hybridrag, Settings


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(file_path)
        text_parts = []

        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            if text.strip():
                text_parts.append(f"--- Page {page_num} ---\n{text}")

        doc.close()
        return "\n\n".join(text_parts)
    except ImportError:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")


def extract_text_from_file(file_path: str, mime_type: str) -> str:
    """Extract text from various file types."""
    if mime_type == "application/pdf" or file_path.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif mime_type in ["text/plain", "text/markdown"] or file_path.endswith((".txt", ".md")):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file type: {mime_type}")


@cl.on_chat_start
async def on_chat_start():
    """Initialize HybridRAG when chat session starts."""

    await cl.Message(
        content="Initializing HybridRAG... Please wait."
    ).send()

    try:
        # Get settings from environment
        settings = Settings()

        # Create and initialize HybridRAG
        rag = await create_hybridrag(settings=settings, auto_initialize=True)

        # Store in session
        cl.user_session.set("rag", rag)
        cl.user_session.set("mode", "mix")  # Default mode

        # Welcome message
        welcome = """**HybridRAG Chat** is ready!

**Upload documents:**
- Drag & drop PDF, TXT, or MD files
- Documents will be processed with knowledge graph extraction

**Chat commands:**
- `/mode <local|global|hybrid|naive|mix>` - Change query mode
- `/status` - Show system status
- `/help` - Show this message

**Current mode:** `mix` (Knowledge Graph + Vector Search)

Upload a document to get started, or ask a question about previously uploaded documents!
"""
        await cl.Message(content=welcome).send()

    except Exception as e:
        await cl.Message(
            content=f"**Error initializing HybridRAG:** {str(e)}\n\nPlease check your environment variables (MONGODB_URI, VOYAGE_API_KEY, etc.)"
        ).send()


# Common greetings that shouldn't trigger RAG retrieval
GREETINGS = {
    "hello", "hi", "hey", "hola", "greetings", "good morning", "good afternoon",
    "good evening", "howdy", "yo", "sup", "what's up", "whats up", "hi there",
    "hello there", "hey there", "hiya", "heya"
}

# Meta-questions about the system/knowledge base
META_PATTERNS = [
    "what do you know", "what's in your", "whats in your", "what is in your",
    "do you have anything", "have anything in", "knowledge base", "what can you",
    "what documents", "what files", "what data", "tell me about yourself",
    "who are you", "what are you"
]


def is_greeting(text: str) -> bool:
    """Check if the message is a simple greeting."""
    normalized = text.lower().strip().rstrip("!?.,:;")
    return normalized in GREETINGS or len(normalized) < 3


def is_meta_question(text: str) -> bool:
    """Check if the message is a meta-question about the system."""
    normalized = text.lower()
    return any(pattern in normalized for pattern in META_PATTERNS)


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages."""

    rag = cl.user_session.get("rag")
    if not rag:
        await cl.Message(content="RAG system not initialized. Please refresh the page.").send()
        return

    # Check for file uploads
    if message.elements:
        await handle_file_upload(message.elements, rag)
        return

    # Check for slash commands
    text = message.content.strip()
    if text.startswith("/"):
        await handle_command(text, rag)
        return

    # Handle greetings without RAG
    if is_greeting(text):
        await cl.Message(
            content="Hello! I'm your HybridRAG assistant. I can help you with questions about your uploaded documents.\n\n"
                    "**Try asking me:**\n"
                    "- Questions about content in your documents\n"
                    "- Summaries or explanations of topics\n"
                    "- Specific details or facts\n\n"
                    "Upload a PDF, TXT, or MD file to get started, or ask about previously uploaded documents!"
        ).send()
        return

    # Handle meta-questions about the system
    if is_meta_question(text):
        await cl.Message(
            content="I'm **HybridRAG** - a hybrid Retrieval-Augmented Generation system powered by:\n\n"
                    "- **MongoDB Atlas** for vector + graph storage\n"
                    "- **Voyage AI** for embeddings & reranking\n"
                    "- **Multi-provider LLM** for response generation\n"
                    "- **Knowledge graph** extraction and querying\n\n"
                    "Upload documents and ask questions about them!\n\n"
                    "**Try asking:**\n"
                    "- Questions about your uploaded documents\n"
                    "- Summaries of specific topics\n"
                    "- Relationship queries between concepts"
        ).send()
        return

    # Handle regular query
    await handle_query(text, rag)


async def handle_file_upload(elements: list, rag):
    """Handle file uploads and ingest into RAG."""

    files_processed = []
    errors = []

    for element in elements:
        if not hasattr(element, "path") or not element.path:
            continue

        file_name = getattr(element, "name", "unknown")
        mime_type = getattr(element, "mime", "")

        try:
            # Extract text from file
            status_msg = await cl.Message(content=f"Processing `{file_name}`...").send()

            text = extract_text_from_file(element.path, mime_type)

            if not text.strip():
                errors.append(f"`{file_name}`: No text content found")
                continue

            # Ingest into RAG
            await status_msg.update(content=f"Ingesting `{file_name}` ({len(text):,} characters)...")

            await rag.insert(text, file_paths=[file_name])

            files_processed.append(f"`{file_name}` ({len(text):,} chars)")

            await status_msg.update(content=f"Successfully ingested `{file_name}`!")

        except Exception as e:
            errors.append(f"`{file_name}`: {str(e)}")

    # Summary message
    summary_parts = []

    if files_processed:
        summary_parts.append(f"**Ingested {len(files_processed)} file(s):**\n" + "\n".join(f"- {f}" for f in files_processed))

    if errors:
        summary_parts.append(f"**Errors ({len(errors)}):**\n" + "\n".join(f"- {e}" for e in errors))

    if summary_parts:
        await cl.Message(content="\n\n".join(summary_parts)).send()

    if files_processed:
        await cl.Message(content="You can now ask questions about the uploaded documents!").send()


async def handle_command(text: str, rag):
    """Handle slash commands."""

    parts = text.split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if command == "/mode":
        valid_modes = ["local", "global", "hybrid", "naive", "mix", "bypass"]
        if args.lower() in valid_modes:
            cl.user_session.set("mode", args.lower())
            await cl.Message(content=f"Query mode changed to **{args.lower()}**").send()
        else:
            await cl.Message(content=f"Invalid mode. Valid modes: {', '.join(valid_modes)}").send()

    elif command == "/status":
        try:
            status = await rag.get_status()
            status_text = f"""**System Status**

- **Initialized:** {status.get('initialized', False)}
- **LLM Provider:** {status.get('llm_provider', 'unknown')}
- **LLM Model:** {status.get('llm_model', 'unknown')}
- **Embedding Provider:** {status.get('embedding_provider', 'unknown')}
- **Embedding Model:** {status.get('embedding_model', 'unknown')}
- **Rerank Model:** {status.get('rerank_model', 'disabled')}
- **Current Mode:** {cl.user_session.get('mode', 'mix')}
"""
            await cl.Message(content=status_text).send()
        except Exception as e:
            await cl.Message(content=f"Error getting status: {e}").send()

    elif command == "/help":
        help_text = """**Available Commands**

- `/mode <mode>` - Change query mode
  - `local` - Entity-focused (graph neighbors)
  - `global` - Community summaries
  - `hybrid` - Local + Global combined
  - `naive` - Direct vector search
  - `mix` - All modes combined (recommended)
  - `bypass` - Direct LLM (no retrieval)

- `/status` - Show system configuration
- `/help` - Show this message

**File Upload**
- Drag & drop PDF, TXT, or MD files to ingest
"""
        await cl.Message(content=help_text).send()

    else:
        await cl.Message(content=f"Unknown command: `{command}`. Type `/help` for available commands.").send()


async def handle_query(query: str, rag):
    """Handle a user query."""

    mode = cl.user_session.get("mode", "mix")

    # Create a message that we'll update
    msg = cl.Message(content="Searching...")
    await msg.send()

    try:
        # Query with sources
        result = await rag.query_with_sources(
            query=query,
            mode=mode,
        )

        answer = result.get("answer", "No answer generated.")
        context = result.get("context", "")

        # Build response text - keep it clean, don't show raw context
        response_text = answer

        # Update the message content directly
        msg.content = response_text
        await msg.update()

    except Exception as e:
        msg.content = f"**Error:** {str(e)}"
        await msg.update()
