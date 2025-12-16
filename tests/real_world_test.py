
import os
import asyncio
import time
import logging
from hybridrag.core.rag import HybridRAG

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Dummy text content
TEST_CONTENT = """
The ancient city of Aethelgard was known for its floating gardens. 
These gardens were suspended by massive crystals that hummed with a low frequency. 
The crystals were mined from the depths of the Singing Caverns, inhabited by the blind Myrmidons.
Queen Elara, the ruler of Aethelgard, formed a pact with the Myrmidons to ensure a steady supply of crystals.
In exchange, the city provided the Myrmidons with exotic fruits that only grew in the sunlight.
One day, a rogue scholar named Thaddeus discovered that the crystals were slowly losing their power.
Thaddeus warned the council, but they ignored him, believing the pact was eternal.
Desperate, Thaddeus ventured into the Singing Caverns to find the source of the draining energy.
He discovered a parasite, the Gloom-Leech, was feeding on the crystals' resonance.
"""

async def run_test():
    logger.info("Starting Real-World Verification Test")
    
    # Initialize RAG
    # We assume env vars are loaded by LightRAG or we load them
    from dotenv import load_dotenv
    load_dotenv()
    
    # Ensure RAG workspace exists (optional, HybridRAG might handle it)
    # HybridRAG uses settings, let's verify essential envs are present
    required_vars = ["MONGODB_URI", "VOYAGE_API_KEY", "GEMINI_API_KEY"]
    missing = [v for v in required_vars if not os.getenv(v)]
    if missing:
        logger.error(f"Missing environment variables: {missing}")
        return

    logger.info("Initializing HybridRAG...")
    rag = HybridRAG(working_dir="./rag_test_storage")
    await rag.initialize()
    
    logger.info("Ingesting document...")
    start_time = time.time()
    
    # HybridRAG.insert takes arguments: documents, ids, file_paths
    await rag.insert(documents=[TEST_CONTENT], file_paths=["test_story.txt"])
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Ingestion complete in {duration:.2f} seconds.")
    
    logger.info("Querying data...")
    query = "Why were the crystals failing?"
    result = await rag.query(query, mode="global") # forcing global map-reduce search
    
    logger.info(f"Query Result: {result}")
    
    if "Gloom-Leech" in result or "parasite" in result or "draining" in result:
        logger.info("VERIFICATION PASSED: Answer contains expected key terms.")
    else:
        logger.error("VERIFICATION FAILED: Answer did not contain expected key terms.")

if __name__ == "__main__":
    asyncio.run(run_test())
