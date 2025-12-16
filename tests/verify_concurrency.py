
import asyncio
import time
import logging
from unittest.mock import MagicMock

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Mocks ---

class MockTokenizer:
    def encode(self, text):
        return [1] * len(text)
    def decode(self, tokens):
        return "decoded"

# Mock helpers that would come from lightrag.utils
def truncate_list_by_token_size(list_data, key, max_token_size, tokenizer):
    # Simple mock that just returns the list, assuming it fits
    return list_data

# --- The Function Under Test (Copied & Adapted from operate.py) ---
# We mock _summarize_descriptions to simulate LLM latency

async def _summarize_descriptions(
    description_type: str,
    description_name: str,
    description_list: list[str],
    global_config: dict,
    llm_response_cache = None,
) -> str:
    # Simulate LLM call
    use_llm_func = global_config["llm_model_func"]
    # We ignore priority/cache logic for this test
    return await use_llm_func("prompt")

async def _handle_entity_relation_summary(
    description_type: str,
    entity_or_relation_name: str,
    description_list: list[str],
    seperator: str,
    global_config: dict,
    llm_response_cache = None,
) -> tuple[str, bool]:
    # Handle empty input
    if not description_list:
        return "", False

    if len(description_list) == 1:
        return description_list[0], False

    tokenizer = global_config["tokenizer"]
    summary_context_size = global_config["summary_context_size"]
    summary_max_tokens = global_config["summary_max_tokens"]
    force_llm_summary_on_merge = global_config["force_llm_summary_on_merge"]

    current_list = description_list[:]
    llm_was_used = False

    while True:
        total_tokens = sum(len(tokenizer.encode(desc)) for desc in current_list)

        if total_tokens <= summary_context_size or len(current_list) <= 2:
            if (
                len(current_list) < force_llm_summary_on_merge
                and total_tokens < summary_max_tokens
            ):
                final_description = seperator.join(current_list)
                return final_description if final_description else "", llm_was_used
            else:
                final_summary = await _summarize_descriptions(
                    description_type,
                    entity_or_relation_name,
                    current_list,
                    global_config,
                    llm_response_cache,
                )
                return final_summary, True

        # Need to split into chunks - Map phase
        chunks = []
        current_chunk = []
        current_tokens = 0

        for i, desc in enumerate(current_list):
            desc_tokens = len(tokenizer.encode(desc))
            if current_tokens + desc_tokens > summary_context_size and current_chunk:
                if len(current_chunk) == 1:
                    current_chunk.append(desc)
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_tokens = 0
                else:
                    chunks.append(current_chunk)
                    current_chunk = [desc]
                    current_tokens = desc_tokens
            else:
                current_chunk.append(desc)
                current_tokens += desc_tokens

        if current_chunk:
            chunks.append(current_chunk)

        logger.info(
            f"   Summarizing {entity_or_relation_name}: Map {len(current_list)} descriptions into {len(chunks)} groups"
        )

        # --- THIS IS THE CRITICAL SECTION WE CHANGED ---
        # Reduce phase: summarize each group from chunks
        valid_chunks = []
        tasks = []
        for chunk in chunks:
            if len(chunk) == 1:
                # Optimization: single description chunks don't need LLM summarization
                valid_chunks.append(chunk[0])
            else:
                # Multiple descriptions need LLM summarization
                valid_chunks.append(None)  # Placeholder to preserve order
                task_index = len(valid_chunks) - 1
                tasks.append(
                    (
                        task_index,
                        _summarize_descriptions(
                            description_type,
                            entity_or_relation_name,
                            chunk,
                            global_config,
                            llm_response_cache,
                        ),
                    )
                )

        if tasks:
            llm_was_used = True
            # Execute all summarization tasks concurrently
            results = await asyncio.gather(*(t[1] for t in tasks))
            
            # Place results back into their original positions
            for (task_index, _), result in zip(tasks, results):
                valid_chunks[task_index] = result

        # Filter out any None values
        new_summaries = [s for s in valid_chunks if s is not None]
        # -----------------------------------------------

        current_list = new_summaries


# --- Test Runner ---

async def mock_llm_func(prompt, _priority=None):
    # Simulate processing time
    await asyncio.sleep(0.5) 
    return "Summary"

global_config = {
    "tokenizer": MockTokenizer(),
    "summary_context_size": 100,
    "summary_max_tokens": 500,
    "force_llm_summary_on_merge": 10,
    "llm_model_func": mock_llm_func,
    "addon_params": {},
    "summary_length_recommended": 50,
    "embedding_token_limit": 1000
}

async def verify_concurrency():
    # 20 descriptions, length 20 each = 400 tokens total.
    # summary_context_size = 100.
    # Should split into approx 4 chunks.
    # 4 chunks * 0.5s parallel = ~0.5s (+ overhead)
    # Sequential would be 2.0s
    
    descriptions = ["Description " + str(i) * 10 for i in range(20)]
    
    print(f"Starting test with {len(descriptions)} descriptions...")
    start_time = time.time()
    
    summary, used_llm = await _handle_entity_relation_summary(
        "entity",
        "TestEntity",
        descriptions,
        "\n",
        global_config,
        None
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Total execution time: {duration:.2f} seconds")
    
    if duration < 1.5:
        print("SUCCESS: Execution time indicates PARALLEL processing.")
    else:
        print("FAILURE: Execution time indicates SEQUENTIAL processing.")

if __name__ == "__main__":
    asyncio.run(verify_concurrency())
