import asyncio
import hashlib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests
from openai import AsyncOpenAI
from PIL import Image, ImageFile
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configuration - OPTIMIZED FOR RATE LIMITING AND API TIMEOUTS
CONFIG = {
    "api_key": "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",
    "base_url": "http://127.0.0.1:1234/v1",  # OpenRouter base URL for OpenAI client
    "model": "gemma-3-4b-it",
    "image_size": (512, 512),
    "invalid_image_size": (130, 60),
    "images_dir": Path("data/images"),
    "prompts_dir": Path("data/prompts"),
    "batch_size": 10,  # Reduced further to avoid rate limits
    "max_retries": 5,  # Increased retries
    "retry_delay": 2.0,  # Increased base delay
    "timeout": 120,  # Increased to 2 minutes for API calls
    "download_timeout": 5.0,  # Skip images that take more than 5 seconds to download
    "max_concurrent_requests": 10,
    "max_workers": 20,  # Thread pool size for image processing
    "connect_timeout": 10,  # Connection timeout
    "read_timeout": 110,  # Read timeout (less than total timeout)
    "download_delay": 0.5,  # Minimum delay between image downloads (in seconds)
    "batch_delay": 1.0,  # Delay between batches to prevent rate limiting
}


# Setup logging - COMPREHENSIVE FIX for duplicate logs
def setup_logger():
    """Set up logger with comprehensive duplicate prevention."""
    # Use a specific logger name to avoid conflicts
    logger_name = "generate_prompts"
    logger = logging.getLogger(logger_name)

    # Clear ALL handlers from this specific logger AND root logger
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Also clear root logger handlers to prevent inheritance issues
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set logger level
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler
    file_handler = logging.FileHandler("generate_prompts.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.WARNING)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.WARNING)
    logger.addHandler(console_handler)

    # CRITICAL: Prevent propagation completely
    logger.propagate = False

    return logger


# Initialize logger
logger = setup_logger()

# Test that logging is working correctly
logger.info(
    "Logger setup completed with enhanced rate limiting - this message should appear only once"
)

# Create directories
CONFIG["images_dir"].mkdir(parents=True, exist_ok=True)
CONFIG["prompts_dir"].mkdir(parents=True, exist_ok=True)

# Session with connection pooling and retry strategy (for image downloads)
# Updated with more conservative settings for rate limiting
session = requests.Session()
retry_strategy = Retry(
    total=CONFIG["max_retries"],
    backoff_factor=3,  # More aggressive backoff
    status_forcelist=[429, 500, 502, 503, 504],
    respect_retry_after_header=True,  # Respect server retry-after headers
)
adapter = HTTPAdapter(
    max_retries=retry_strategy,
    pool_maxsize=5,  # Reduced pool size
    pool_block=True,
)
session.mount("http://", adapter)
session.mount("https://", adapter)

# Add rate limiting headers to be more respectful
session.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
)


SYSTEM_PROMPT = """
**Task:** You are an expert in creating image editing instructions. Your goal is to analyze an image and its accompanying caption to generate a concise instruction that describes a specific, tangible edit to the main subject of the image.

**Instructions:**
1.  **Focus on the Subject:** The edit instruction must apply *only* to the primary subject of the image.
2.  **Describe the Action:** The instruction should describe a clear action or transformation based on the caption.
3.  **Be Concise:** Output only the edit instruction itself, with no extra text, labels, or explanations.
4.  **Exclude:** Do not mention changes to the background, lighting, contrast, saturation, or overall image style.

---
**Example:**

**Input Caption:**
This image displays: a group of nine mountaineers on a snow-covered mountain summit. They are dressed in warm gear and carrying backpacks. There is a mountain in the background and clouds below and behind them. One of the mountaineers is waving. The image is a photograph.

**Generated Instruction:**
Give the waving mountaineer a soda can in their hand.
---

Caption:
"""


def get_image_hash(url: str) -> str:
    """Generate a hash for the URL to create unique filenames."""
    return hashlib.md5(url.encode()).hexdigest()


def is_processed(image_hash: str) -> bool:
    """Check if an image has already been processed."""
    prompt_file = CONFIG["prompts_dir"] / f"{image_hash}.txt"
    return prompt_file.exists()


def save_prompt(image_hash: str, prompt: str) -> None:
    """Save the generated prompt to a file."""
    prompt_file = CONFIG["prompts_dir"] / f"{image_hash}.txt"
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)


def download_and_process_image(
    url: str, timeout: float = None
) -> Tuple[Optional[str], Optional[str]]:
    """Download and process image from URL with ENFORCED 5-second timeout."""
    if timeout is None:
        timeout = CONFIG.get("download_timeout", 5.0)  # Default 5-second timeout

    # Skip all imgur URLs immediately
    # if "imgur" in url.lower():
    #     # logger.info(f"🚫 Skipping imgur URL: {url}")
    #     return None, None

    def _download():
        """Internal download function."""
        start_time = time.time()
        try:
            # Configure session with very aggressive timeouts
            local_session = requests.Session()
            local_session.mount("http://", requests.adapters.HTTPAdapter(max_retries=0))
            local_session.mount(
                "https://", requests.adapters.HTTPAdapter(max_retries=0)
            )

            response = local_session.get(
                url,
                timeout=(2.0, 3.0),  # Very short timeouts (connect, read)
                stream=True,
                allow_redirects=False,  # No redirects to avoid delays
                headers={"User-Agent": "Mozilla/5.0"},
            )

            download_time = time.time() - start_time
            response.raise_for_status()

            # Quick checks
            content_type = response.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                return None, download_time, "not an image"

            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB limit
                return None, download_time, "too large"

            # Get content quickly
            content = response.content
            total_time = time.time() - start_time

            return content, total_time, "success"

        except Exception as e:
            return None, time.time() - start_time, str(e)
        finally:
            try:
                local_session.close()
            except Exception as e:
                logger.error(f"Error closing local session: {e}")
                pass

    try:
        # logger.debug(f"Downloading: {url} (max {timeout}s)")

        # Use thread with strict timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_download)

            try:
                content, download_time, status = future.result(timeout=timeout)

                if content is None:
                    # Suppress warning log for failed downloads
                    return None, None

                if download_time > 3.0:
                    logger.warning(f"Slow download ({download_time:.1f}s): {url}")
                    pass

                # Process image
                img = Image.open(BytesIO(content)).convert("RGB")

                # Skip placeholders
                if img.size == CONFIG["invalid_image_size"]:
                    # logger.debug(f"Skipping placeholder: {url}")
                    return None, None

                # Resize and save
                img = img.resize(CONFIG["image_size"], Image.Resampling.LANCZOS)

                image_hash = get_image_hash(url)
                img_name = f"{image_hash}.jpg"
                img_path = CONFIG["images_dir"] / img_name

                img.save(img_path, "JPEG", quality=95, optimize=True)

                # logger.debug(f"✅ Downloaded: {url} ({download_time:.1f}s)")
                return str(img_path), url

            except FutureTimeoutError:
                # logger.warning(f"⏰ TIMEOUT: {url} (>{timeout}s) - skipping")
                future.cancel()
                return None, None

    except Exception:
        # logger.warning(f"❌ Error: {url} - {type(e).__name__}: {str(e)}")
        return None, None

    return None, None


def encode_image_to_base64(image_path: str) -> Optional[str]:
    """Encode image file to base64 string."""
    try:
        import base64

        with open(image_path, "rb") as image_file:
            base64_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_string}"
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        return None


async def call_api_async(
    client: AsyncOpenAI, caption: str, image_path: str, max_retries: int = None
) -> Optional[str]:
    """Make async API call with proper error handling and retries using OpenAI client with base64 encoded images."""
    if max_retries is None:
        max_retries = CONFIG["max_retries"]

    # Encode image to base64
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        logger.error(f"Failed to encode image to base64: {image_path}")
        return None

    for attempt in range(max_retries + 1):
        try:
            logger.debug(
                f"Making API call (attempt {attempt + 1}/{max_retries + 1}) for image: {image_path}"
            )

            response = await client.chat.completions.create(
                model=CONFIG["model"],
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": SYSTEM_PROMPT + caption},
                            {"type": "image_url", "image_url": {"url": base64_image}},
                        ],
                    }
                ],
                timeout=CONFIG["timeout"],
            )

            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content.strip()
                logger.debug(f"Successfully generated prompt for image: {image_path}")
                return content
            else:
                logger.error(f"No choices in API response for image: {image_path}")
                return None

        except Exception as e:
            import traceback

            error_details = f"API call error: {type(e).__name__}: {str(e)}"

            # Handle specific OpenAI errors
            if hasattr(e, "status_code"):
                error_details += f" (status: {e.status_code})"

                # Rate limiting
                if e.status_code == 429:
                    if attempt < max_retries:
                        wait_time = CONFIG["retry_delay"] * (
                            3**attempt
                        )  # More aggressive backoff
                        logger.warning(
                            f"Rate limited, waiting {wait_time}s before retry"
                        )
                        await asyncio.sleep(wait_time)
                        continue

                # Server errors
                elif e.status_code >= 500:
                    if attempt < max_retries:
                        wait_time = CONFIG["retry_delay"] * (2**attempt)
                        logger.warning(
                            f"Server error {e.status_code}, waiting {wait_time}s before retry"
                        )
                        await asyncio.sleep(wait_time)
                        continue

            # Timeout errors
            elif "timeout" in str(e).lower():
                if attempt < max_retries:
                    wait_time = CONFIG["retry_delay"] * (2**attempt)
                    logger.warning(
                        f"API call timed out (attempt {attempt + 1}/{max_retries + 1}) after {CONFIG['timeout']}s. Waiting {wait_time}s before retry..."
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(
                        f"API call timed out after {max_retries + 1} attempts. Consider increasing timeout or reducing concurrent requests."
                    )
                    return None

            # General retry for other errors
            if attempt < max_retries:
                wait_time = CONFIG["retry_delay"] * (2**attempt)
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{max_retries + 1}): {error_details}. Waiting {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
            else:
                stack_trace = traceback.format_exc()
                logger.error(
                    f"API call failed after {max_retries + 1} attempts: {error_details}\nFull traceback:\n{stack_trace}"
                )
                return None

    return None


def call_api(caption: str, image_path: str, max_retries: int = None) -> Optional[str]:
    """Synchronous wrapper for backward compatibility."""

    async def run_async():
        client = AsyncOpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"])
        return await call_api_async(client, caption, image_path, max_retries)

    return asyncio.run(run_async())


async def process_single_item(
    client: AsyncOpenAI, data: pd.Series, pbar: tqdm
) -> Dict[str, Any]:
    """Process a single item (image + caption) asynchronously with rate limiting."""
    result = {"processed": 0, "errors": 0, "cached": 0}

    img_url = data.get("url", "")
    if not img_url:
        result["errors"] = 1
        pbar.update(1)
        return result

    # Create caption
    vlm_caption = data.get("vlm_caption", "")
    original_caption = data.get("original_caption", "")
    caption = f"{vlm_caption}\n{original_caption}".strip()

    if not caption:
        logger.warning(f"Empty caption for URL: {img_url}")
        result["errors"] = 1
        pbar.update(1)
        return result

    # Check if already processed
    image_hash = get_image_hash(img_url)
    if is_processed(image_hash):
        result["cached"] = 1
        pbar.update(1)
        return result

    try:
        # Add a small delay before downloading to prevent overwhelming servers
        await asyncio.sleep(CONFIG["download_delay"])

        # Download and process image (synchronous - file I/O intensive)
        img_path, processed_url = download_and_process_image(img_url)
        if img_path is None:
            result["errors"] = 1
            pbar.update(1)
            return result

        # Generate prompt via API (asynchronous) - now using image path for base64 encoding
        generated_prompt = await call_api_async(client, caption, img_path)
        if generated_prompt:
            save_prompt(image_hash, generated_prompt)
            result["processed"] = 1
            logger.info(f"Generated prompt for {img_url}: {generated_prompt[:100]}...")
        else:
            result["errors"] = 1

    except Exception as e:
        logger.error(f"Error processing {img_url}: {e}")
        result["errors"] = 1

    pbar.update(1)
    return result


async def process_batch_async(df_batch: pd.DataFrame, pbar: tqdm) -> Dict[str, Any]:
    """Process a batch of images concurrently with rate limiting."""
    stats = {"processed": 0, "skipped": 0, "errors": 0, "cached": 0}

    # Create OpenAI client for async operations
    client = AsyncOpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"])

    # Create semaphore to limit concurrent requests (keep it conservative)
    semaphore = asyncio.Semaphore(CONFIG["max_concurrent_requests"])

    async def process_with_semaphore(data):
        async with semaphore:
            return await process_single_item(client, data, pbar)

    # Process items with rate limiting - process them sequentially if max_concurrent_requests is 1
    if CONFIG["max_concurrent_requests"] == 1:
        # Sequential processing to be more respectful to servers
        results = []
        for _, data in df_batch.iterrows():
            result = await process_single_item(client, data, pbar)
            results.append(result)
    else:
        # Create tasks for all items in the batch
        tasks = [process_with_semaphore(data) for _, data in df_batch.iterrows()]

        # Process all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Aggregate results
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Task failed with exception: {result}")
            stats["errors"] += 1
        else:
            for key in stats:
                stats[key] += result.get(key, 0)

    return stats


def process_batch(df_batch: pd.DataFrame, pbar: tqdm) -> Dict[str, Any]:
    """Synchronous wrapper for batch processing."""
    try:
        # Check if we're already in an event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in a notebook with an existing event loop
            import nest_asyncio

            nest_asyncio.apply()
            return asyncio.run(process_batch_async(df_batch, pbar))
        else:
            return asyncio.run(process_batch_async(df_batch, pbar))
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(process_batch_async(df_batch, pbar))


def load_data_efficiently() -> pd.DataFrame:
    """Load parquet files efficiently."""
    try:
        parquet_files = ["data/vlm_captions_redcaps_00.parquet"]

        dataframes = []
        for file in parquet_files:
            if os.path.exists(file):
                df = pd.read_parquet(file)
                dataframes.append(df)
                logger.info(f"Loaded {len(df)} records from {file}")

        if not dataframes:
            raise FileNotFoundError("No parquet files found")

        df = dataframes[0].sample(n=20000)
        logger.info(f"Total records loaded: {len(df)}")
        return df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def main():
    """Main processing function with rate limiting and batch delays."""
    try:
        # Install nest_asyncio for notebook compatibility
        try:
            import nest_asyncio

            nest_asyncio.apply()
        except ImportError:
            logger.warning(
                "nest_asyncio not available. Install it with: pip install nest-asyncio"
            )

        # Load data
        logger.info("Loading data...")
        df = load_data_efficiently()

        # Process in batches
        total_batches = (len(df) + CONFIG["batch_size"] - 1) // CONFIG["batch_size"]
        overall_stats = {"processed": 0, "skipped": 0, "errors": 0, "cached": 0}

        logger.info(f"Processing {len(df)} records in {total_batches} batches")
        logger.info(
            f"Rate limiting settings: {CONFIG['max_concurrent_requests']} concurrent, {CONFIG['download_delay']}s download delay, {CONFIG['batch_delay']}s batch delay"
        )

        # Create a single progress bar for the entire process
        with tqdm(
            total=len(df),
            desc="Generating edit instructions",
            unit="images",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        ) as pbar:
            for batch_idx in range(total_batches):
                start_idx = batch_idx * CONFIG["batch_size"]
                end_idx = min(start_idx + CONFIG["batch_size"], len(df))
                df_batch = df.iloc[start_idx:end_idx]

                # logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(df_batch)} items)")

                # Process batch with rate limiting
                batch_stats = process_batch(df_batch, pbar)

                # Update overall statistics
                for key in overall_stats:
                    overall_stats[key] += batch_stats[key]

                # Update progress bar postfix with current stats
                pbar.set_postfix(
                    {
                        "Processed": overall_stats["processed"],
                        "Cached": overall_stats["cached"],
                        "Errors": overall_stats["errors"],
                        "Success Rate": f"{(overall_stats['processed'] / max(1, overall_stats['processed'] + overall_stats['errors']) * 100):.1f}%",
                    }
                )

                # Add delay between batches to prevent overwhelming servers
                # if batch_idx < total_batches - 1:  # Don't delay after the last batch
                #     logger.info(f"Waiting {CONFIG['batch_delay']}s between batches to prevent rate limiting...")
                #     time.sleep(CONFIG["batch_delay"])

        # Print final statistics
        logger.info("Processing completed!")
        logger.info("Final Statistics:")
        logger.info(f"  - Processed: {overall_stats['processed']}")
        logger.info(f"  - Cached: {overall_stats['cached']}")
        logger.info(f"  - Errors: {overall_stats['errors']}")
        logger.info(
            f"  - Success Rate: {(overall_stats['processed'] / max(1, overall_stats['processed'] + overall_stats['errors']) * 100):.1f}%"
        )

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
