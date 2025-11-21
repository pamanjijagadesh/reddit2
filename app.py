# test.py — Refactored, error-handled Streamlit app for Reddit -> LLM relevance scoring
import os
from dotenv import load_dotenv
load_dotenv()

import re
import json
import logging
import threading
import asyncio
from typing import List, Dict, Any

import praw
import prawcore
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize

from tenacity import retry, wait_exponential, stop_after_attempt
import streamlit as st

# LLM client import - keep if you have the package installed
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import SystemMessage

# ------------------------- Logging -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ------------------------- Constants -------------------------
SYSTEM_PROMPT = """
You are an evaluator tasked with judging how relevant social media posts are to a given research question.

Research Question:
{research_question}

Instructions:
- For each post, assign a relevance score from 1 to 5:
  1 = Not relevant at all (unrelated, off-topic).
  2 = Slightly relevant (mentions the topic but not methods).
  3 = Somewhat relevant (mentions methods but vague or indirect).
  4 = Relevant (clearly mentions a method related to the topic).
  5 = Highly relevant (directly describes practical, evidence-based methods related to the topic).
- Return results in JSON format as a list.

Posts to evaluate:
{posts_to_evaluate}

Output format (JSON only):
[
  {{
    "post_number": <number>,
    "score": <1-5>
  }},
  ...
]
"""  # note the double braces around the JSON sample so .format() works

MAX_CONCURRENT_REQUESTS = 2
MAX_RETRIES = 4

# ------------------------- Utilities -------------------------

def safe_nltk_download(package_path: str):
    """
    Ensure NLTK resource exists; if not, download it.
    package_path is the NLTK resource path, e.g. "tokenizers/punkt"
    """
    try:
        nltk.data.find(package_path)
    except LookupError:
        pkg = package_path.split("/")[-1]
        try:
            nltk.download(pkg, quiet=True)
            logger.info(f"Downloaded NLTK package: {pkg}")
        except Exception as e:
            logger.error(f"Failed to download NLTK package {pkg}: {e}")
            raise

# Ensure required NLTK resources
# Ensure required NLTK resources
for pkg in (
    "tokenizers/punkt",
    "tokenizers/punkt_tab",   # NEW — required by newer NLTK versions
    "corpora/stopwords",
    "taggers/averaged_perceptron_tagger",
):
    safe_nltk_download(pkg)

try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))

# ------------------------- Text processing -------------------------

def extract_keywords(text: str) -> str:
    """Normalize, tokenize, remove stopwords and pick nouns / adj+noun pairs."""
    if not text:
        return ""
    text = re.sub(r"[^\w\s]", "", text.lower())
    try:
        tokens = word_tokenize(text)
    except LookupError:
        nltk.download("punkt", quiet=True)
        tokens = word_tokenize(text)

    tokens = [t for t in tokens if t not in STOP_WORDS]
    try:
        tagged = pos_tag(tokens)
    except Exception:
        # If POS tagger isn't available for some reason, just return tokens joined
        return " ".join(tokens)

    keywords = []
    i = 0
    while i < len(tagged):
        word, pos = tagged[i]
        if pos.startswith("JJ") and i + 1 < len(tagged) and tagged[i + 1][1].startswith("NN"):
            keywords.append(f"{word} {tagged[i + 1][0]}")
            i += 2
            continue
        if pos.startswith("NN"):
            keywords.append(word)
        i += 1

    return " ".join(keywords)

def clean_subreddit_name(name: str) -> str:
    if not name:
        return ""
    cleaned = name.strip()
    cleaned = re.sub(r"^(?:r/|/r/)+", "", cleaned, flags=re.IGNORECASE)
    return cleaned

# ------------------------- Reddit fetch (sync, run in thread) -------------------------

def _fetch_reddit_sync(creds: Dict[str, str], key_words: str, subreddit_name: str, limit: int, time_filter: str) -> List[Dict[str, Any]]:
    """
    Synchronous PRAW fetch, meant to be run in a separate thread.
    Raises RuntimeError for known failure modes to be handled upstream.
    """
    try:
        reddit = praw.Reddit(
            client_id=creds.get("client_id"),
            client_secret=creds.get("client_secret"),
            username=creds.get("username"),
            password=creds.get("password"),
            user_agent=creds.get("user_agent"),
        )

        # This request can raise prawcore.exceptions.NotFound on invalid subreddit
        submissions = reddit.subreddit(subreddit_name).search(key_words, limit=limit, time_filter=time_filter)

        posts = []
        for idx, submission in enumerate(submissions, start=1):
            posts.append({
                "post_number": idx,
                "title": submission.title,
                "body": submission.selftext or "",
                "url": submission.url,
            })
        return posts

    except prawcore.exceptions.NotFound as e:
        logger.error("Subreddit not found", exc_info=True)
        raise RuntimeError(f"Subreddit '{subreddit_name}' not found.") from e
    except prawcore.exceptions.PrawcoreException as e:
        logger.error("PRAW API error", exc_info=True)
        raise RuntimeError(f"Reddit API error: {e}") from e
    except Exception as e:
        logger.critical("Unexpected error fetching Reddit", exc_info=True)
        raise RuntimeError(f"Unexpected Reddit error: {e}") from e

# Async wrapper that runs the sync reddit fetch in a threadpool, with error capture
async def get_reddit_posts(creds: Dict[str, str], key_words: str, subreddit_name: str, limit: int, time_filter: str) -> Any:
    try:
        return await asyncio.to_thread(_fetch_reddit_sync, creds, key_words, subreddit_name, limit, time_filter)
    except RuntimeError as e:
        # Return a structured error object to the orchestrator
        return {"error": str(e)}
    except Exception as e:
        logger.exception("Unhandled exception in get_reddit_posts")
        return {"error": f"Unhandled Reddit fetch error: {e}"}

# ------------------------- LLM batching and calls -------------------------

semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

@retry(wait=wait_exponential(multiplier=1, min=2, max=20), stop=stop_after_attempt(MAX_RETRIES))
async def _get_llm_response(research_question: str, posts: List[Dict[str, Any]], llm: ChatMistralAI) -> List[Dict[str, Any]]:
    """Send a batch to the LLM and parse the JSON list response."""
    async with semaphore:
        posts_str = ""
        for post in posts:
            posts_str += f"\n- Post {post['post_number']}:\n"
            posts_str += f"  Title: {post['title']}\n"
            content = post.get("body") or post.get("url") or ""
            posts_str += f"  Content: {content}\n"
            posts_str += "--- \n"

        formatted_prompt = SYSTEM_PROMPT.format(research_question=research_question, posts_to_evaluate=posts_str)
        response = await llm.ainvoke([SystemMessage(content=formatted_prompt)])
        output_text = (response.content or "").strip()

        # Remove markdown code fences if present
        cleaned_text = re.sub(r"```json\s*|```", "", output_text, flags=re.DOTALL).strip()

        if not cleaned_text.startswith("["):
            logger.error("LLM response did not start with '['; raw content: %s", output_text)
            raise json.JSONDecodeError("LLM response not JSON list", cleaned_text, 0)

        parsed = json.loads(cleaned_text)
        return parsed

# ------------------------- Orchestrator -------------------------

async def reddit_main(query: str, subreddit_name: str, limit: int, creds: Dict[str, str], llm: ChatMistralAI, time_filter: str, batch_size: int = 3) -> Any:
    """
    Orchestrates:
    - keyword extraction
    - reddit fetch
    - batching & LLM calls
    Returns either a list of sorted posts or a dict {'error': '...'} on failure.
    """
    cleaned_subreddit_name = clean_subreddit_name(subreddit_name)
    if not cleaned_subreddit_name:
        return {"error": "Invalid subreddit name."}

    try:
        key_words = extract_keywords(query)
    except Exception as e:
        logger.exception("Keyword extraction failed")
        return {"error": f"Keyword extraction failed: {e}"}

    # Fetch posts (may return an error dict)
    reddit_results = await get_reddit_posts(creds, key_words, cleaned_subreddit_name, limit, time_filter)
    if isinstance(reddit_results, dict) and "error" in reddit_results:
        return reddit_results  # propagate error

    if not reddit_results:
        # No posts found — return empty list
        return []

    # chunk into batches
    batches = [reddit_results[i:i + batch_size] for i in range(0, len(reddit_results), batch_size)]

    # LLM calls with error aggregation
    try:
        tasks = [_get_llm_response(query, batch, llm) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        logger.exception("LLM batch submission failed")
        return {"error": f"LLM batch submission failed: {e}"}

    all_evaluations = []
    for r in results:
        if isinstance(r, Exception):
            logger.warning("A batch failed: %s", r)
        else:
            all_evaluations.extend(r)

    # Build a mapping of post_number -> score
    try:
        scores_map = {int(ev["post_number"]): int(ev["score"]) for ev in all_evaluations if "post_number" in ev and "score" in ev}
    except Exception as e:
        logger.exception("Failed to build scores_map")
        return {"error": f"Failed to parse LLM evaluations: {e}"}

    posts_with_scores = []
    for p in reddit_results:
        num = p.get("post_number")
        if num in scores_map:
            p["score"] = scores_map[num]
            posts_with_scores.append(p)

    sorted_posts = sorted(posts_with_scores, key=lambda x: x.get("score", 0), reverse=True)
    return sorted_posts

# ------------------------- Async runner (safe for Streamlit) -------------------------

def run_coro_in_thread(coro):
    """
    Run an async coroutine in its own event loop inside a separate thread and return result.
    The result can be a normal value or a dict with an 'error' key.
    """
    result_holder = {}

    def _worker():
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result_holder["value"] = loop.run_until_complete(coro)
        except Exception as e:
            # Capture exceptions so main thread can display them
            logger.exception("Exception in background coroutine")
            result_holder["value"] = {"error": f"Background task error: {e}"}
        finally:
            loop.close()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join()
    return result_holder.get("value")

# ------------------------- LLM Client (cached) -------------------------

@st.cache_resource
def get_llm_client(endpoint: str, api_key: str) -> ChatMistralAI:
    if not endpoint or not api_key:
        raise ValueError("Missing LLM endpoint or API key")
    return ChatMistralAI(endpoint=endpoint, mistral_api_key=api_key)

# ------------------------- Streamlit UI -------------------------

st.set_page_config(page_title="NourIQ-SocialAi")
st.title("Reddit Search & Relevance Analysis — Refactored")

st.markdown("Find posts and rank them by relevance to a research question using an LLM.")

# LLM credentials (prefer environment or st.secrets)
llm_endpoint = "https://mistral-small-reddit.swedencentral.models.ai.azure.com"
llm_api_key = "GzpY33xGmeqGkd6zEH9lZEm2yDSA9ej1"

if not llm_endpoint:
    llm_endpoint = st.text_input("LLM Endpoint (Mistral/Azure)")
if not llm_api_key:
    llm_api_key = st.text_input("LLM API Key", type="password")

# Reddit credentials
st.subheader("1. Reddit API Credentials")
client_id = st.text_input("Client ID", type="password")
client_secret = st.text_input("Client Secret", type="password")
username = st.text_input("Username")
password = st.text_input("Password", type="password")
user_agent = st.text_input("User Agent", "nouriq-app")

st.subheader("2. Search Parameters")
research_question = st.text_input("Research Question / Search Query")
subreddit_name = st.text_input("Enter the Subreddit Name (e.g., mentalhealth, diabetes)")

TIME_FILTER_OPTIONS = {"Past Day": "day", "Past Week": "week", "Past Month": "month", "Past Year": "year", "All Time": "all"}
selected_time_filter_label = st.selectbox("Post Age Limit", options=list(TIME_FILTER_OPTIONS.keys()), index=3)
limit = st.number_input("Maximum Number of Posts to Retrieve", min_value=1, max_value=100, value=10)
batch_size = 3

if st.button("Search and Analyze Reddit Posts"):
    # Basic validation
    if not all([client_id, client_secret, username, password]):
        st.error("Please fill in all Reddit API credentials.")
    elif not research_question or not subreddit_name:
        st.error("Please provide a Research Question and Subreddit Name.")
    elif not llm_endpoint or not llm_api_key:
        st.error("LLM credentials are required for relevance analysis.")
    else:
        try:
            llm_client = get_llm_client(llm_endpoint, llm_api_key)
        except Exception as e:
            st.error(f"LLM Initialization Error: {e}")
            logger.exception("LLM init error")
            st.stop()

        user_creds = {
            "client_id": client_id,
            "client_secret": client_secret,
            "username": username,
            "password": password,
            "user_agent": user_agent,
        }

        time_filter_value = TIME_FILTER_OPTIONS[selected_time_filter_label]

        with st.spinner("Connecting to Reddit and analyzing posts..."):
            try:
                coro = reddit_main(
                    research_question,
                    subreddit_name,
                    int(limit),
                    user_creds,
                    llm_client,
                    time_filter_value,
                    batch_size=int(batch_size),
                )
                result = run_coro_in_thread(coro)

                # If error dict returned, show it in UI
                if isinstance(result, dict) and "error" in result:
                    st.error(result["error"])
                    st.stop()

                sorted_posts = result  # safe to assume it's list

                display_subreddit_name = clean_subreddit_name(subreddit_name)

                if sorted_posts:
                    st.header(f"--- \n Search Results in r/{display_subreddit_name}")
                    st.markdown("Posts are ranked by **Relevance Score** (5 = highly relevant, 1 = not relevant).")
                    st.markdown(f"**:hourglass: Note: Posts are limited to the {selected_time_filter_label}.**")

                    for submission in sorted_posts:
                        st.markdown(f"**Relevance Score:** {submission.get('score', 'N/A')} / 5")
                        st.markdown(f"**Title:** {submission.get('title', 'N/A')}")
                        body_preview = (submission.get("body") or "")[:200]
                        if body_preview:
                            st.caption(f"Body Preview: {body_preview}...")
                        st.markdown(f"**Link:** [{submission.get('url')}]({submission.get('url')})")
                        st.markdown("---")
                else:
                    st.warning(f"No posts were found or analyzed in r/{display_subreddit_name}.")

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

                logger.exception("Unhandled error in main flow")
