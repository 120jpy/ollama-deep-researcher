import os
import requests
from typing import Dict, Any, List, Optional
from langsmith import traceable
from tavily import TavilyClient
import re
import fitz  # PyMuPDF
from selectolax.parser import HTMLParser
from urllib.parse import urljoin

def deduplicate_and_format_sources(search_response, max_tokens_per_source, include_raw_content=False):
    """
    Takes either a single search response or list of responses from search APIs and formats them.
    Limits the raw_content to approximately max_tokens_per_source.
    include_raw_content specifies whether to include the raw_content from Tavily in the formatted string.
    
    Args:
        search_response: Either:
            - A dict with a 'results' key containing a list of search results
            - A list of dicts, each containing search results
            
    Returns:
        str: Formatted string with deduplicated sources
    """
    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response['results']
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and 'results' in response:
                sources_list.extend(response['results'])
            else:
                sources_list.extend(response)
    else:
        raise ValueError("Input must be either a dict with 'results' or a list of search results")
    
    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        if source['url'] not in unique_sources:
            unique_sources[source['url']] = source
    
    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
                
    return formatted_text.strip()

def format_sources(search_results):
    """Format search results into a bullet-point list of sources.
    
    Args:
        search_results (dict): Tavily search response containing results
        
    Returns:
        str: Formatted string with sources and their URLs
    """
    return '\n'.join(
        f"* {source['title']} : {source['url']}"
        for source in search_results['results']
    )

@traceable
def tavily_search(query, include_raw_content=True, max_results=3):
    """ Search the web using the Tavily API.
    
    Args:
        query (str): The search query to execute
        include_raw_content (bool): Whether to include the raw_content from Tavily in the formatted string
        max_results (int): Maximum number of results to return
        
    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Full content of the page if available"""
     
    tavily_client = TavilyClient()
    return tavily_client.search(query, 
                         max_results=max_results, 
                         include_raw_content=include_raw_content)

@traceable
def perplexity_search(query: str, perplexity_search_loop_count: int) -> Dict[str, Any]:
    """Search the web using the Perplexity API.
    
    Args:
        query (str): The search query to execute
        perplexity_search_loop_count (int): The loop step for perplexity search (starts at 0)
  
    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result
                - url (str): URL of the search result
                - content (str): Snippet/summary of the content
                - raw_content (str): Full content of the page if available
    """

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {os.getenv('PERPLEXITY_API_KEY')}"
    }
    
    payload = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "system",
                "content": "Search the web and provide factual information with sources."
            },
            {
                "role": "user",
                "content": query
            }
        ]
    }
    
    response = requests.post(
        "https://api.perplexity.ai/chat/completions",
        headers=headers,
        json=payload
    )
    response.raise_for_status()  # Raise exception for bad status codes
    
    # Parse the response
    data = response.json()
    content = data["choices"][0]["message"]["content"]

    # Perplexity returns a list of citations for a single search result
    citations = data.get("citations", ["https://perplexity.ai"])
    
    # Return first citation with full content, others just as references
    results = [{
        "title": f"Perplexity Search {perplexity_search_loop_count + 1}, Source 1",
        "url": citations[0],
        "content": content,
        "raw_content": content
    }]
    
    # Add additional citations without duplicating content
    for i, citation in enumerate(citations[1:], start=2):
        results.append({
            "title": f"Perplexity Search {perplexity_search_loop_count + 1}, Source {i}",
            "url": citation,
            "content": "See above for full content",
            "raw_content": None
        })
    
    return {"results": results}

def clean_text(text: str) -> str:
    """Remove unnecessary whitespace, newlines, and tabs from text."""
    text = re.sub(r'\s+', ' ', text)  # 連続する空白・改行・タブを1つのスペースに置換
    return text.strip()

def fetch_full_content(url: str) -> str:
    """Fetches and cleans readable text from a webpage, including PDFs if available."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        tree = HTMLParser(response.text)

        # Extract text from common tags
        main_tags = ["article", "main", "body"]
        full_text = None
        for tag in main_tags:
            node = tree.css_first(tag)
            if node:
                full_text = clean_text(node.text(separator=" "))[:2000]  # 最大2000文字
                break
        if not full_text:
            full_text = "Content extraction failed."

        # Find PDF links
        pdf_texts = []
        for node in tree.css("a[href]"):
            href = node.attributes.get("href", "")
            if href.endswith(".pdf"):
                pdf_url = urljoin(url, href)  # 絶対URLに変換
                pdf_text = clean_text(fetch_pdf_text(pdf_url)[:1000])
                if pdf_text:
                    pdf_texts.append(f"[PDF] {pdf_url}: {pdf_text}")

        # Merge text content
        content = full_text
        if pdf_texts:
            content += "\n\n" + "\n".join(pdf_texts)

        return content

    except Exception as e:
        return f"Error fetching content: {str(e)}"

def fetch_pdf_text(pdf_url: str) -> str:
    """Downloads and extracts cleaned text from a PDF file."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(pdf_url, headers=headers, timeout=10)
        response.raise_for_status()

        with open("/tmp/temp.pdf", "wb") as f:
            f.write(response.content)

        # Extract text using PyMuPDF
        doc = fitz.open("/tmp/temp.pdf")
        text = "\n".join([page.get_text("text") for page in doc])
        doc.close()

        return clean_text(text[:5000])
    
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"

def google_search(query: str, google_search_loop_count: int) -> Dict[str, Any]:
    """Search the web using Google Custom Search API.
    
    Args:
        query (str): The search query to execute.
        google_search_loop_count (int): The loop step for Google search (starts at 0).
  
    Returns:
        dict: Search response containing:
            - results (list): List of search result dictionaries, each containing:
                - title (str): Title of the search result.
                - url (str): URL of the search result.
                - content (str): Snippet/summary of the content.
                - raw_content (str): Full content of the page if available.
    """
    api_key = os.getenv('GOOGLE_API_KEY')
    cx = os.getenv('GOOGLE_PSE')
    
    if not api_key or not cx:
        raise ValueError("Missing GOOGLE_API_KEY or GOOGLE_CSE_ID environment variable")
    
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": api_key,
        "cx": cx,
        "num": 5
    }
    
    response = requests.get(search_url, params=params)
    
    if response.status_code == 403:
        raise PermissionError("API request failed with 403 Forbidden. Check API key, CSE ID, and quota.")
    
    response.raise_for_status()
    data = response.json()
    
    results = []
    for i, item in enumerate(data.get("items", []), start=1):
        page_url = item.get("link")
        page_content = fetch_full_content(page_url)

        results.append({
            "title": item.get("title", f"Google Search {google_search_loop_count + 1}, Result {i}"),
            "url": page_url,
            "content": page_content
        })

    # Sort results by content length (descending order)
    results.sort(key=lambda x: len(x["content"]), reverse=True)
    
    return {"results": results}