import os
import requests
from typing import Dict, Any, List, Optional
from langsmith import traceable
from tavily import TavilyClient
import re
from tabulate import tabulate  # 表を見やすく表示するため
import pdfplumber
import pandas as pd
from selectolax.parser import HTMLParser
from urllib.parse import urljoin
from langchain_core.runnables import RunnableConfig
from assistant.configuration import Configuration
from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama
from trafilatura import extract

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

def fetch_full_content(url: str, config: RunnableConfig) -> str:
    """Fetches and cleans readable text from a webpage, or summarizes a PDF if the URL points to one."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}

        # PDF の場合、直接要約
        if url.lower().endswith(".pdf"):
            return fetch_pdf_text(url, config)

        # HTMLページの処理
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # HTML のテキストを抽出（タグ・CSS 除去）
        full_text = extract(response.text, include_links=False)  # trafilatura を使用
        full_text = full_text.strip() if full_text else "Content extraction failed."

        # HTML をパースして PDF リンクを抽出
        tree = HTMLParser(response.text)
        pdf_summaries = []
        for node in tree.css("a[href]"):
            href = node.attributes.get("href", "")
            if href.lower().endswith(".pdf"):
                pdf_url = urljoin(url, href)  # 絶対URLに変換
                pdf_summary = fetch_pdf_text(pdf_url, config)  # PDF を要約
                if pdf_summary:
                    pdf_summaries.append(f"[PDF] {pdf_url}: {pdf_summary}")

        # テキストと PDF 要約を結合
        if pdf_summaries:
            full_text += "\n\n" + "\n".join(pdf_summaries)

        return full_text

    except requests.RequestException as e:
        return f"Error fetching content (RequestException): {str(e)}"
    except Exception as e:
        return f"Error fetching content (General): {str(e)}"

def clean_table(df):
    """ DataFrame の空白を削除し、None を空文字に変換 """
    df = df.map(lambda x: x.strip() if isinstance(x, str) else ("" if x is None else x))
    # セル内の余分な空白を削除
    df.replace(r"\s+", " ", regex=True, inplace=True)
    # 空白のみのセルを None に変換
    df.replace("", None, inplace=True)
    # 空の列・行を削除
    df.dropna(axis=1, how="all", inplace=True)  # すべて空の列を削除
    df.dropna(axis=0, how="all", inplace=True)  # すべて空の行を削除
    return df

def extract_text_and_tables(pdf_path):

    # 表の抽出設定
    TABLE_SETTINGS = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "intersection_x_tolerance": 5,
        "intersection_y_tolerance": 5
    }
    extracted_data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            elements = []
            
            # 文章を抽出
            text = page.extract_text()
            if text:
                elements.append(("text", text.strip()))
            
            # 表を抽出（設定適用）
            tables = page.extract_tables(TABLE_SETTINGS)
            for table in tables:
                df = pd.DataFrame(table)
                # 空白処理
                df = clean_table(df)
                elements.append(("table", df))
            
            extracted_data.append(elements)
    
    return extracted_data

def format_output(data):
    ret_txt = ""
    for page_num, elements in enumerate(data, start=1):
        ret_txt += f"\n--- Page {page_num} ---\n"
        for elem_type, content in elements:
            if elem_type == "text":
                ret_txt += content
            elif elem_type == "table":
                ret_txt += "\n[Table]\n"
                ret_txt += tabulate(content, headers="keys", tablefmt="grid")  # 表を見やすく整形
    return ret_txt


def fetch_pdf_text(pdf_url: str,config: RunnableConfig) -> str:
    """Downloads, extracts text from a PDF file, and summarizes it using Ollama."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(pdf_url, headers=headers, timeout=10)
        response.raise_for_status()

        with open("/tmp/temp.pdf", "wb") as f:
            f.write(response.content)

        # Extract text using PyMuPDF
        pdfdata = extract_text_and_tables("/tmp/temp.pdf")
        text = format_output(pdfdata)
        if not text.strip():
            return ""
        
        return text
    except:
        return ""

def google_search(query: str, google_search_loop_count: int,config: RunnableConfig) -> Dict[str, Any]:
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
    
    for i in range(2):
        response = requests.get(search_url, params=params)
        
        if response.status_code == 403:
            pass
        else :
            break
    
    response.raise_for_status()
    data = response.json()
    page_content = ""
    
    results = []
    for i, item in enumerate(data.get("items", []), start=1):
        page_url = item.get("link")
        page_content += fetch_full_content(page_url,config)

        results.append({
            "title": item.get("title", f"Google Search {google_search_loop_count + 1}, Result {i}"),
            "url": page_url,
            "content": page_content
        })

    # Sort results by content length (descending order)
    results.sort(key=lambda x: len(x["content"]), reverse=True)
    
    return {"results": results}