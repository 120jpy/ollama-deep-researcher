import json
import textwrap
import logging
from typing_extensions import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt

from assistant.configuration import Configuration, SearchAPI
from assistant.utils import deduplicate_and_format_sources, tavily_search, format_sources, perplexity_search, google_search
from assistant.state import SummaryState, SummaryStateInput, SummaryStateOutput
from assistant.prompts import query_writer_instructions, summarizer_instructions, reflection_instructions

logger = logging.getLogger('langsmith')
logger.setLevel(logging.DEBUG)  # Set the logging level

# Nodes
def generate_query(state: SummaryState, config: RunnableConfig):
    """ Generate a query for web search """

    configurable = Configuration.from_runnable_config(config)

    # Format the prompt
    query_writer_instructions_formatted = query_writer_instructions.format(research_topic=state.research_topic)

    # Generate a query
    llm_json_mode = ChatOllama(base_url=configurable.ollama_base_url, model=configurable.local_llm, temperature=0, format="json")
    result = llm_json_mode.invoke(
        [SystemMessage(content=query_writer_instructions_formatted),
        HumanMessage(content=f"Generate a query for web search:")]
    )
    query = json.loads(result.content)

    return {"search_query": query['query']}

def user_question(state: SummaryState, config: RunnableConfig):
    # Configure
    configurable = Configuration.from_runnable_config(config)
    user_response = ""

    if configurable.user_question is True :
        llm = ChatOllama(
            base_url=configurable.ollama_base_url,
            model=configurable.local_llm,
            temperature=0.7,
            format="json"  # JSON形式の出力を期待
        )

        # English
        #prompt = (
        #    f"You are a research assistant helping a user gather more information on the topic: '{state.research_topic}'.\n\n"
        #    "Your task is to generate a meaningful follow-up question that will clarify or expand upon the research topic.\n"
        #    "Focus primarily on the research topic itself rather than relying too much on existing summaries.\n\n"
        #    "<Reference Summary (for context, not strict guidance)>\n"
        #    f"{state.running_summary}\n"
        #    "<End of Reference Summary>\n\n"
        #    "Ensure your question explores new perspectives, deepens understanding, or introduces critical subtopics.\n"
        #    "Consider angles such as:\n"
        #    "- What impact does [specific aspect] have on the overall topic?\n"
        #    "- How can we further investigate [particular element] within this context?\n"
        #    "- In what ways could [related concept] influence our understanding of this topic?\n\n"
        #    "Generate a single well-structured follow-up question that enhances the research direction.\n"
        #    "Return your response as a JSON object with a single key 'question'."
        #)

        prompt = (
                    f"あなたはリサーチアシスタントです。ユーザーが以下の研究テーマについてより多くの情報を収集できるよう支援してください。\n\n"
                    f"【研究テーマ】\n{state.research_topic}\n\n"
                    "あなたの役割は、この研究テーマに関して有益なフォローアップ質問を考えることです。\n"
                    "既存の要約は参考情報として提供しますが、内容に過度に依存せず、新しい視点を取り入れてください。\n\n"
                    "【参考要約（あくまで参考として）】\n"
                    f"{state.running_summary}\n"
                    "【参考要約ここまで】\n\n"
                    "研究を深めるために、以下のような観点から質問を考えてください。\n"
                    "- 特定の要素が全体のテーマにどのような影響を与えるか？\n"
                    "- このテーマに関連する新たな視点や課題はあるか？\n"
                    "- さらなる調査を進めるために、どのようなデータや分析が必要か？\n\n"
                    "研究の方向性を発展させる、具体的で洞察に富んだフォローアップ質問を1つ生成してください。\n"
                    "出力は、以下のフォーマットでJSONオブジェクトとして返してください。\n\n"
                    '{"question": "ここに質問を記述"}'
                )
        
        # LLM に質問を生成させる
        result = llm.invoke([SystemMessage(content=prompt)])
        
        try:
            response_json = json.loads(result.content)
            generated_question = response_json.get('question', "Can you provide more details on your research topic?")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response: {e}")
            generated_question = "Can you provide more details on your research topic?"

        # ユーザーからの回答を入力として受け付ける
        user_response = interrupt(f"{generated_question}")
        state.ollama_question = generated_question
        state.user_input = user_response

    return {"info": user_response}


def web_research(state: SummaryState, config: RunnableConfig):
    """ Gather information from the web """

    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Handle both cases for search_api:
    # 1. When selected in Studio UI -> returns a string (e.g. "tavily")
    # 2. When using default -> returns an Enum (e.g. SearchAPI.TAVILY)
    if isinstance(configurable.search_api, str):
        search_api = configurable.search_api
    else:
        search_api = configurable.search_api.value

    # Search the web
    if search_api == "tavily":
        search_results = tavily_search(state.search_query, include_raw_content=True, max_results=1)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=True)
    elif search_api == "perplexity":
        search_results = perplexity_search(state.search_query, state.research_loop_count)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=False)
    elif search_api == "googlesearch":
        search_results = google_search(state.search_query, state.research_loop_count,config)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=False)
    else:
        raise ValueError(f"Unsupported search API: {configurable.search_api}")

    return {"sources_gathered": [format_sources(search_results)], "research_loop_count": state.research_loop_count + 1, "web_research_results": [search_str]}

def split_text(text:str):
    """指定された長さでテキストを分割"""
    length = len(text) / 2
    return textwrap.wrap(text, length)

def sumarize_ollama_split(state: SummaryState, config: RunnableConfig):

    ret_content = ""

    # Existing summary
    existing_summary = state.running_summary

    # Most recent web research
    most_recent_web_research = state.web_research_results[-1]

    chunks = split_text(most_recent_web_research)

    for text in chunks:
        # Build the human message
        if existing_summary:
            if state.user_input :
                human_message_content = (
                    f"<Research Topic>\n"
                    f"{state.research_topic}\n"
                    f"<Research Topic>\n\n"
                    f"<Supplementary Information>\n"
                    f"User Question: {state.ollama_question}\n"
                    f"User Answer: {state.user_input}\n"
                    f"<Supplementary Information>\n\n"
                    f"<Existing Summary>\n"
                    f"{existing_summary}\n"
                    f"<Existing Summary>\n\n"
                    f"<New Search Results>\n"
                    f"{text}\n"
                    f"<New Search Results>\n\n"
                    f"Generate a comprehensive and updated summary based primarily on the research topic, while also considering the supplementary information, existing summary, and latest search results."
                )
            else:
                human_message_content = (
                    f"<Research Topic>\n"
                    f"{state.research_topic}\n"
                    f"<Research Topic>\n\n"
                    f"<Existing Summary>\n"
                    f"{existing_summary}\n"
                    f"<Existing Summary>\n\n"
                    f"<New Search Results>\n"
                    f"{text}\n"
                    f"<New Search Results>\n\n"
                    f"Generate a comprehensive and updated summary based on the research topic, existing summary, and the latest search results."
                )
        else:
            if state.user_input :
                human_message_content = (
                    f"<Research Topic>\n"
                    f"{state.research_topic}\n"
                    f"<Research Topic>\n\n"
                    f"<Supplementary Information>\n"
                    f"User Question: {state.ollama_question}\n"
                    f"User Answer: {state.user_input}\n"
                    f"<Supplementary Information>\n\n"
                    f"<New Search Results>\n"
                    f"{text}\n"
                    f"<New Search Results>\n\n"
                    f"Generate a comprehensive and updated summary based primarily on the research topic, while also considering the supplementary information, latest search results."
                )
            else:
                human_message_content = (
                    f"<Research Topic>\n"
                    f"{state.research_topic}\n"
                    f"<Research Topic>\n\n"
                    f"<New Search Results>\n"
                    f"{text}\n"
                    f"<New Search Results>\n\n"
                    f"Generate a comprehensive and updated summary based on the research topic, the latest search results."
                )

        # Run the LLM
        configurable = Configuration.from_runnable_config(config)
        llm = ChatOllama(base_url=configurable.ollama_base_url, model=configurable.local_llm, temperature=0.7)
        result = llm.invoke(
            [SystemMessage(content=summarizer_instructions),
            HumanMessage(content=human_message_content)]
        )
        
        ret_content += result.content + "\n"

    return ret_content

def summarize_sources(state: SummaryState, config: RunnableConfig):
    """ Summarize the gathered sources """

    content = sumarize_ollama_split(state,config)

    running_summary = content

    # TODO: This is a hack to remove the <think> tags w/ Deepseek models
    # It appears very challenging to prompt them out of the responses
    while "<think>" in running_summary and "</think>" in running_summary:
        start = running_summary.find("<think>")
        end = running_summary.find("</think>") + len("</think>")
        running_summary = running_summary[:start] + running_summary[end:]

    return {"running_summary": running_summary}

def reflect_on_summary(state: SummaryState, config: RunnableConfig):
    """ Reflect on the summary and generate a follow-up query """

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = ChatOllama(base_url=configurable.ollama_base_url, model=configurable.local_llm, temperature=0, format="json")
    result = llm_json_mode.invoke(
        [SystemMessage(content=reflection_instructions.format(research_topic=state.research_topic)),
        HumanMessage(content=f"Identify a knowledge gap and generate a follow-up web search query based on our existing knowledge: {state.running_summary}")]
    )
    follow_up_query = json.loads(result.content)

    # Get the follow-up query
    query = follow_up_query.get('follow_up_query')

    # JSON mode can fail in some cases
    if not query:

        # Fallback to a placeholder query
        return {"search_query": f"Tell me more about {state.research_topic}"}

    # Update search query with follow-up query
    return {"search_query": follow_up_query['follow_up_query']}

def finalize_summary(state: SummaryState, config: RunnableConfig):
    """ Finalize the summary """

    configurable = Configuration.from_runnable_config(config)

    llm = ChatOllama(base_url=configurable.ollama_base_url, model=configurable.local_llm, temperature=0.7)
    prompt = "以下の文章を日本語に翻訳して。 また、翻訳した文章のみを出力して \n" + state.running_summary
    result = llm.invoke([SystemMessage(prompt)])
    
    # Format all accumulated sources into a single bulleted list
    all_sources = "\n".join(source for source in state.sources_gathered)
    state.running_summary = f"## Summary\n\n{result.content}\n\n ### Sources:\n{all_sources}"
    return {"running_summary": state.running_summary}

def route_research(state: SummaryState, config: RunnableConfig) -> Literal["finalize_summary", "web_research"]:
    """ Route the research based on the follow-up query """

    configurable = Configuration.from_runnable_config(config)
    if state.research_loop_count <= configurable.max_web_research_loops:
        return "web_research"
    else:
        return "finalize_summary"
    
def pre_web_research(state: SummaryState, config: RunnableConfig):
    return web_research(state,config)

def pre_summarize_sources(state: SummaryState, config: RunnableConfig):
    return summarize_sources(state,config)


builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput, config_schema=Configuration)
builder.add_node("generate_query", generate_query)
builder.add_node("pre_web_research", pre_web_research)
builder.add_node("pre_summarize_sources", pre_summarize_sources)
builder.add_node("user_question", user_question)
builder.add_node("web_research", web_research)
builder.add_node("summarize_sources", summarize_sources)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("finalize_summary", finalize_summary)

builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "pre_web_research")
builder.add_edge("pre_web_research", "pre_summarize_sources")
builder.add_edge("pre_summarize_sources", "user_question")
builder.add_edge("user_question", "web_research")
builder.add_edge("web_research", "summarize_sources")
builder.add_edge("summarize_sources", "reflect_on_summary")
builder.add_conditional_edges("reflect_on_summary", route_research)
builder.add_edge("finalize_summary", END)

"""
builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "user_question")
builder.add_edge("user_question", "web_research")
builder.add_edge("web_research", "summarize_sources")
builder.add_edge("summarize_sources", "reflect_on_summary")
builder.add_conditional_edges("reflect_on_summary", route_research)
builder.add_edge("finalize_summary", END)
"""

graph = builder.compile()