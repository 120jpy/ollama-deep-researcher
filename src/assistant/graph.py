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

    logger.info("max_web_research_loops : " + str(configurable.max_web_research_loops))
    logger.info("local_llm : " + str(configurable.local_llm))
    logger.info("search_api : " + str(configurable.search_api))
    logger.info("fetch_full_page : " + str(configurable.fetch_full_page))
    logger.info("user_question : " + str(configurable.user_question))
    logger.info("ollama_base_url : " + str(configurable.ollama_base_url))

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

        # LLM に具体的な質問を生成させるプロンプト
        prompt = (f"You are a research assistant helping a user gather more information on a topic."
                f"The topic is: '{state.research_topic}'."
                f"<Existing Summary> \n {state.running_summary} \n <Existing Summary>\n\n"
                f"Generate a meaningful follow-up question to clarify or expand the research topic."
                f"Ensure that your question helps gather useful details."
                f"Return your response as a JSON object with a single key 'question'.")

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
            if state.info :
                human_message_content = (
                    f"<User Input> \n {state.research_topic}  \n {state.info} \n <User Input>\n\n"
                    f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
                    f"<New Search Results> \n {text} \n <New Search Results>"
                )
            else:
                human_message_content = (
                    f"<User Input> \n {state.research_topic} \n <User Input>\n\n"
                    f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
                    f"<New Search Results> \n {text} \n <New Search Results>"
                )
        else:
            if state.info :
                human_message_content = (
                    f"<User Input> \n {state.research_topic} \n {state.info} \n <User Input>\n\n"
                    f"<Search Results> \n {text} \n <Search Results>"
                )
            else:
                human_message_content = (
                    f"<User Input> \n {state.research_topic} \n <User Input>\n\n"
                    f"<Search Results> \n {text} \n <Search Results>"
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