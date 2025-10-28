# src/agent_orchestrator.py

from typing import List, TypedDict
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

from .agent_tools import criar_ferramenta_sql, criar_ferramenta_rag_noticias

class AgentState(TypedDict):
    messages: List[BaseMessage]

def criar_orquestrador():
    """
    Configura e retorna o cérebro do agente, com as ferramentas vinculadas,
    e o grafo de LangGraph inicializado.
    """
    print("\n--- Modelando o Agente Principal (Orquestrador) ---")

    print("1. Criando e vinculando ferramentas ao LLM...")

    ferramenta_sql = criar_ferramenta_sql()
    ferramenta_rag = criar_ferramenta_rag_noticias()
    tools = [ferramenta_sql, ferramenta_rag]

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    agent_brain = llm.bind_tools(tools)
    
    print("✅ LLM configurado e ferramentas vinculadas com sucesso.")

    print("2. Criando a arquitetura StateGraph...")

    graph = StateGraph(AgentState)
    
    print("✅ StateGraph inicializado com sucesso.")

    return agent_brain, tools, graph