# src/agent_orchestrator.py

# *** CORREÇÃO APLICADA AQUI ***
# Adicionamos TypedDict à lista de importações do módulo typing.
from typing import List, Literal, Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# Importa nossas funções de criação de ferramentas
from .agent_tools import criar_ferramenta_sql, criar_ferramenta_rag_noticias

# -----------------------------------------------------------------------------
# 1. DEFINIÇÃO DO ESTADO DO AGENTE (COM REDUTOR)
# -----------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# -----------------------------------------------------------------------------
# 2. DEFINIÇÃO DO GRAFO DO AGENTE
# -----------------------------------------------------------------------------

def construir_grafo_agente():
    """
    Constrói o grafo do agente LangGraph, definindo os nós, as arestas
    e o fluxo de raciocínio ReAct.
    """
    print("\n--- Construindo o Grafo do Agente Orquestrador ---")

    print("1. Configurando LLM e vinculando ferramentas...")
    ferramenta_sql = criar_ferramenta_sql()
    ferramenta_rag = criar_ferramenta_rag_noticias()
    tools = [ferramenta_sql, ferramenta_rag]
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    agent_brain = llm.bind_tools(tools)
    print("✅ LLM e ferramentas configurados.")

    # --- Definição dos Nós (Nodes) ---
    def call_model(state: AgentState):
        """O cérebro do agente: invoca o LLM para decidir o próximo passo."""
        print("\n>> Cérebro do Agente: Decidindo o próximo passo...")
        response = agent_brain.invoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    def should_continue(state: AgentState) -> Literal["action", "__end__"]:
        """Roteador: Verifica a decisão do agente para direcionar o fluxo."""
        print("\n>> Roteador: Verificando a decisão do agente...")
        last_message = state['messages'][-1]
        if last_message.tool_calls:
            print("   - Decisão: Chamar uma ferramenta. Roteando para o Nó de Ação.")
            return "action"
        print("   - Decisão: Gerar resposta final. Roteando para o Fim.")
        return END

    # --- Montagem do Grafo (StateGraph) ---
    print("\n2. Montando a arquitetura do grafo...")
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("agent", call_model)
    graph_builder.add_node("action", tool_node)

    graph_builder.set_entry_point("agent")

    graph_builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "action": "action",
            END: END,
        },
    )

    graph_builder.add_edge("action", "agent")

    agent_graph = graph_builder.compile()
    print("✅ Grafo do agente compilado com sucesso!")
    
    return agent_graph