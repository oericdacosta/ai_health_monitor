# src/agent_orchestrator.py

from typing import Annotated, TypedDict, Literal
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from .agent_tools import criar_ferramenta_sql, criar_ferramenta_rag_noticias

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def construir_grafo_agente():
    """
    Constrói um grafo de agente mais simples, focado em coletar métricas
    e contexto para sintetizar um relatório.
    """
    print("\n--- Construindo o Grafo do Agente (Arquitetura Simplificada) ---")

    ferramenta_sql = criar_ferramenta_sql()
    ferramenta_rag = criar_ferramenta_rag_noticias()

    tools = [ferramenta_sql, ferramenta_rag]
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0, max_retries=3)
    agent_brain = llm.bind_tools(tools)
    
    tool_node = ToolNode(tools)

    def call_model(state: AgentState):
        """O cérebro do agente: decide o próximo passo."""
        print("\n>> Cérebro do Agente: Decidindo o próximo passo (SQL/RAG)...")
        response = agent_brain.invoke(state["messages"])
        return {"messages": [response]}

    def generate_report_node(state: AgentState):
        """Nó final que sintetiza os dados em um relatório."""
        print("\n>> Nó de Síntese: Gerando o relatório final...")
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             """Você é um Analista de Saúde Pública sênior. Sua tarefa é gerar um relatório final.
             A coleta de dados (métricas SQL e contexto de notícias) foi concluída. Os gráficos visuais JÁ FORAM GERADOS e salvos em arquivos separados.
             Sua única tarefa agora é sintetizar todas as informações da conversa em um relatório bem estruturado em Markdown, mencionando que os gráficos estão disponíveis.
             Siga estritamente o formato de saída.
             
             [FORMATO DE SAÍDA]
             ## Relatório de Monitoramento de SRAG

             ### Resumo Executivo
             [Escreva sua análise sintetizada, conectando os dados SQL com o contexto das notícias.]

             ### Métricas Principais
             [Liste as 4 métricas principais com seus valores.]

             ### Visualizações
             - **Casos Diários (Últimos 30 dias):** (Gráfico gerado e salvo em img/grafico_diario.png)
             - **Casos Mensais (Últimos 12 meses):** (Gráfico gerado e salvo em img/grafico_mensal.png)

             ### Contexto Recente
             [Resuma os pontos chave encontrados nas notícias.]
             """),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        report_chain = prompt_template | llm
        final_report = report_chain.invoke({"messages": state["messages"]})
        return {"messages": [final_report]}

    def should_continue(state: AgentState) -> Literal["action", "generate_report"]:
        """Decide se continua chamando ferramentas ou se vai para o relatório final."""
        last_message = state['messages'][-1]
        if last_message.tool_calls:
            return "action"
        return "generate_report"

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", call_model)
    graph_builder.add_node("action", tool_node)
    graph_builder.add_node("generate_report", generate_report_node)
    
    graph_builder.set_entry_point("agent")
    
    graph_builder.add_conditional_edges(
        "agent",
        should_continue,
        {"action": "action", "generate_report": "generate_report"}
    )
    graph_builder.add_edge("action", "agent")
    graph_builder.add_edge("generate_report", END)
    
    agent_graph = graph_builder.compile()
    print("✅ Grafo do agente simplificado compilado com sucesso!")
    return agent_graph