# src/agent_orchestrator.py

import operator
from typing import List, Annotated, TypedDict, Literal
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .agent_tools import criar_ferramenta_sql, criar_ferramenta_rag_noticias

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

def construir_grafo_agente():
    """
    Constrói o grafo completo do agente, incluindo o Nó de Síntese para o relatório final.
    """
    print("\n--- Construindo o Grafo do Agente Orquestrador ---")

    ferramenta_sql = criar_ferramenta_sql()
    ferramenta_rag = criar_ferramenta_rag_noticias()
    tools = [ferramenta_sql, ferramenta_rag]
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    agent_brain = llm.bind_tools(tools)
    
    # --- Definição dos Nós (Nodes) ---
    
    def call_model(state: AgentState):
        """O cérebro do agente: invoca o LLM para decidir o próximo passo."""
        print("\n>> Cérebro do Agente: Decidindo o próximo passo...")
        response = agent_brain.invoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    def generate_report_node(state: AgentState):
        """Nó final que sintetiza os dados coletados em um relatório estruturado."""
        print("\n>> Nó de Síntese (Report_Node): Gerando o relatório final...")

        prompt_template = ChatPromptTemplate.from_messages([
            ("system",
             """Você é um Analista de Saúde Pública sênior, especialista em análise de dados epidemiológicos.
             Sua tarefa é gerar um relatório de monitoramento conciso e informativo sobre a Situação da Síndrome Respiratória Aguda Grave (SRAG).

             Use o processo de Chain-of-Thought (CoT) abaixo para estruturar seu raciocínio:
             1.  **Análise das Métricas:** Revise os dados numéricos e métricas fornecidos pela ferramenta de SQL. Identifique os números chave.
             2.  **Contextualização com Notícias:** Use o contexto de notícias fornecido pela ferramenta de busca para explicar o PORQUÊ dos números. Conecte os dados com os eventos recentes.
             3.  **Síntese e Geração:** Combine a análise dos dados e o contexto das notícias para escrever um resumo executivo claro. Em seguida, formate o relatório final estritamente de acordo com a seção [FORMATO DE SAÍDA].

             **NUNCA** inclua informações que não foram fornecidas nos resultados das ferramentas. Seu relatório deve ser 100% fundamentado nos dados apresentados."""),
            MessagesPlaceholder(variable_name="messages"),
            ("system",
             """
             ---
             [DADOS E CONTEXTO COLETADOS PELAS FERRAMENTAS ACIMA]
             ---
             Agora, aplique o processo de Chain-of-Thought e gere o relatório final no formato Markdown abaixo.

             [FORMATO DE SAÍDA]
             ## Relatório de Monitoramento de SRAG

             ### Resumo Executivo
             [Escreva aqui sua análise sintetizada, conectando os dados SQL com o contexto das notícias.]

             ### Métricas Principais
             - **Total de Óbitos (período consultado):** [valor]
             - **Taxa de Mortalidade (se calculável):** [valor]
             - **Taxa de Ocupação de UTI (se calculável):** [valor]
             - **Outros dados relevantes:** [valor]

             ### Contexto Recente
             [Resuma os pontos chave encontrados nas notícias.]
             """
             )
        ])
        
        report_chain = prompt_template | llm
        
        final_report = report_chain.invoke({"messages": state["messages"]})
        return {"messages": [final_report]}

    def should_continue(state: AgentState) -> Literal["action", "generate_report"]:
        """Roteador: Decide se continua chamando ferramentas ou gera o relatório."""
        print("\n>> Roteador: Verificando a decisão do agente...")
        last_message = state['messages'][-1]
        
        if last_message.tool_calls:
            print("   - Decisão: Chamar uma ferramenta. Roteando para o Nó de Ação.")
            return "action"
        
        print("   - Decisão: Dados coletados. Roteando para o Nó de Síntese (Report_Node).")
        return "generate_report"

    # --- Montagem do Grafo (StateGraph) ---
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("agent", call_model)
    graph_builder.add_node("action", tool_node)
    graph_builder.add_node("generate_report", generate_report_node)

    graph_builder.set_entry_point("agent")

    graph_builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "action": "action",
            "generate_report": "generate_report"
        },
    )

    graph_builder.add_edge("action", "agent")
    graph_builder.add_edge("generate_report", END)

    agent_graph = graph_builder.compile()
    print("✅ Grafo do agente com Nó de Síntese compilado com sucesso!")
    
    return agent_graph