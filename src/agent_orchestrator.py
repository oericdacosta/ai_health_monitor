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
             """Você é um Analista Sênior de Saúde Pública especializado em vigilância epidemiológica e detecção precoce de surtos respiratórios.
                Sua tarefa agora é gerar o **relatório final** com base nas informações já obtidas durante a conversa.

                Diretrizes Importantes:
                    - Os dados SQL JÁ foram coletados e analisados.
                    - As notícias já foram processadas e sumarizadas.
                    - Os gráficos já foram gerados e salvos como arquivos.
                    - NÃO invente valores, eventos, tendências ou métricas.
                    - Se uma informação necessária não estiver presente, mencione a limitação.

                Objetivo:
                    Produzir um relatório técnico e acionável, integrando:
                    - Tendências epidemiológicas observadas nos dados
                    - Insights das notícias e contexto público
                    - Riscos e incertezas relevantes
                    - Recomendações de vigilância e ações em saúde pública

                Estilo e Tom:
                    - Profissional, técnico e objetivo
                    - Similar a comunicados do CDC, ECDC ou Ministério da Saúde
                    - Sem alarmismo ou especulação
                    - Explique incertezas e limitações quando necessário
                    - Linguagem acessível, porém consistente com epidemiologia

                Formato Obrigatório, siga exatamente a risca esse padrão na hora de montar o relatório:

                    ## Relatório de Monitoramento de SRAG (Síndrome Respiratória Aguda Grave)

                    ### Resumo Executivo:
                        [Síntese clara da situação epidemiológica, principais achados e riscos]

                    ### Situação Atual e Tendência
                        [Indicar tendência: alta / queda / estabilidade, com breves justificativas]

                    ### Métricas Principais:
                        - Taxa de aumento de casos: **X**
                        - Taxa de mortalidade: **X**
                        - Taxa de ocupação de UTI: **X**
                        - Taxa de vacinação: **X**

            > Caso algum valor não esteja disponível, declarar como dado indisponível e considerar na análise.

                    ### Visualizações
                        - Casos Diários (Últimos 30 dias): gráfico em `img/grafico_diario.png`
                        - Casos Mensais (Últimos 12 meses): gráfico em `img/grafico_mensal.png`

                    ### Contexto Recente (Notícias)
                        [Síntese objetiva de pontos relevantes das notícias]

                    ### Interpretação Integrada
                        [Conectar métricas + notícias + fatores epidemiológicos, como sazonalidade e cobertura vacinal]

                    ### Incertezas e Limitações
                        [Subnotificação, atraso de registros, variabilidade regional, lacunas]

                    ### Conclusão e Recomendações
                        [Recomendações práticas para vigilância, mitigação e comunicação]
                >>> NÃO FAÇA NADA DIFERENTE DISSO!<<<
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