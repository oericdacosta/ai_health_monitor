# src/agent_tools.py

import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_core.tools import Tool
from langchain_community.tools import TavilySearchResults

def criar_ferramenta_sql() -> Tool:
    """
    Cria e configura um agente SQL completo e o encapsula como uma única ferramenta.
    Utiliza SQLDatabaseToolkit e create_sql_agent com guardrails de segurança via prompt.
    """
    print("\n--- Configurando a Ferramenta de Consulta SQL (SQLTool) ---")

    load_dotenv()

    db_path = os.path.join("data", "srag_data.db")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Banco de dados não encontrado em {db_path}. Execute 'build_database.py' primeiro.")
    
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    print(f"Conectado ao banco de dados. Tabelas disponíveis: {db.get_usable_table_names()}")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    system_prompt_guardrail = f"""
    Você é um agente especialista em SQL projetado para interagir com um banco de dados de casos de SRAG.
    Dada uma pergunta do usuário, sua tarefa é gerar uma consulta SQL sintaticamente correta para o dialeto '{db.dialect}', executá-la e retornar a resposta.

    **REGRAS DE SEGURANÇA ESTRITAS:**
    - **NUNCA, sob nenhuma circunstância, execute comandos DML (INSERT, UPDATE, DELETE, DROP, etc.).** Se o usuário pedir para modificar dados, recuse-se educadamente.
    
    **DIRETRIZES DE FORMATAÇÃO E CONSULTA:**
    - Ao gerar a query para a ferramenta `sql_db_query`, retorne **APENAS o código SQL puro**, sem nenhuma formatação extra, como blocos de código Markdown (```sql...```).
    - Após executar a query e obter o resultado, formule a resposta final para o usuário.
    - Sua resposta final DEVE estar no formato: `Final Answer: [sua resposta aqui]`.
    - Sempre limite suas consultas (com `LIMIT`) para um número razoável de linhas, a menos que a pergunta seja uma agregação.

    **Schema da Tabela `casos_srag`:**
    - `data_sintomas` (DATE): Data de início dos sintomas ('YYYY-MM-DD').
    - `uf` (TEXT): Sigla do estado de residência.
    - `sexo` (TEXT): 'Masculino' ou 'Feminino'.
    - `idade` (INTEGER): Idade em anos.
    - `uti` (TEXT): 'Sim' ou 'Não'.
    - `evolucao` (TEXT): 'Cura' ou 'Óbito'.
    - `vacina_covid` (TEXT): 'Sim' ou 'Não'.
    - `data_entrada_uti`, `data_saida_uti`, `data_dose1_covid` (DATE): Podem ser nulos.
    """

    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        prefix=system_prompt_guardrail,
        handle_parsing_errors=True
    )

    sql_agent_tool = Tool(
        name="ferramenta_consulta_sql",
        description="""
            Use esta ferramenta para responder perguntas sobre dados de SRAG, como contagens, médias, taxas e distribuições.
            A entrada deve ser uma pergunta completa em linguagem natural.
            Exemplo: 'Qual o número total de óbitos no estado de São Paulo no ano de 2023?'
        """,
        func=lambda question: agent_executor.invoke({"input": question})
    )

    print("✅ Agente SQL encapsulado como ferramenta com sucesso.")
    return sql_agent_tool


def criar_ferramenta_busca_noticias() -> Tool:
    """
    Cria e configura a ferramenta de busca de notícias em tempo real usando Tavily.
    """
    print("\n--- Configurando a Ferramenta de Busca de Notícias (NewsTool) ---")
    load_dotenv()
    
    if not os.getenv("TAVILY_API_KEY"):
        raise ValueError("A chave de API TAVILY_API_KEY não foi encontrada no arquivo .env")

    # Inicializa a ferramenta TavilySearchResults, buscando os 5 resultados mais relevantes.
    tavily_tool = TavilySearchResults(
        name="ferramenta_busca_noticias",
        description="""
            Use esta ferramenta para buscar notícias e informações recentes sobre saúde,
            especialmente sobre Síndrome Respiratória Aguda Grave (SRAG), COVID-19, Influenza e vacinação.
            A entrada deve ser um tópico de busca claro e conciso.
            Exemplo: 'aumento de casos de SRAG em crianças no Brasil 2025'
        """,
        max_results=5
    )

    print("✅ Ferramenta de Busca de Notícias configurada com sucesso.")
    return tavily_tool