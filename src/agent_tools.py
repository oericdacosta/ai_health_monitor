# src/agent_tools.py

import os
from dotenv import load_dotenv
from typing import List

# LangChain Imports
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_core.tools import Tool
from langchain_community.tools import TavilySearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

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


def criar_ferramenta_rag_noticias() -> Tool:
    """
    Cria uma ferramenta RAG completa que:
    1. Busca notícias recentes usando Tavily.
    2. Processa e segmenta o conteúdo.
    3. Cria um banco de dados vetorial em memória.
    4. Retorna os trechos mais relevantes para a pergunta.
    """
    print("\n--- Configurando a Ferramenta RAG de Notícias (NewsRAGTool) ---")
    load_dotenv()
    
    if not os.getenv("TAVILY_API_KEY"):
        raise ValueError("A chave de API TAVILY_API_KEY não foi encontrada no arquivo .env")

    # Ferramenta interna para a busca inicial
    tavily_search = TavilySearchResults(max_results=5)

    def rag_pipeline(query: str) -> str:
        """Executa o pipeline completo de RAG para uma dada consulta."""
        print(f"\n[NewsRAGTool] Iniciando pipeline para a consulta: '{query}'")

        # 1. Busca em Tempo Real (Retrieval - Etapa 1)
        print("[NewsRAGTool] Buscando notícias com Tavily...")
        raw_documents = tavily_search.invoke(query)
        
        if not raw_documents:
            return "Nenhuma notícia relevante encontrada."

        # 2. Parsing e Chunking
        print("[NewsRAGTool] Processando e segmentando os documentos...")
        documents = [Document(page_content=doc["content"], metadata={"source": doc["url"], "title": doc["title"]}) for doc in raw_documents]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)
        
        # 3. Embeddings e Banco Vetorial (Indexação)
        print("[NewsRAGTool] Criando embeddings e indexando em FAISS...")
        embeddings_model = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings_model)
        
        # 4. Recuperação Final (Retrieval - Etapa 2)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Pega os 3 chunks mais relevantes
        retrieved_chunks = retriever.invoke(query)
        
        # 5. Formatação da Saída
        contexto = "\n\n---\n\n".join(
            f"Fonte: {doc.metadata.get('title', doc.metadata.get('source'))}\nConteúdo: {doc.page_content}"
            for doc in retrieved_chunks
        )
        print("[NewsRAGTool] Pipeline concluído. Contexto relevante extraído.")
        return contexto

    # Encapsula o pipeline RAG completo como uma única ferramenta para o agente
    news_rag_tool = Tool(
        name="ferramenta_rag_noticias",
        description="""
            Use esta ferramenta para buscar e extrair informações contextuais de notícias recentes sobre saúde,
            especialmente sobre Síndrome Respiratória Aguda Grave (SRAG), COVID-19, Influenza e vacinação.
            A entrada deve ser um tópico de busca claro para encontrar contexto relevante.
            Exemplo: 'contexto sobre o aumento de casos de SRAG em crianças no Brasil 2025'
        """,
        func=rag_pipeline
    )

    print("✅ Ferramenta RAG de Notícias configurada com sucesso.")
    return news_rag_tool