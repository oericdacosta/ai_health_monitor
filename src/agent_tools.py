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
from langchain_community.retrievers import BM25Retriever

def combine_retrievers(retrievers, weights, query, top_k=4):
    """
    Consulta cada retriever, pontua resultados por peso e rank,
    remove duplicatas e retorna os melhores documentos.
    """
    scores = {}
    doc_map = {}

    for retriever, weight in zip(retrievers, weights):
        # Compatível com retrievers novos (invoke) e antigos (get_relevant_documents)
        if hasattr(retriever, "invoke"):
            docs = retriever.invoke(query)
        elif hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(query)
        else:
            raise AttributeError(f"Retriever {type(retriever)} não tem método de busca compatível.")

        for rank, doc in enumerate(docs[:top_k]):
            key = (doc.page_content, doc.metadata.get("source"), doc.metadata.get("title"))
            contribution = weight * (1.0 / (rank + 1))
            scores[key] = scores.get(key, 0.0) + contribution
            doc_map[key] = doc

    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    merged_docs = [doc_map[k] for k in sorted_keys][:top_k]
    return merged_docs


def executar_pipeline_rag(query: str, tavily_search: TavilySearchResults) -> str:
    """
    Executa o pipeline completo de RAG para uma dada consulta.
    """
    print(f"\n[NewsRAGTool] Iniciando pipeline para a consulta: '{query}'")

    print("[NewsRAGTool] Buscando notícias com Tavily...")
    raw_documents = tavily_search.invoke(query)

    if not raw_documents:
        return "Nenhuma notícia relevante encontrada."

    print("[NewsRAGTool] Processando e segmentando os documentos...")
    documents = [Document(page_content=doc["content"], metadata={"source": doc["url"], "title": doc["title"]}) for doc in raw_documents]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        return "Não foi possível processar o conteúdo das notícias."

    print("[NewsRAGTool] Criando retriever semântico (FAISS)...")
    embeddings_model = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings_model)
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    print("[NewsRAGTool] Criando retriever de palavra-chave (BM25)...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 4

    print("[NewsRAGTool] Combinando retrievers manualmente (fallback)...")
    retrieved_docs = combine_retrievers(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.5, 0.5],
        query=query,
        top_k=4
    )

    contexto = "\n\n---\n\n".join(
        f"Fonte: {doc.metadata.get('title', doc.metadata.get('source'))}\nConteúdo: {doc.page_content}"
        for doc in retrieved_docs
    )

    print("[NewsRAGTool] Pipeline concluído. Contexto relevante extraído.")
    return contexto

def criar_ferramenta_sql() -> Tool:
    """
    Cria e configura um agente SQL completo e o encapsula como uma única ferramenta.
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
    Cria uma ferramenta RAG completa com busca híbrida (semântica + palavra-chave).
    """
    print("\n--- Configurando a Ferramenta RAG de Notícias (NewsRAGTool) ---")
    load_dotenv()

    if not os.getenv("TAVILY_API_KEY"):
        raise ValueError("A chave de API TAVILY_API_KEY não foi encontrada no arquivo .env")

    tavily_search = TavilySearchResults(max_results=7)

    news_rag_tool = Tool(
        name="ferramenta_rag_noticias",
        description="""
            Use esta ferramenta para buscar e extrair informações contextuais de notícias recentes sobre saúde,
            especialmente sobre Síndrome Respiratória Aguda Grave (SRAG), COVID-19, Influenza e vacinação.
            A entrada deve ser um tópico de busca claro para encontrar contexto relevante.
        """,
        func=lambda query: executar_pipeline_rag(query, tavily_search)
    )

    print("✅ Ferramenta RAG de Notícias configurada com sucesso.")
    return news_rag_tool
