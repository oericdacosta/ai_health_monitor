import streamlit as st
import os
import asyncio
from langchain_core.messages import HumanMessage
from build_database import main as build_db
from src.data_extractor import main as extract_data
from src.chart_generator import generate_and_save_charts
from src.agent_orchestrator import construir_grafo_agente

st.set_page_config(page_title="Monitor de SRAG - Indicium HealthCare", layout="wide")

st.title("🤖 Agente de Monitoramento de SRAG")
st.caption("Uma solução de IA Generativa para análise de dados de saúde da Indicium HealthCare Inc.")
st.markdown("---")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def inicializar_agente():
    """Constrói e retorna o agente LangGraph, mantendo-o em cache."""
    return construir_grafo_agente()

st.subheader("Relatório de Análise Automatizado")
st.write(
    "Clique no botão abaixo para iniciar o pipeline completo. O processo envolve a verificação dos dados, "
    "a geração de gráficos e a execução de um agente de IA para gerar um relatório de monitoramento."
)

if st.button("Gerar Relatório Completo", type="primary", use_container_width=True):

    try:
        with st.status("1/4 - Verificando o ambiente de dados...", expanded=True) as status:
            st.write("Verificando a existência de arquivos de dados (.parquet)...")
            data_dir = os.path.join(PROJECT_ROOT, "data")
            os.makedirs(data_dir, exist_ok=True)
            parquet_files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]

            if not parquet_files:
                st.write("⚠️ Nenhum arquivo .parquet encontrado. Iniciando extração do OpenDataSUS...")
                asyncio.run(extract_data())
                st.write("✅ Extração de dados concluída.")
            else:
                st.write(f"✅ {len(parquet_files)} arquivos de dados encontrados.")

            st.write("Verificando a existência do banco de dados (srag_data.db)...")
            db_path = os.path.join(data_dir, "srag_data.db")
            if not os.path.exists(db_path):
                st.write("⚠️ Banco de dados não encontrado. Construindo agora (ETL)...")
                build_db()
                st.write("✅ Banco de dados construído com sucesso.")
            else:
                st.write("✅ Banco de dados já existe.")
            
            status.update(label="✅ Ambiente de dados verificado com sucesso!", state="complete", expanded=False)

        with st.status("2/4 - Gerando visualizações gráficas...", expanded=True) as status:
            st.write("Iniciando a geração dos gráficos de casos diários e mensais...")
            generate_and_save_charts()
            st.write("✅ Gráficos salvos com sucesso na pasta /img.")
            status.update(label="✅ Gráficos gerados!", state="complete", expanded=False)

        with st.status("3/4 - Agente de IA está analisando os dados...", expanded=True) as status:
            st.write("Inicializando o agente de IA (pode levar um momento)...")
            agent_app = inicializar_agente()
            st.write("Agente inicializado. Enviando missão de análise...")

            missao = (
                    """
                        Você é um Analista Sênior de Saúde Pública especializado em epidemiologia e vigilância 
                        da Síndrome Respiratória Aguda Grave (SRAG) no Brasil. Sua missão é elaborar um relatório 
                        técnico final, claro, objetivo e orientado à tomada de decisão em saúde pública.\n\n

                        Utilize exclusivamente os dados já obtidos na conversa, incluindo:\n
                        - Métricas SQL\n
                        - Contexto de notícias recentes\n
                        - Sinais epidemiológicos extraídos previamente\n\n

                        O relatório deve:\n
                        1) Apresentar e interpretar as quatro métricas principais:\n
                           - Taxa de aumento de casos\n
                           - Taxa de mortalidade\n
                           - Taxa de ocupação de UTI\n
                           - Taxa de vacinação\n
                        2) Relacionar cada métrica com o contexto atual do país e com informações de notícias.\n
                        3) Fornecer análise rigorosa, objetiva, fundamentada e sem sensacionalismo.\n
                        4) Indicar limitações dos dados (ex.: atraso de notificação, sub-registro) quando pertinente.\n
                        5) Informar explicitamente que os gráficos de casos diários e mensais já foram gerados 
                        e estão disponíveis como arquivos de imagem externos.\n\n

                        Formato obrigatório do relatório:\n
                        ## Relatório de Monitoramento de SRAG\n\n
                        ### Resumo Executivo\n
                        - Apresente a visão geral da situação epidemiológica e principais sinais.\n\n
                        ### Métricas Principais\n
                        - Liste e interprete os valores das métricas.\n\n
                        ### Visualizações\n
                        - Informe que os gráficos foram salvos:\n
                          - Casos Diários (Últimos 30 dias)\n
                          - Casos Mensais (Últimos 12 meses)\n\n
                        ### Contexto Recente\n
                        - Sintetize insights das notícias e sua relação com a situação epidemiológica.\n\n
                        Mantenha tom técnico, claro, conciso e alinhado às práticas oficiais de vigilância.
                    """
                    )

            
            inputs = {"messages": [HumanMessage(content=missao)]}
            final_report = None
            
            st.write("O agente está buscando métricas e notícias... Este processo pode levar um momento.")
            for s in agent_app.stream(inputs):
                node_name = list(s.keys())[0]
                st.write(f"Executando nó do agente: `{node_name}`...")
                if "generate_report" in s:
                    final_report_message = s["generate_report"]["messages"][-1]
                    if hasattr(final_report_message, 'content'):
                        final_report = final_report_message.content

            st.write("✅ Análise concluída. Sintetizando o relatório final.")
            status.update(label="✅ Análise do Agente de IA concluída!", state="complete", expanded=False)

        st.markdown("---")
        st.subheader("4/4 - Relatório e Resultados Finais")

        if final_report:
            st.markdown("### Relatório Gerado pelo Agente")
            st.markdown(final_report)
            
            st.markdown("### Visualizações Geradas")
            
            img_dir = os.path.join(PROJECT_ROOT, "img")
            grafico_diario = os.path.join(img_dir, "grafico_diario.png")
            grafico_mensal = os.path.join(img_dir, "grafico_mensal.png")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if os.path.exists(grafico_diario):
                    st.image(grafico_diario, caption="Casos diários de SRAG nos últimos 30 dias.")
                else:
                    st.warning("Arquivo do gráfico diário (grafico_diario.png) não foi encontrado.")
            
            with col2:
                if os.path.exists(grafico_mensal):
                    st.image(grafico_mensal, caption="Casos mensais de SRAG nos últimos 12 meses.")
                else:
                    st.warning("Arquivo do gráfico mensal (grafico_mensal.png) não foi encontrado.")
        else:
            st.error("Ocorreu um erro e o agente não conseguiu gerar o relatório final. Verifique os logs do terminal para mais detalhes.")

    except Exception as e:
        st.error(f"O pipeline falhou com o seguinte erro: {e}")