import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console

# Ponto de referência robusto para a raiz do projeto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "srag_data.db")
IMG_DIR = os.path.join(PROJECT_ROOT, "img")

console = Console()

def get_max_date(conn):
    """Busca a data mais recente dos sintomas no banco de dados."""
    return pd.read_sql_query("SELECT MAX(data_sintomas) FROM casos_srag;", conn).iloc[0, 0]

def plot_and_save(df, x_col, y_col, title, filename):
    """Função genérica para plotar e salvar um gráfico."""
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=x_col, y=y_col, data=df, marker='o', palette='viridis')
    plt.title(title, fontsize=16)
    plt.xlabel(x_col.replace('_', ' ').title())
    plt.ylabel(y_col.replace('_', ' ').title())
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--')
    plt.tight_layout()
    
    filepath = os.path.join(IMG_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    console.print(f"✅ Gráfico '[cyan]{title}[/cyan]' salvo em: [green]{filepath}[/green]")

def generate_and_save_charts():
    """
    Função principal que se conecta ao DB, executa as queries
    e gera os dois gráficos necessários para o relatório.
    """
    console.rule("[bold yellow]📊 Fase de Geração de Gráficos[/bold yellow]")
    os.makedirs(IMG_DIR, exist_ok=True)

    if not os.path.exists(DB_PATH):
        console.print(f"[red]❌ Erro: Banco de dados não encontrado em {DB_PATH}. Execute o ETL primeiro.[/red]")
        return

    try:
        with sqlite3.connect(DB_PATH) as conn:
            data_referencia = get_max_date(conn)
            console.print(f"Data de referência para os gráficos: [bold]{data_referencia}[/bold]")

            # 1. Gráfico de Casos Diários (Últimos 30 dias)
            query_diario = f"""
            SELECT data_sintomas, COUNT(*) as total_casos
            FROM casos_srag
            WHERE data_sintomas BETWEEN DATE('{data_referencia}', '-29 days') AND '{data_referencia}'
            GROUP BY data_sintomas
            ORDER BY data_sintomas;
            """
            df_diario = pd.read_sql_query(query_diario, conn, parse_dates=['data_sintomas'])
            plot_and_save(df_diario, 'data_sintomas', 'total_casos', 'Casos Diários de SRAG (Últimos 30 dias)', 'grafico_diario.png')

            # 2. Gráfico de Casos Mensais (Últimos 12 meses)
            query_mensal = f"""
            SELECT STRFTIME('%Y-%m', data_sintomas) as mes, COUNT(*) as total_casos
            FROM casos_srag
            WHERE data_sintomas BETWEEN DATE('{data_referencia}', '-12 months') AND '{data_referencia}'
            GROUP BY mes
            ORDER BY mes;
            """
            df_mensal = pd.read_sql_query(query_mensal, conn, parse_dates=['mes'])
            plot_and_save(df_mensal, 'mes', 'total_casos', 'Casos Mensais de SRAG (Últimos 12 meses)', 'grafico_mensal.png')

    except Exception as e:
        console.print(f"[red]❌ Ocorreu um erro ao gerar os gráficos: {e}[/red]")

if __name__ == '__main__':
    generate_and_save_charts()