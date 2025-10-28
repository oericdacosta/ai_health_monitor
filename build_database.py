import os
import sqlite3
import asyncio
import pandas as pd
from rich.console import Console

# Importa módulos locais
from src.data_extractor import main as extract_data
from src.data_loader import load_data, rename_columns
from src.data_transformer import prepare_dataframe

console = Console()


def run_extraction():
    """
    Executa a extração de dados SRAG se ainda não houver arquivos .parquet.
    """
    console.rule("[bold cyan]Fase 1 — Extração de Dados SRAG[/bold cyan]")
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    os.makedirs(data_dir, exist_ok=True)

    parquet_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".parquet")
    ]

    if parquet_files:
        console.print(f"[green]✅ {len(parquet_files)} arquivos .parquet já existentes. Pulando extração.[/green]")
    else:
        console.print("[yellow]Nenhum arquivo encontrado. Iniciando extração automática...[/yellow]")
        asyncio.run(extract_data())

    return data_dir


def run_etl(data_dir: str):
    """
    Executa o pipeline completo de ETL (Carregar → Renomear → Transformar → Salvar).
    """
    console.rule("[bold cyan]Fase 2 — Construção do Banco de Dados[/bold cyan]")

    db_path = os.path.join(data_dir, "srag_data.db")
    json_path = os.path.join(data_dir, "dicionario_dados.json")

    essential_columns = [
        "DT_SIN_PRI", "SG_UF", "CS_SEXO", "DT_NASC",
        "UTI", "DT_ENTUTI", "DT_SAIDUTI", "EVOLUCAO",
        "VACINA_COV", "DOSE_1_COV"
    ]

    parquet_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".parquet")
    ]

    if not parquet_files:
        console.print(f"[red]❌ Nenhum arquivo .parquet encontrado em '{data_dir}'.[/red]")
        return

    # Carregamento e transformação
    df_raw = load_data(parquet_files, columns=essential_columns)
    df_renamed = rename_columns(df_raw, json_path=json_path)
    df_final = prepare_dataframe(df_renamed)

    console.print(f"\n💾 Salvando [bold]{len(df_final):,}[/bold] registros limpos no banco de dados...")

    with sqlite3.connect(db_path) as conn:
        df_final.to_sql("casos_srag", conn, if_exists="replace", index=False)

    console.print(f"[green]✅ Banco de dados criado com sucesso![/green]")
    console.print(f"   Caminho: [cyan]{db_path}[/cyan]")
    console.print(f"   Tabela: [bold]casos_srag[/bold] — {len(df_final):,} linhas x {len(df_final.columns)} colunas\n")
    console.rule("[bold green]Pipeline ETL concluído com sucesso![/bold green]")


def main():
    data_dir = run_extraction()
    run_etl(data_dir)


if __name__ == "__main__":
    main()
