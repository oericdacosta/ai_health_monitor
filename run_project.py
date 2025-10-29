# run_project.py

import os
import asyncio
from rich.console import Console

from src.data_extractor import main as extract_data
from build_database import main as build_db
from app import main as run_agent_in_terminal

console = Console()

def main():
    """
    Orquestra a execução completa do projeto:
    1. Garante que os dados brutos (.parquet) existam.
    2. Garante que o banco de dados (.db) exista e esteja atualizado.
    3. Inicia a aplicação do agente no terminal.
    """
    console.rule("[bold green]Iniciando o Projeto de Monitoramento SRAG[/bold green]")

    project_root = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    parquet_files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]

    if not parquet_files:
        console.print("[yellow]Nenhum arquivo .parquet encontrado. Iniciando extração automática...[/yellow]")
        asyncio.run(extract_data())
    else:
        console.print(f"[green]✅ {len(parquet_files)} arquivos .parquet encontrados.[/green]")

    db_path = os.path.join(data_dir, "srag_data.db")
    if not os.path.exists(db_path):
        console.print(f"[yellow]Banco de dados '{os.path.basename(db_path)}' não encontrado. Construindo agora...[/yellow]")
        build_db()
    else:
        console.print(f"[green]✅ Banco de dados '{os.path.basename(db_path)}' já existe.[/green]")

    console.rule("[bold cyan]Iniciando o Agente no Terminal[/bold cyan]")
    run_agent_in_terminal()

if __name__ == "__main__":
    main()