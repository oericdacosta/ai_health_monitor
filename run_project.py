# run_project.py

import os
import subprocess
from rich.console import Console

console = Console()

def main():
    """
    Script simplificado para lançar a aplicação Streamlit.
    A lógica de verificação de dados e construção do DB foi movida para dentro
    do app.py para que o status possa ser exibido na interface do usuário.
    """
    console.rule("[bold green]Iniciando o Lançador do Projeto de Monitoramento SRAG[/bold green]")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(project_root, "app.py")

    console.print(f"🚀 Lançando a aplicação Streamlit a partir de: [cyan]{app_path}[/cyan]")
    console.print("➡️  Acesse a URL fornecida no seu navegador para interagir com o agente.")
    
    # Executa o comando para iniciar o servidor Streamlit
    subprocess.run(["streamlit", "run", app_path])

if __name__ == "__main__":
    main()