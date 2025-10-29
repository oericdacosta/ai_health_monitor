import os
from rich.console import Console
from rich.markdown import Markdown
from langchain_core.messages import HumanMessage
from src.agent_orchestrator import construir_grafo_agente
from src.chart_generator import generate_and_save_charts

console = Console()

def main():
    """
    Executa o fluxo completo: gera gráficos e depois aciona o agente para o relatório.
    """
    console.rule("[bold cyan]🤖 Iniciando Pipeline Completo de Monitoramento SRAG[/bold cyan]")

    generate_and_save_charts()

    console.rule("[bold cyan]🤖 Iniciando Agente de IA para Relatório[/bold cyan]")
    agent_app = construir_grafo_agente()
    console.print("✅ Agente construído com sucesso!")

    missao = (
        "Gere um relatório completo sobre a situação da SRAG no Brasil. "
        "O relatório deve incluir as 4 métricas principais (taxa de aumento de casos, taxa de mortalidade, taxa de ocupação de UTI, taxa de vacinação). "
        "Use notícias recentes para dar contexto às métricas. "
        "O seu relatório deve mencionar que os dois gráficos (casos diários e mensais) já foram gerados e salvos em arquivos de imagem."
    )
    console.print(f"\n[bold]📝 Missão do Agente:[/bold]\n[yellow]{missao}[/yellow]")

    console.rule("[bold cyan]🚀 Executando o fluxo do agente...[/bold cyan]")
    
    inputs = {"messages": [HumanMessage(content=missao)]}
    final_report = None
    
    for s in agent_app.stream(inputs):
        node_name = list(s.keys())[0]
        if "generate_report" in s:
            final_report_message = s["generate_report"]["messages"][-1]
            if hasattr(final_report_message, 'content'):
                final_report = final_report_message.content

    console.rule("[bold cyan]📄 Resultados Finais[/bold cyan]")

    if final_report:
        console.print(Markdown(final_report))
        
        console.print("\n--- [ Verificação dos Gráficos (Status Final) ] ---")
        img_dir = os.path.join(os.getcwd(), "img")
        grafico_diario = os.path.join(img_dir, "grafico_diario.png")
        grafico_mensal = os.path.join(img_dir, "grafico_mensal.png")

        if os.path.exists(grafico_diario):
            console.print(f"✅ [green]Gráfico diário está presente.[/green]")
        else:
            console.print(f"❌ [red]Alerta: Gráfico diário não foi encontrado.[/red]")
        
        if os.path.exists(grafico_mensal):
            console.print(f"✅ [green]Gráfico mensal está presente.[/green]")
        else:
            console.print(f"❌ [red]Alerta: Gráfico mensal não foi encontrado.[/red]")
    else:
        console.print("[bold red]❌ Erro: O agente não conseguiu gerar um relatório final.[/bold red]")

if __name__ == "__main__":
    main()