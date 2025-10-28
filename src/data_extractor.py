import asyncio
import aiohttp
import aiofiles
from bs4 import BeautifulSoup
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, BarColumn, DownloadColumn, TextColumn, TimeRemainingColumn, TransferSpeedColumn

BASE_URL = "https://opendatasus.saude.gov.br"
DATASET_URL = f"{BASE_URL}/dataset/srag-2021-a-2024"
DOWNLOAD_DIR = Path("./data")
MAX_CONCURRENT_DOWNLOADS = 7

console = Console()


async def fetch(session, url):
    """Baixa o conteúdo HTML de uma página."""
    try:
        async with session.get(url, timeout=30) as response:
            response.raise_for_status()
            return await response.text()
    except Exception as e:
        console.print(f"[yellow]⚠️ Erro ao buscar {url}: {e}[/yellow]")
        return None


async def get_download_links(session):
    """Coleta todos os links .parquet dentro das páginas de recursos."""
    console.print("🔍 Buscando links de recursos na página principal...")

    html = await fetch(session, DATASET_URL)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")

    # Encontra todos os links de recursos dentro da página principal
    resource_links = [
        BASE_URL + a["href"] if not a["href"].startswith("http") else a["href"]
        for a in soup.select("a.heading[href]")
    ]

    console.print(f"✅ {len(resource_links)} páginas de recursos encontradas.")
    parquet_links = []

    for link in resource_links:
        sub_html = await fetch(session, link)
        if not sub_html:
            continue
        sub_soup = BeautifulSoup(sub_html, "html.parser")

        # Procura por links .parquet
        for a in sub_soup.find_all("a", href=True):
            href = a["href"]
            if href.endswith(".parquet"):
                if not href.startswith("http"):
                    href = BASE_URL + href
                parquet_links.append(href)

    console.print(f"✅ {len(parquet_links)} arquivos .parquet encontrados para download.")
    return parquet_links


async def download_file(session, url, progress, task_id):
    """Baixa um único arquivo .parquet com progresso."""
    filename = url.split("/")[-1]
    filepath = DOWNLOAD_DIR / filename

    if filepath.exists():
        console.print(f"[green]✅ Arquivo já existe, pulando: {filename}[/green]")
        progress.update(task_id, completed=1)
        return

    try:
        async with session.get(url) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0))
            progress.update(task_id, total=total)

            async with aiofiles.open(filepath, "wb") as f:
                async for chunk in resp.content.iter_chunked(1024 * 64):
                    await f.write(chunk)
                    progress.update(task_id, advance=len(chunk))

        console.print(f"[green]✅ Download concluído:[/green] {filename}")
    except Exception as e:
        console.print(f"[red]❌ Erro ao baixar {filename}:[/red] {e}")


async def main():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    console.rule("[bold cyan]Extração de Dados SRAG — OpenDataSUS[/bold cyan]")

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_DOWNLOADS)
    async with aiohttp.ClientSession(connector=connector) as session:
        parquet_links = await get_download_links(session)
        if not parquet_links:
            console.print("[red]Nenhum link .parquet encontrado.[/red]")
            return

        with Progress(
            TextColumn("[bold blue]{task.fields[filename]}[/bold blue]"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            tasks = []
            for url in parquet_links:
                filename = url.split("/")[-1]
                task_id = progress.add_task("download", filename=filename, start=False)
                tasks.append(download_file(session, url, progress, task_id))
            await asyncio.gather(*tasks)

    console.rule("[bold green]✅ Extração concluída com sucesso![/bold green]")


if __name__ == "__main__":
    asyncio.run(main())
