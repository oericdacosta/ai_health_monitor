import os
import json
import re
import pandas as pd
from typing import List


def load_data(file_paths: List[str], /, *, columns: List[str]) -> pd.DataFrame:
    """
    Carrega apenas as colunas especificadas de múltiplos arquivos Parquet locais.
    """
    print("Iniciando carregamento otimizado de dados Parquet...")
    
    dataframes = []
    for path in file_paths:
        try:
            df = pd.read_parquet(path, columns=columns)
            dataframes.append(df)
        except Exception as error:
            print(f"❌ Erro ao carregar o arquivo {os.path.basename(path)}: {error}")
            
    full_df = pd.concat(dataframes, ignore_index=True).astype(str)
    print(f"✅ {len(full_df):,} registros carregados com sucesso.")
    return full_df


def rename_columns(df: pd.DataFrame, /, *, json_path: str) -> pd.DataFrame:
    """
    Renomeia as colunas do DataFrame com base em um arquivo JSON de dicionário de dados.
    """
    print("Iniciando processo de renomeação de colunas...")
    
    with open(json_path, 'r', encoding='utf-8') as file:
        data_dict = json.load(file)['dicionario_de_dados_sivep_gripe']

    def normalize_name(name: str) -> str:
        name = (
            name.lower()
            .replace('(ibge)', '')
            .replace('(cnes)', '')
            .replace('código', '')
        )
        name = re.sub(r'^\d+-?\s*', '', name)
        name = re.sub(r'[^a-z0-9\s_]', '', name)
        return re.sub(r'\s+', '_', name).strip('_')

    rename_map = {}
    for item in data_dict:
        old_names_str = item.get("nome_coluna_dbf")
        if not old_names_str or old_names_str == 'N/A':
            continue

        new_base_name = normalize_name(item["nome_campo_ficha"])
        old_names = [n.strip() for n in old_names_str.split(' OU ')]

        for old_name in old_names:
            if old_name in df.columns:
                final_name = new_base_name
                if len(old_names) > 1:
                    final_name += "_codigo" if old_name.startswith("CO_") else "_nome"
                rename_map[old_name] = final_name

    renamed_df = df.rename(columns=rename_map)
    print("✅ Colunas renomeadas com sucesso.")
    return renamed_df
