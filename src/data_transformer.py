import pandas as pd


def prepare_dataframe(df: pd.DataFrame, /) -> pd.DataFrame:
    """
    Executa o pipeline completo de limpeza, transformação e formatação de dados SRAG.
    """
    print("\nIniciando pipeline completo de limpeza e transformação com Pandas...")

    df['idade'] = (
        pd.to_datetime(df['data_de_1s_sintomas'], errors='coerce', format='mixed') -
        pd.to_datetime(df['data_de_nascimento'], errors='coerce', format='mixed')
    ).dt.days / 365.25

    df['evolucao'] = df['evoluo_do_caso'].map({'1': 'Cura', '1.0': 'Cura', '2': 'Óbito', '2.0': 'Óbito'})
    df['uti'] = df['internado_em_uti'].map({'1': 'Sim', '1.0': 'Sim', '2': 'Não', '2.0': 'Não'})
    df['vacina_covid'] = df['recebeu_vacina_covid19'].map({'1': 'Sim', '2': 'Não'})
    df['sexo'] = df['sexo'].map({'M': 'Masculino', 'F': 'Feminino', 'I': 'Ignorado'})

    selected_columns = {
        'data_de_1s_sintomas': 'data_sintomas',
        'uf_residncia': 'uf',
        'sexo': 'sexo',
        'idade': 'idade',
        'uti': 'uti',
        'data_da_entrada_na_uti': 'data_entrada_uti',
        'data_da_sada_da_uti': 'data_saida_uti',
        'evolucao': 'evolucao',
        'vacina_covid': 'vacina_covid',
        'data_1_dose_da_vacina_covid19': 'data_dose1_covid'
    }

    df = df.rename(columns=selected_columns)[list(selected_columns.values())]

    critical_columns = ["data_sintomas", "uf", "idade", "evolucao", "uti"]
    df = df.dropna(subset=critical_columns).copy()

    df.loc[:, 'idade'] = df['idade'].astype(int)

    date_columns = ["data_sintomas", "data_entrada_uti", "data_saida_uti", "data_dose1_covid"]
    for column in date_columns:
        df.loc[:, column] = (
            pd.to_datetime(df[column], errors='coerce', format='mixed')
            .dt.strftime('%Y-%m-%d')
        )

    print("✅ Pipeline de limpeza e transformação concluído.")
    return df
