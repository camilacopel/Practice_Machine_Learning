# -*- coding: utf-8 -*-
"""
Modelos de previsão para alguns meses à frente baseado em dados históricos.

Em desenvolvimento.
"""
from typing import Optional

import pandas as pd


def calc_corr_last12(df_hist: pd.DataFrame) -> pd.DataFrame:
    """
    Cálculo da correlação dos últimos 12 meses informados com anos anteriores.

    Parameters
    ----------
    df_hist : DataFrame
        Dataframe do histórico a ser analisado.
        As 12 últimas linhas devem ser os 12 meses com os quais será calculada
        a correlação com dados de anos anteriores.

    Returns
    -------
    DataFrame
        Dataframe com as correlações calculadas.
        Como index é informado o mês final do período de 12 meses com o qual
        foi calculada a correlação.

    """
    # pylint: disable=protected-access
    # Conferindo se o index está no formato desejado (pd.PeriodIndex freq=M)
    if not isinstance(df_hist.index.freq, pd._libs.tslibs.offsets.MonthEnd):
        raise Exception("Favor informar dataframe periodizado mensalmente.")

    df_ref = df_hist.copy()
    # Trecho de 12 meses com o qual será calculada a correlação
    df_trecho = df_ref.iloc[-12:]
    # Espalha o trecho por uma cópia do dataframe respeitando os meses
    df_other = df_ref.copy()
    for month in range(1, 13):
        df_other[df_other.index.month == month] = df_trecho[df_trecho.index.month == month].iloc[0]

    # Calcula a correlação
    df_corr = df_ref.rolling(12).corr(df_other)
    # Filtra os dados da correlação com o mês final
    df_corr = df_corr[df_corr.index.month == df_trecho.index.month[-1]]
    df_corr.index.name = 'mes_final'

    return df_corr


class SimpleCorrelation12:
    """
    Classe que representa o um modelo de correlações simples.

    Attributes
    ----------
    coluna : int
        Coluna referência para a escolha da <posicao> maior correlação.
    posicao : int
        Posição na ordem de maior correlação relativo à <coluna>.
    df_base : DataFrame
        Base utilizada para o modelo
    df_correlacao_ : pd.DataFrame
        Tabela com a correlação entre os últimos 12 meses e
        estes doze meses em outros anos.
    correlacao_ : float
        Correlação encontrada para <posicao> melhor correlação para <coluna>
    correlacao_outros_ : Series
        Correlação para as outras colunas na <posicao>
    mes_final_periodo_ : str
        'Ano-mes' do perído com a <posicao> melhor correlação para <coluna>
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self,
                 coluna: int,
                 posicao: int = 1,
                 ):
        """
        Criação do modelo.

        Parameters
        ----------
        coluna : int
            Coluna a ser usada para ordenação das maiores correlações.
        posicao : int, optional
            Posição na ordenação de maiores correlações.
            O default é a usar a primeira posição (1).

        """
        self.coluna = coluna
        self.posicao = posicao

        # Atributos que serão definidos apenas ao realizar o fit
        self.df_base = None
        self.df_correlacao = None
        self._period_ = None
        self.correlacao_ = None
        self.correlacao_outros_ = None
        self.mes_final_periodo_ = None


    def fit(self, df_base: pd.DataFrame):
        """
        Cálculo das correlações.

        Parameters
        ----------
        df_base : DataFrame
            Dados históricos para cálculo das correlações.

        """
        self.df_base = df_base.copy()
        self.df_correlacao = calc_corr_last12(df_base)

        top_corr = self.df_correlacao[self.coluna].sort_values(ascending=False)

        # Dados que serão usados para previsão
        self._period_ = top_corr.index[self.posicao]

        # Outras informações sobre o modelo 'treinado'
        # Usando sufixo _ semelhante ao scikit-learn
        self.correlacao_ = top_corr.iloc[self.posicao]
        self.correlacao_outros_ = self.df_correlacao.loc[top_corr.index[self.posicao]]
        self.mes_final_periodo_ = self._period_.strftime('%Y-%m')

        # Retorna o próprio objeto para o caso de ser encadeado com .predict()
        return self


    def predict(self,
                num_meses: Optional[int] = None,
                ) -> pd.DataFrame:
        """
        Previsão para os próximos 'num_meses'.

        Parameters
        ----------
        num_meses : int, optional
            Número de meses a serem previstos.
            Se não informado será feita a previsão até final do último ano informado.

        Returns
        -------
        DataFrame
            Previsão para os próximos meses.

        """
        # Mês seguinte ao período selecionado pelo modelo
        try:
            # Não havendo erro supomos que o fit foi realizado
            mes_escolhido_ini = self._period_ + 1
        except TypeError:
            # Podemos futuramente usar o erro sklearn.exceptions.NotFittedError,
            # mas para simplificar:
            raise Exception("Realizar o fit do modelo antes") from None

        # Primeiro mês a ser previsto
        next_month = self.df_base.index[-1] + 1

        # Se num_meses não for informado vai o final do ano
        num_meses = num_meses if num_meses else (13 - mes_escolhido_ini.month)

        # Dados do período escolhido
        df_escolhido = self.df_base.loc[mes_escolhido_ini:].iloc[:num_meses]

        # Ajusta o index para o novo período
        df_trecho_ajust = df_escolhido.copy()
        df_trecho_ajust.index = pd.period_range(next_month,
                                                periods=len(df_trecho_ajust),
                                                freq='M')

        return df_trecho_ajust
