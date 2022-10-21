# -*- coding: utf-8 -*-
"""Apenas para testes de funcionamento.

Pode ser modificado à vontade.

# TODO: A ideia é a criação de um novo modelo, semelhante ao SimpleCorrelation12.
As complexidades para escolha do melhor período ficarão no método fit.
"""

#%% Ajuste Local
# Manobra inicial para funcionar localmente para testes
import sys
from pathlib import Path

folderpath = Path(__file__).parent

if (folder := str(folderpath)) not in sys.path:
    sys.path.append(folder)


#%% Script de testes
# Aqui começa o script para testes
from vazoes_txt import VazoesTxt
from modelos import SimpleCorrelation12

# Postos principais
POSTOS_PRINCIPAIS = {
    6: 'FURNAS',
    74: 'GBM',
    169: 'SOBRADINHO',
    275: 'TUCURUÍ',
    }

arquivo = folderpath / 'teste_arquivos_entrada' / 'VAZOES-P50.txt'
arquivo_new = folderpath / 'teste_arquivos_saida' / 'VAZOES_MOD.txt'

# Criação do objeto de vazões
vazoes = VazoesTxt(arquivo)

# Criação do modelo.
# O modelo criado está bem simplificado, e estamos tentando seguir o exemplo
# do scikit-learn, com métodos fit e predict.
model = SimpleCorrelation12(coluna=6, posicao=1)
# Parametrização do modelo
model.fit(vazoes.df_period)
# Predição para um novo período
novo_trecho = model.predict()

# Junção do período do arquivo com o período previsto
vazoes_new = vazoes.append(novo_trecho)
# Salva novo arquivo de vazões no formato txt
vazoes_new.salvar_txt(arquivo_new)

# Apenas para visualização das correlações do principais postos
correlacoes_principais = model.df_correlacao[POSTOS_PRINCIPAIS.keys()]
correlacoes_principais.columns = POSTOS_PRINCIPAIS.values()
