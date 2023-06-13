# Trabalho de aplicação de série temporal

Este repositório é um trabalho voltado à automatização do modelo Sazonal Arima, abrangendo os seguintes tópicos:

- Métricas de Avaliação
- Escolha dos melhores parâmetros (p, d, q)
- Divisão de treino e teste
- Aplicação do modelo com melhores variáveis

## Métricas de Availiação

Foi utilizada algumas métricas para avaliação final do modelo, afim de futuramente criar uma lógica à qual diz o quão confiável a predição é, afim de que o usuário final decida se deve levar os dados previstos em prática ou não

## Escolha dos melhores parâmetros

Foi realizada uma lógica simples à qual testa diversas combinações de parâmetros do calculo de Sarima (Método de escolha de hiper-parâmetros Random Search), utilizando como métrica de "melhor combinação" o teste de .aicc

## Divisão de Treino e Teste

Recebe o dataframe com coluna única e separa por index, recebendo como parâmetro a porcentagem que será utilizada para teste

## Aplicação do modelo

Chama as funções anteriores, explicita o modelo de arima e retorna o dataframe com linhas adicionais como as linhas da previsão, além do calculo das métricas de avaliação do modelo

## Como usar

Para utilizar, basta realizar o dowload das bilbiotecas necessárias, chamar a função "modelo_Sarimax" e passar como parâmetro um dataframe com coluna única e a quantidade que você quer realizar de previsão (Importante identificar sazonalidade que o dataset possui, para assim, alterar o objeto explicitado dentro da função, com sazonalidade padrão semanal '7')

### Possíveis melhorias

O código disposto neste repositório abre portas para diversas oportunidades de melhoria, como por exemplo a adição de uma lógica mais forte de escolha de melhores parâmetros, podendo ser aplicado um algorítmo genético afim de otimizar o tempo e a solução, ou até mesmo a variação de parâmetros que o modelo recebe.
