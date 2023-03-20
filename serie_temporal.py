import pandas as pd
import itertools    
import statsmodels.api as sm

from sklearn.metrics import explained_variance_score, mean_absolute_error
from sklearn.metrics import mean_squared_log_error, median_absolute_error, r2_score

class Avalia_modelo:

    # ---------------------------------------------------------------------------------------------------------- #
    # Função: metricas
    #
    # Descrição: Realiza calculo de diversos tipos de medidas para se avaliar qualidade de uma previsão de séries temporais
    #
    # Parâmetros: 1) df_original: y_teste
    #             2) df_predito: y_pred
    #
    # Quem procurar: Otávio Augusto Iasbeck
    # ---------------------------------------------------------------------------------------------------------- #
    def metricas(df_original, df_predito):
        try:
            print(f"(> melhor) Variância Explicada do modelo {explained_variance_score(df_original, df_predito)}")
        except:
            print("Não foi possível calcular a variança explicada!")
            pass
        
        try:
            print(f"(< melhor) Erro Médio Absoluto do modelo {mean_absolute_error(df_original, df_predito)}")
        except:
            print("Não foi possível calcular a média absoluta de erro")
            pass
            
        try:
            print(f"(> melhor) Erro Logarítmico Quadrado Médio do modelo {mean_squared_log_error(df_original, df_predito)}")
        except:
            print("Não foi possível calcular o erro logarítimo quadrado médio")
            pass
        
        try:
            print(f"(< melhor mede precisão) Erro Absoluto Mediano do modelo {median_absolute_error(df_original, df_predito)}")
        except:
            print("Não foi possível calcular o erro absoluto mediano do modelo")
            pass
        
        try:
            print(f"(> melhor) Coeficiente de Determinação (R²) do modelo {r2_score(df_original, df_predito)}")
        except:
            print("Não foi possível calcular o R^2 do modelo ")
            pass



class Sarima:
    
    # ---------------------------------------------------------------------------------------------------------- #
    # Função: best_parameters
    #
    # Descrição: Acha os melhores parâmetros para o calculo de Sarima, através de chute e análise
    #
    # Parâmetros: 1) df: dataframe que possui os dados históricos do indicador
    #             2) sazonalidade: sazonalidade dos dados (geralmente 7)
    #
    # Quem procurar: Otávio Augusto Iasbeck
    # ---------------------------------------------------------------------------------------------------------- #
    def best_parameters(df, sazonalidade):
        
        '''
            Parâmetros:
                1) df: Dataframe possuindo um atributo com o indicador analisado
                2) sazonalidade: sazonalidade do atributo passado
                
            Retorno:
                1) param_mini = parâmetros p,d,q 
                2) param_sazonal_mini = parâmetros com sazonalidade
                
        '''
        
        ## Seleciono qual variança os 3 parâmetros poderão ter (cuidado! o tempo de processamento cresce exponencialmente)
        p = d = q = range(0, 2)
        
        ## Realizo as combinações possíveis entre eles, e aloco em lista
        pdq = list(itertools.product(p, d, q))
        
        ## Realizo a combinação entre eles e coloco o período sazonal
        seasonal_pdq = [(x[0], x[1], x[2],sazonalidade) for x in list(itertools.product(p, d, q))]

        ## Cria variável com valor mínimo, apenas para ser atribuido ao primeiro loop abaixo
        mini = float('+inf')

        ## Percorr as combinações possíveis de p d q 
        for param in pdq:
            ## Percorro as combinações possíveis de p q d m 
            for param_seasonal in seasonal_pdq:
                try:
                    ## Aplico os parâmetros no modelo SARIMAX
                    mod = sm.tsa.statespace.SARIMAX(df,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=True,
                                                    enforce_invertibility=True)

                    ## Treino o modelo
                    results = mod.fit()
                    
                    ## Utilizo o calculo de .aic para verificar acertividade do modelo com os parâmetros que estão no loop
                    if results.aicc < mini:
                        mini = results.aicc
                        param_mini = param
                        ## Aloco o melhor modelo, que é o menor .aic 
                        param_seasonal_mini = param_seasonal
                        
                    #print(f"Tentativa Parâmetros {param_mini}x{param_seasonal_mini} - Com AIC:{mini} e mae:{results.mae}")

                except:
                    print("Ops!")
                    continue
                    
        print(f"Parâmetros ótimos encontrados: SARIMA{param_mini}x{param_seasonal_mini} - Com AIC:{mini}")
        return param_mini, param_seasonal_mini
    
    
    # ---------------------------------------------------------------------------------------------------------- #
    # Função: split_test_size
    #
    # Descrição: Realiza o corte do dataframe em treino e teste
    #
    # Parâmetros: 1) df: dataframe que possui os dados históricos
    #             2) size: Quantos % o treino terá da base?
    #
    # Quem procurar: Otávio Augusto Iasbeck
    # ---------------------------------------------------------------------------------------------------------- #
    def split_test_size(df, size):
        '''
            Parâmetros:
                1) df: Dataframe possuindo um atributo com o indicador analisado
                2) size = tamanho (em %) do teste
                
            Retorno:
                1) treino = dataframe com dados do tamanho da porcentagem passada  
                2) teste = dataframe com o restante dos dados
                
        '''
        ## Basado no valor passado em %, seleciono o tamanho do dataframe 
        tam = round((len(df) * (size/100)))
        
        ## Seleciono as linhas de treino
        treino = df[:tam]

        ## Seleciono as linhas de teste
        teste = df[tam:]
        
        return treino, teste 
    
    
        
    # ---------------------------------------------------------------------------------------------------------- #
    # Função: model_Sarimax
    #
    # Descrição: Explicita o modelo propriamente dito, que já chama as funções que necessita, por parâmetros escolhidos por mim
    #
    # Parâmetros: 1) df: dataframe que possui os dados históricos
    #             2) qtd_forecast: Quantos dados quer predizer
    #
    # Quem procurar: Otávio Augusto Iasbeck
    # ---------------------------------------------------------------------------------------------------------- #
    def model_Sarimax(df, qtd_forecast):
        
        '''
            Parâmetros:
                1) df: Dataframe possuindo um atributo com o indicador analisado
                2) qtd_forecast = quantidade a ser predita
                
            Retorno:
                1) pred: Dataframe com os dados originais e com coluna de predição

            Observação:
                Passe como "df" apenas um atributo em formato de coluna dataframe
                
        '''
        

        size_train = 80
        sazonalidade = 7
        
        
        df_manipulacao = df.copy()
        
        ## Crio linhas nulas para depois dar concat com os valores preditos
        df_manipulacao = df_manipulacao.append([pd.Series()]*qtd_forecast, ignore_index = True)

        ## Pego os melhores parâmetros para modelo
        param_mini, param_sazonal_mini = Sarima.best_parameters(df, sazonalidade)
        
        ## Separo treino e teste
        treino, teste = Sarima.split_test_size(df, size_train)
        
        ## Exolicito o modelo com os parâmetros recebidos da função anterior
        sarima = sm.tsa.statespace.SARIMAX(treino,
                                           order = param_mini,
                                           seasonal_order = param_sazonal_mini
                                          )
        ## Treino o modelo
        sarima_treino = sarima.fit()
        
        predict = sarima_treino.predict(
                                        start = teste[teste.columns[0]].index[0],
                                        end = (teste[teste.columns[0]].index[::-1][0]) + qtd_forecast,
                                        dynamic=True
                                        )
        ## Concateno os dados preditos com os dados de treino e teste
        df_manipulacao = pd.concat([predict, df_manipulacao], axis = 1)
        
        ## Dos dados contatenados, separo apenas os dados de teste 
        df_avaliacao = (df_manipulacao.loc[~(pd.isnull(df_manipulacao[df_manipulacao.columns[0]]))]).dropna()

        print("\n--------Avaliação do modelo: -----------")

        ## Envio para avaliação 
        Avalia_modelo.metricas(df_avaliacao[df_avaliacao.columns[1]], df_avaliacao[df_avaliacao.columns[0]])

        ## Print dos valores históricos junto à predição
        df_manipulacao.plot()
        
        
        return df_manipulacao