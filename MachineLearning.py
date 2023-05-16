import pandas as pd
from DataProcessing import DataProcessing
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import pyhomogeneity as hg

class MachineLearning:
    def __init__(self):
        self.models = [
            LinearRegression(fit_intercept=False, positive=True),
            RandomForestRegressor(n_estimators=1000, criterion='absolute_error', random_state=0),
            MLPRegressor(hidden_layer_sizes=5, activation='identity', max_iter=100000, random_state=0)
        ]

        # DEFININDO A FONTE E TAMANHO PARA A PLOTAGEM DOS GRÁFICOS
        self.graphics_fontfamily = "Arial"
        self.graphics_fontsize = 10
        plt.rcParams['font.family'] = self.graphics_fontfamily
        plt.rcParams['font.size'] = self.graphics_fontsize

    def main(self):
        data = self.get_data()
        scores = self.run_models(data)
        self.show_metrics(scores)
        self.show_graphs(scores, data)

    def get_data(self):
        return DataProcessing().main()

    def run_models(self, data):
        tscv = TimeSeriesSplit(
            n_splits=data.index.year.max() - data.index.year.min()
        )

        scores = []
        for model in self.models:
            fold = 1
            print(type(model).__name__)
            r2_values = []
            mse_values = []
            rmse_values = []
            pred_values_last_fold = []
            
            for fold_train, fold_test in tscv.split(data):
                x_train = data.iloc[fold_train, :-1]
                x_test = data.iloc[fold_test, :-1]
                y_train = data.iloc[fold_train, -1]
                y_test = data.iloc[fold_test, -1]

                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)

                r2_value, mse_value, rmse_value = self.evaluate_model(y_test, y_pred)

                print(f'FOLD {fold}: R² Score= {r2_value}')

                r2_values.append(r2_value)
                mse_values.append(mse_value)
                rmse_values.append(rmse_value)

                if fold == data.index.year.max() - data.index.year.min():
                    for value in y_pred:
                        pred_values_last_fold.append(value)

                fold += 1
            
            self.model_scores(model, r2_values, mse_values, rmse_values, scores, pred_values_last_fold)
            print('\n')

        return scores

    def evaluate_model(self, y_test, y_pred):
        return r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred, squared=True), mean_squared_error(y_test, y_pred, squared=False)

    def model_scores(self, model, r2_values, mse_values, rmse_values, scores, pred_values_last_fold):
        scores.append({
            'name': type(model).__name__,
            'r2_average': (sum(r2_values)/len(r2_values)),
            'mse_average': (sum(mse_values)/len(mse_values)),
            'rmse_average': (sum(rmse_values)/len(rmse_values)),
            'folds': r2_values,
            'pred_values_last_fold': pred_values_last_fold
        })

        scores = scores.sort(reverse=True, key=lambda element: element['r2_average'])

    def show_metrics(self, scores):
        for model in scores:
            print(f"O modelo {model['name']} obteve as seguintes pontuações: \n"
                f"-> Média de R²: {model['r2_average']} \n"
                f"-> Média de MSE: {model['mse_average']} \n"
                f"-> Média de RMSE: {model['rmse_average']} \n\n")

    def show_graphs(self, scores, data):
        # self.graphics_export(data)
        # self.graphics_seasonality_and_trend(data)
        # self.graphics_performance_models(scores)
        self.graphics_predicted_and_actual(scores, data)
        # self.graphics_pettitt_test(data)

    def graphics_export(self, data):
        # GRÁFICO DE EXPORTAÇÃO EM DECORRÊNCIA DOS ANOS
        y = data.loc[:, 'brazil_export']

        plt.figure(figsize=(16, 8))
        plt.plot(y, color='red', marker='o')
        plt.xlabel('Ano')
        plt.ylabel('Exportação')
        plt.ticklabel_format(style='plain', axis='y')
        # plt.title("Série histórica de produção brasileira de amendoim (KG)")
        plt.show()
            
    def graphics_seasonality_and_trend(self, data):
        # GRÁFICO DE TENDÊNCIA
        result = seasonal_decompose(data.loc[:, 'brazil_export'], extrapolate_trend='freq')

        plt.figure(figsize=(16, 8))

        plt.plot(result.trend, color='red')
        # plt.title('Gráfico de tendência')
        plt.xlabel('Anos')
        plt.ylabel('Exportação')
        plt.ticklabel_format(style='plain', axis='y')
        plt.show()

    def graphics_performance_models(self, scores):
        # EXIBIR DESEMPENHO DO MODELO EM RELAÇÃO AOS FOLDS
        x_folds_performance = [i+1 for i in range(len(scores[0]['folds']))]
        y_folds_performance = {}

        for model in scores:
            y_folds_performance.update({
                model['name']: model['folds']
            })
        
        colors = ['red', 'green', 'blue']

        plt.figure(figsize=(16, 8))
        for number, model in enumerate(y_folds_performance):
            plt.plot(x_folds_performance, y_folds_performance[model], marker="o", label=model, color=colors[number])

        plt.xlabel('Fold')
        plt.xticks(range(min(x_folds_performance), max(x_folds_performance)+1))
        plt.ylabel('R² Score')
        # plt.title(f'Desempenho dos modelos R² em relação aos FOLDS')
        plt.legend(loc=4)
        plt.show()

    def graphics_predicted_and_actual(self, scores, data):
        # MOSTRANDO O VALOR PREDITO VS O REAL PARA O ANO DE 2022
        colors = ['black', 'red', 'green', 'blue']
        x = [month for month in range(1, 12+1)]
        y_real = data.loc[data.index.year == data.index.year.max(), 'brazil_export'].values

        y_pred = {}
        for model in scores:
            y_pred.update({
                model['name']: model['pred_values_last_fold']
            })
                
        plt.figure(figsize=(16, 8))
        plt.plot(x, y_real, label='Real', linewidth=2, color=colors[0], marker='o')
        for number, predict_model in enumerate(y_pred):
            plt.plot(x, y_pred[predict_model], label=predict_model, linewidth=1, color=colors[number+1], marker='o')
        
        plt.xlabel('Mês')
        plt.xticks(range(min(x), max(x)+1))
        plt.ylabel('Exportação')
        plt.ticklabel_format(style='plain', axis='y')
        # plt.title(f'Valor predito x Valor real para o ano de 2022')
        plt.legend(loc=4)
        plt.show()

    def graphics_pettitt_test(self, data):
        target = data['brazil_export']

        result = hg.pettitt_test(target)

        mn = data.index.min()
        mx = data.index.max()
        loc = pd.to_datetime(result.cp, format='%Y/%m/%d')
        mu1 = result.avg.mu1
        mu2 = result.avg.mu2

        plt.figure(figsize=(16, 6))
        plt.plot(target, label='Exportação')

        plt.hlines(mu1, xmin=mn, xmax=loc, linestyles='--', colors='green', lw=1.5, label='Média anterior: ' + str(round(mu1, 2)))
        plt.hlines(mu2, xmin=loc, xmax=mx, linestyles='--', colors='red', lw=1.5, label='Média posterior: ' + str(round(mu2, 2)))
        plt.axvline(x=loc, linestyle='-', color='black', lw=1.5, label='Breakpoint: '+ str(loc.strftime('%m/%Y')) + '\n p-value: ' + str(result.p))
        # plt.title('Pettitt Test')
        plt.ticklabel_format(style='plain', axis='y')
        plt.xlabel('Ano')
        plt.ylabel('Exportação')
        plt.legend(loc=4)
        plt.show()


x = MachineLearning()
x.main()
