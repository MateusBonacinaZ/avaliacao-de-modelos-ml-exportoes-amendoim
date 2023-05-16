import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression
from calendar import monthrange


class DataProcessing:
    def __init__(self):
        self.beginning_year = 2012
        self.end_year = 2022
        self.target = 'brazil_export'

    def main(self):
        files = self.collect_files()
        data = self.collect_data(files)
        data = self.replace_empty_values(data)
        data = self.gather_data(data)
        data = self.select_features(data)
        # data.to_excel('apresentacao-deise.xlsx')
        return data

    def collect_files(self):
        files = {
            'production': [],
            'area': [],
            'quotation': [],
            'export': []
        }

        for file in os.listdir(os.path.dirname(os.path.realpath(__file__)) + r'\Bases de dados'):
            if 'production' in file:
                files['production'].append(os.path.dirname(os.path.realpath(__file__)) + fr'\Bases de dados\{file}')
            elif 'area' in file:
                files['area'].append(os.path.dirname(os.path.realpath(__file__)) + fr'\Bases de dados\{file}')
            elif 'quotation' in file:
                files['quotation'].append(os.path.dirname(os.path.realpath(__file__)) + fr'\Bases de dados\{file}')
            elif 'export' in file:
                files['export'].append(os.path.dirname(os.path.realpath(__file__)) + fr'\Bases de dados\{file}')

        return files

    def collect_data(self, files):
        data = {
            'production': pd.DataFrame(
                    {
                        'country': [],
                        'year': [],
                        'culture': [],
                        'value': []
                    }
                ),
            'area': pd.DataFrame(
                {
                    'country': [],
                    'year': [],
                    'culture': [],
                    'value': []
                }
            ),
            'quotation': pd.DataFrame(
                {
                    'coin': [],
                    'year': [],
                    'month': [],
                    'value': []
                }
            ),
            'export': pd.DataFrame(
                {
                    'country': [],
                    'year': [],
                    'month': [],
                    'value': []
                }
            )

        }

        for type_file in files.keys():

            if type_file == 'export':
                dataframe_all = pd.DataFrame()
                for file in files[type_file]:
                    dataframe_file = pd.read_csv(file, delimiter=',')
                    dataframe_all = pd.concat([dataframe_all, dataframe_file], ignore_index=True)

                data_file = {
                    'country': [],
                    'year': [],
                    'month': [],
                    'value': []
                }

                for country in dataframe_all['ReporterDesc'].unique():
                    for year in range(self.beginning_year, self.end_year+1):
                        for month in range(1, 12+1):
                            data_file['country'].append(str(country).strip().lower())
                            data_file['year'].append(int(year))
                            data_file['month'].append(int(month))
                            try:
                                data_file['value'].append(float(
                                    dataframe_all.loc[(dataframe_all['ReporterDesc'] == country) &
                                                      (dataframe_all['RefYear'] == year) &
                                                      (dataframe_all['RefMonth'] == month), 'NetWgt'].values[0]
                                ))

                            except:
                                data_file['value'].append(np.nan)

                data[type_file] = pd.concat([data[type_file], pd.DataFrame(data_file)], ignore_index=True)

            elif type_file == 'production' or type_file == 'area':
                dataframe_all = pd.DataFrame()
                for file in files[type_file]:
                    dataframe_file = pd.read_csv(file, delimiter=',')
                    dataframe_all = pd.concat([dataframe_all, dataframe_file], ignore_index=True)

                data_file = {
                    'country': [],
                    'year': [],
                    'culture': [],
                    'value': []
                }

                for country in dataframe_all['Area'].unique():
                    for culture in dataframe_all.loc[dataframe_all['Area'] == country, 'Item'].unique():
                        for year in range(self.beginning_year, self.end_year+1):
                            data_file['country'].append(str(country).strip().lower())
                            data_file['year'].append(int(year))
                            data_file['culture'].append(str(culture).split(',')[0].strip().lower())
                            try:
                                if type_file == 'production':
                                    data_file['value'].append(
                                        dataframe_all.loc[(dataframe_all['Area'] == country) &
                                                        (dataframe_all['Item'] == culture) &
                                                        (dataframe_all['Year'] == year), 'Value'].values[0]*1000)
                                elif type_file == 'area':
                                    data_file['value'].append(
                                        dataframe_all.loc[(dataframe_all['Area'] == country) &
                                                        (dataframe_all['Item'] == culture) &
                                                        (dataframe_all['Year'] == year), 'Value'].values[0])
                            except:
                                data_file['value'].append(np.nan)

                data[type_file] = pd.concat([data[type_file], pd.DataFrame(data_file)], ignore_index=True)

            elif type_file == 'quotation':
                dataframe_all = pd.DataFrame()
                for file in files[type_file]:
                    dataframe_file = pd.read_csv(file, delimiter=',')
                    dataframe_file['coin'] = str(file.split(r'\ '.strip())[-1])\
                        .replace('.csv', '').replace('quotation_', '')
                    dataframe_all = pd.concat([dataframe_all, dataframe_file], ignore_index=True)

                dataframe_all['Data'] = pd.to_datetime(dataframe_all['Data'], format='%d.%m.%Y')
                dataframe_all['year'] = dataframe_all['Data'].dt.year
                dataframe_all['month'] = dataframe_all['Data'].dt.month

                data_file = {
                    'coin': [],
                    'year': [],
                    'month': [],
                    'value': []
                }

                for coin in dataframe_all['coin'].unique():
                    for year in range(self.beginning_year, self.end_year+1):
                        for month in range(1, 12+1):
                            data_file['coin'].append(str(coin).strip().lower())
                            data_file['year'].append(int(year))
                            data_file['month'].append(int(month))
                            try:
                                data_file['value'].append(dataframe_all.loc[(dataframe_all['coin'] == coin) &
                                                                            (dataframe_all['year'] == year) &
                                                                            (dataframe_all['month'] == month), 'Último']
                                                          .values[0].replace(',', '.'))
                            except:
                                data_file['value'].append(np.nan)

                data[type_file] = pd.concat([data[type_file], pd.DataFrame(data_file)], ignore_index=True)

        return data

    def replace_empty_values(self, data):

        for type in data.keys():
            if type == 'production' or type == 'area':
                for country_name in data[type].loc[data[type]['value'].isnull(), 'country'].unique():
                    for culture_name in data[type].loc[data[type]['country'] == country_name, 'culture'].unique():

                        mean = data[type]['value'].loc[(data[type]['country'] == country_name) &
                                                       (data[type]['culture'] == culture_name)].mean()

                        data[type].loc[(data[type]['country'] == country_name) &
                                       (data[type]['culture'] == culture_name) &
                                       (np.isnan(data[type]['value'])), 'value'] = mean

            elif type == 'quotation':

                for coin_cod in data[type].loc[data[type]['value'].isnull(), 'coin'].unique():

                    for year_cod in data[type].loc[(data[type]['coin'] == coin_cod) &
                                                   (data[type]['value'].isnull()), 'year'].unique():

                        mean = data[type]['value'].loc[(data[type]['coin'] == coin_cod) &
                                                       (data[type]['year'] == year_cod)].mean()

                        data[type].loc[(data[type]['coin'] == coin_cod) &
                                       (data[type]['year'] == year_cod) &
                                       (np.isnan(data[type]['value'])), 'value'] = mean

            elif type == 'export':
                for country in data[type].loc[data[type]['value'].isnull(), 'country'].unique():
                    for year in data[type].loc[(data[type]['country'] == country) &
                                               (data[type]['value'].isnull()), 'year'].unique():

                        mean = data[type].loc[(data[type]['country'] == country) &
                                              (data[type]['year'] == year), 'value'].mean()

                        data[type].loc[(data[type]['country'] == country) &
                                       (data[type]['year'] == year) &
                                       (np.isnan(data[type]['value'])), 'value'] = mean

        return data

    def gather_data(self, data):
        data_final = {
            'date': []
        }

        for type in data:
            if type == 'production' or type == 'area':
                for country in data[type]['country'].unique():
                    for culture in data[type].loc[data[type]['country'] == country, 'culture'].unique():
                        name_key = f'{country}_{type}_{culture}'.replace(' ', '_')
                        data_final.update({name_key: []})

            elif type == 'quotation':
                for coin in data[type]['coin'].unique():
                    name_key = f'{type}_{coin}'.replace(' ', '_')
                    data_final.update({name_key: []})

            elif type == 'export':
                for country in data[type]['country'].unique():
                    name_key = f'{country}_{type}'.replace(' ', '_')
                    data_final.update({name_key: []})

        data_final.update({
            'brazil_export_groundnut_last_year': [],
            'brazil_export_groundnut_last_month': []}
        )

        for year in range(self.beginning_year, self.end_year+1):
            for month in range(1, 12+1):

                data_final['date'].append(f'{int(year)}-{int(month)}-{monthrange(int(year), int(month))[1]}')

                for type in data:

                    if type == 'production' or type == 'area':
                        for country in data[type]['country'].unique():
                            for culture in data[type].loc[data[type]['country'] == country, 'culture'].unique():
                                name_key = f'{country}_{type}_{culture}'.replace(' ', '_')
                                data_final[name_key].append(
                                    float(data[type].loc[(data[type]['country'] == country) &
                                                         (data[type]['year'] == year) &
                                                         (data[type]['culture'] == culture), 'value'].values[0])
                                )

                    elif type == 'quotation':
                        for coin in data[type]['coin'].unique():
                            name_key = f'{type}_{coin}'.replace(' ', '_')
                            data_final[name_key].append(
                                float(data[type].loc[(data[type]['coin'] == coin) &
                                                     (data[type]['year'] == year) &
                                                     (data[type]['month'] == month), 'value'].values[0])
                            )

                    elif type == 'export':
                        for country in data[type]['country'].unique():
                            name_key = f'{country}_{type}'.replace(' ', '_')
                            data_final[name_key].append(
                                float(data[type].loc[(data[type]['country'] == country) &
                                                     (data[type]['year'] == year) &
                                                     (data[type]['month'] == month), 'value'].values[0])
                            )

                if month == 1 and year == self.beginning_year:
                    data_final['brazil_export_groundnut_last_month'].append(0)

                elif month == 1 and year != self.beginning_year:
                    data_final['brazil_export_groundnut_last_month'].append(
                        float(data['export'].loc[(data['export']['country'] == 'brazil') &
                                                 (data['export']['year'] == year-1) &
                                                 (data['export']['month'] == 12), 'value'].values[0])
                    )

                else:
                    data_final['brazil_export_groundnut_last_month'].append(
                        float(data['export'].loc[(data['export']['country'] == 'brazil') &
                                                 (data['export']['year'] == year) &
                                                 (data['export']['month'] == month-1), 'value'].values[0])
                    )

                if year == self.beginning_year:
                    data_final['brazil_export_groundnut_last_year'].append(0)

                else:
                    data_final['brazil_export_groundnut_last_year'].append(
                        float(data['export'].loc[(data['export']['country'] == 'brazil') &
                                                 (data['export']['year'] == year-1) &
                                                 (data['export']['month'] == month), 'value'].values[0])
                    )

        dataframe_final = pd.DataFrame(data_final)
        target = dataframe_final.pop(self.target)
        dataframe_final[self.target] = target
        dataframe_final['date'] = pd.to_datetime(dataframe_final['date'], format='%Y-%m-%d')
        # dataframe_final = dataframe_final.set_index('date')

        return dataframe_final

    def select_features(self, data):
        features = data.loc[:, (data.columns != 'date') &
                               (data.columns != self.target)]

        target = data.loc[:, self.target]
        weight = f_regression(features, target)[0]
        features_selected = []

        for index, column in enumerate(features.columns):
            if weight[index] > weight.mean():
                features_selected.append(column)

        columns_selected = ['date']
        for feature in features_selected:
            columns_selected.append(feature)
        columns_selected.append(self.target)

        dataframe_final = data.loc[:, columns_selected]
        dataframe_final = dataframe_final.set_index('date')

        return dataframe_final


DataProcessing().main()
