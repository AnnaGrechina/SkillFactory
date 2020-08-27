import pandas as pd
import pickle
import re
import numpy as np

#from catboost import CatBoostRegressor

########   ФУНКЦИИ ###############3

# парсинг конфигурации автомобиля
def configuration_parsing(txt):
    configuration = []
    pattern = r'\"values\":\[(.+?)\]}'
    for txt_elem in re.findall(pattern, txt):
        pattern_txt_elem = r'\"(.+?)\"'
        for config_item in re.findall(pattern_txt_elem, txt_elem):
            configuration.append(config_item)
    return configuration

def preproc_data(df_input):
    '''includes several functions to pre-process the predictor data.'''
    
    df_output = df_input.copy()
    
    # ################### Предобработка ############################################################## 
    # убираем малозначащие для модели признаки,а также 'vehicleConfiguration' - информация дублируется в других признаках
    df_output.drop(['id', 'Таможня', 'Состояние', 'vehicleConfiguration'], axis=1, inplace=True,)
    
    
    # в явном виде переводим в числовые данные объем двигателя и мощность
    df_output['enginePower'] = df_output['enginePower'].apply(lambda x: int(x[:-4]))
    df_output['engineDisplacement'] = df_output['engineDisplacement'].apply(lambda x: 
                                                                            0 if x == 'undefined LTR' else 10 * float(x[:-4]))

    # ################### fix ############################################################## 
    # Переводим признаки из float в int (иначе catboost выдает ошибку)
    for feature in ['modelDate', 'numberOfDoors', 'mileage', 'productionDate', 'enginePower', 'engineDisplacement']:
        df_output[feature]=df_output[feature].astype('int32')
    

    
    # ################### Feature Engineering ####################################################
    # Добавим признак = количеству опций
    df_output['lenConfiguration'] = df_input['Комплектация'].apply(lambda x: len(configuration_parsing(x)))
    
    # ################### Clean #################################################### 
    # убираем признаки оставшиеся необработанные признаки и исходную комлектацию 
    df_output.drop(['description', 'Владение', 'Комплектация'], axis=1, inplace=True,)
    
    
    return df_output

###################################################



#############   УСТАНАВЛИВАЕМ КОНСТАНТЫ #############33

filename = 'data/set_configuration.pkl'
with open(filename, 'rb') as f:
    set_configuration = pickle.load(f)


#########  ВХОДНЫЕ ДАННЫЕ #############
# вход модели - одна строчка из файла test.csv
# выберем строчку 13, и будем по ней делать предсказание.
# должны получить предсказанную цену: 1440000.0

test_raw = pd.read_csv('data/test.csv')
# создадит тестовые исходные данные по одному автомобилю, назовем 
test_13 = test_raw.iloc[13:14].copy()
print(test_13.columns)

##############  ПРЕДОБРАБОТКА ДАННЫХ #######################
# 1.удалим пропуски в "Владельцы" и "ПТС"
test_13['Владельцы'].fillna('3 или более', inplace = True)
test_13['ПТС'].fillna('Оригинал', inplace = True)

#Предобработка: нужно привести данные к обработанному виду, который будем подавать на вход 
# 2. Убедимся, что колонки называются правильно
test_13.columns = ['bodyType', 'brand', 'color', 'fuelType', 'modelDate', 'name',
       'numberOfDoors', 'productionDate', 'vehicleConfiguration',
       'vehicleTransmission', 'engineDisplacement', 'enginePower',
       'description', 'mileage', 'Комплектация', 'Привод', 'Руль', 'Состояние',
       'Владельцы', 'ПТС', 'Таможня', 'Владение', 'id']
# 3. Добавим колонки, соответствующие конфигурации автомобиля
test_13 = test_13.reindex(columns = test_13.columns.tolist() + list(set_configuration))

# 4. Отмечаем единицами, что вхоит в конфигурацию
for col in set_configuration:
    test_13[col] = test_13['Комплектация'].apply(lambda x: 1 if col in configuration_parsing(x) else 0)

test_13 = preproc_data(test_13)
#
# вектор признаков подготовлен и обработан.

# Получаем предсказания для части 1 - CatBoost
n_foldes = 5

X_meta_test_features13 = []

X_meta_test13 = np.zeros(len(test_13), dtype = np.float32)
for i in range(n_foldes):
    filename = 'models/folded_model_' + str(i) + '_CatB.pkl'
    with open (filename, 'rb') as f:
        folded_model = pickle.load(f)
    X_meta_test13 += folded_model.predict(test_13)
    

X_meta_test13 = X_meta_test13 / 5

X_meta_test_features13.append(X_meta_test13) #добавили полученные фичи в общие метапризнаки

print('From CatBoost: ', X_meta_test13)

# для части 2 - RandomForest

filename = 'models/folded_model_RandomForest.pkl'
with open (filename, 'rb') as f:
    meta_model = pickle.load(f)

X_meta_test13 = meta_model.predict(test_13.drop(cat_features_ids, axis = 1))
print('From RandomForest: ', X_meta_test13)
X_meta_test_features13.append(X_meta_test13)
X_meta_test_features13

# Финальная модель - объединяем предсказания
stacked_features_test13 = np.vstack(X_meta_test_features13).T

filename = 'models/final_model.pkl'
with open (filename, 'rb') as f:
    final_model = pickle.load(f)
    
predicted_price_13 = np.floor(final_model.predict(stacked_features_test13) / 10000) * 10000 
print('Prediction price: ', predicted_price_13)