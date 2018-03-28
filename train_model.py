import lib
import numpy as np
import catboost

target_indicator = 'SP.DYN.CDRT.IN'
source_indicators = [
        'SE.XPD.TOTL.GD.ZS',
        'SH.XPD.TOTL.ZS',
        'GB.XPD.RSDV.GD.ZS',
        'MS.MIL.XPND.GD.ZS',
]
all_indicators = source_indicators + [target_indicator]

#expenditure_education = lib.get_data('SE.XPD.TOTL.GD.ZS')
#expenditure_health = lib.get_data('SH.XPD.TOTL.ZS')
#expenditure_rnd = lib.get_data('GB.XPD.RSDV.GD.ZS')
#expenditure_military = lib.get_data('MS.MIL.XPND.GD.ZS')
#death_rate = lib.get_data('SP.DYN.CDRT.IN')

countries = {}
for ind in all_indicators:
    data = lib.get_data(ind)
    for d in data:
        value = d['value']
        if value is None:
            continue

        country_name = d['country']['value']
        country = countries.get(country_name, {})
        countries[country_name] = country

        bin_name = d['date']
        bin = country.get(bin_name, {})
        country[bin_name] = bin

        bin[ind] = float(value)

X = []
Y = []

for country_name, country in countries.items():
    for bin_name, bin in country.items():
        if len(bin) != len(all_indicators):
            continue

        X += [[bin[ind] for ind in source_indicators]]
        Y += [bin[target_indicator]]
        print('{} - {} len = {}'.format(country_name, bin_name, len(bin)))

train_pool = catboost.Pool(X, Y)

model = catboost.CatBoostRegressor()
model.fit(train_pool)

training_error = sum((model.predict(train_pool) - Y)**2)/len(Y)

print('model training error = {}'.format(training_error))

import ipdb; ipdb.set_trace()
