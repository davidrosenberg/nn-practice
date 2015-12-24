import numpy as np
import datetime
import pymongo
from bson import json_util
import pprint
import json
import pandas as pd

client = pymongo.MongoClient()
db = client['MY_DB']
runs = db.default.runs


#filt = {'config.datasetname': 'mnist.small.pkl.gz', 'status': 'COMPLETED'}
#cursor = runs.find(filt)
#cursor = runs.find()
#df =  pd.DataFrame(list(cursor))

def get_results(filt, project=None):
    # https://docs.mongodb.org/getting-started/python/query/
    project_dict = {'result': True, '_id': False}
    if project is None:
        project_dict['config'] = True
    elif isinstance(project, dict):
        for k, v in project.items():
            project_dict[k] = v
    else:
        for k in project:
            project_dict[k] = True
    return json_normalize(runs.find(filt, project_dict).
                          sort('result', direction=pymongo.DESCENDING))

filt = {'config.datasetname': 'mnist.small.pkl.gz', 'status': 'COMPLETED'}
dd = pd.io.json.json_normalize(runs.find(filt))

dd = dd[dd.stop_time > "2015-12-03"]

colsToKeep = ['config.activation',
              'config.batch_size','config.dropout_rate',
              'config.learning_rate', 'config.n_epochs',
              'info.num_steps','config.n_hidden', 'info.num_epochs',
              'info.run_time', 'info.validation_perf',
              'info.test_perf','info.learning_curve']

d = dd[colsToKeep]
d[pd.notnull(d['info.learning_curve'])]

tp = d[['

ldf = pd.DataFrame(results)
df = pd.DataFrame(list(db.default.runs.find()))

from ggplot import *
meat_lng = pd.melt(meat[['date', 'beef', 'pork', 'broilers']], id_vars='date')
ggplot(aes(x='date', y='value', colour='variable'), data=meat_lng) + \
    geom_point() + \
    stat_smooth(color='red')

ggplot(aes(x='date', y='beef'), data=meat) + \
    geom_point(color='lightblue') + \
    stat_smooth(span=.15, color='black', se=True) + \
    ggtitle("Beef: It's What's for Dinner") + \
    xlab("Date") + \
    ylab("Head of Cattle Slaughtered")

