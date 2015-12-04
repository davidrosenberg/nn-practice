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

d = dd[dd.stop_time > "2015-12-03"]

colsToKeep = ['config.activation', 'config.batch_size','config.dropout_rate',
              'config.learning_rate', 'config.n_epochs', 'config.n_hidden',
              'info.num_epochs', 'info.run_time', 'info.validation_perf']

d[colsToKeep]

df = pd.DataFrame(results)
df = pd.DataFrame(list(db.default.runs.find()))
