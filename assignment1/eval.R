library(rmongodb)
library(data.table)
library(plyr)

mongo <- mongo.create()
mongo.is.connected(mongo)

mongo.get.databases(mongo)
db <- "MY_DB"
mongo.get.database.collections(mongo, db)
coll <- "MY_DB.default.runs"

filt = '{"config.datasetname": "mnist.small.pkl.gz", "status": "COMPLETED"}'
mongo.find.one(mongo, coll, filt)

dd =  mongo.find.all(mongo, coll, filt)

configToKeep = c('n_hidden','activation',
              'batch_size','dropout_rate',
              'learning_rate', 'n_epochs',
              

infoToKeep = c('num_steps', 'num_epochs',
              'run_time', 'validation_perf',
              'test_perf')#,'learning_curve'

ldply(dd, function(l) l[colsToKeep])


df = fromJSON(dd)

dt = data.table(dd)

dd = pd.io.json.json_normalize(runs.find(filt))



dd = dd[dd.stop_time > "2015-12-03"]


d = dd[colsToKeep]
d[pd.notnull(d['info.learning_curve'])]
