from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.classification import DecisionTreeClassifier,RandomForestClassifier,NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import IndexToString,StringIndexer,VectorIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

spark=SparkSession.builder.getOrCreate()

df=spark.read.csv("/home/makram/Downloads/project/Iris.csv",inferSchema=True,header=True)
df.show()
df.groupBy('Species').count().show()
df.printSchema()

assembler=VectorAssembler(inputCols=df.columns[:-1],outputCol='features')
second_df = assembler.transform(df).select('features','Species')
labelIndexer = StringIndexer(inputCol='Species',
                             outputCol='indexedSpecies').fit(second_df)
labelIndexer.transform(second_df).show(10, True)

featureIndexer =VectorIndexer(inputCol="features", \
                                  outputCol="indexedFeatures", \
                                  maxCategories=4).fit(second_df)
featureIndexer.transform(second_df).show(5, True)

train_df, test_df=second_df.randomSplit([0.65,0.35])
train_df.count()
test_df.count()

dTree=DecisionTreeClassifier(labelCol='indexedSpecies',featuresCol='features')

rdForest=RandomForestClassifier(labelCol='indexedSpecies',featuresCol='features',numTrees=20)

nBayes=DecisionTreeClassifier(labelCol='indexedSpecies',featuresCol='features')


labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)

tpipeline = Pipeline(stages=[labelIndexer,featureIndexer, dTree,labelConverter])
rpipeline = Pipeline(stages=[labelIndexer, featureIndexer, rdForest,labelConverter])
npipeline = Pipeline(stages=[labelIndexer, featureIndexer,nBayes,labelConverter])


tmodel = tpipeline.fit(train_df)
tpredictions = tmodel.transform(test_df)
tpredictions.select("features","Species","predictedLabel").show(5)

rmodel = rpipeline.fit(train_df)
rpredictions = rmodel.transform(test_df)
rpredictions.select("features","Species","predictedLabel").show(5)

nmodel = npipeline.fit(train_df)
npredictions = nmodel.transform(test_df)
tpredictions.select("features","Species","predictedLabel").show(5)


#Evaluate Tree
tevaluator = MulticlassClassificationEvaluator(
    labelCol="indexedSpecies", predictionCol="prediction", metricName="accuracy")
accuracy = tevaluator.evaluate(tpredictions)
print(accuracy)
print("tTest Error = %g" % (1.0 - accuracy))
dtModel = tmodel.stages[-2]
print(dtModel) 


#evaluate Random forest 
revaluator = MulticlassClassificationEvaluator(
    labelCol="indexedSpecies", predictionCol="prediction", metricName="accuracy")
accuracy = revaluator.evaluate(rpredictions)
print(accuracy)
print("rTest Error = %g" % (1.0 - accuracy))
rfModel = rmodel.stages[-2]
print(rfModel) 


#evaluate Naive Bayes
nevaluator = MulticlassClassificationEvaluator(
    labelCol="indexedSpecies", predictionCol="prediction", metricName="accuracy")
accuracy = nevaluator.evaluate(npredictions)
print(accuracy)
print("nTest Error = %g" % (1.0 - accuracy))
nbModel = nmodel.stages[-2]
print(nbModel) 
