import tqdm
import features
import extractfeature
import trainer
import predictor
import utility

r = []
r.append(features.ByteHistogram())
r.append(features.ByteEntropyHistogram())
r.append(features.StringExtractor())
r.append(features.GeneralFileInfo())
r.append(features.HeaderFileInfo())
r.append(features.SectionInfo())
r.append(features.ImportsInfo())
r.append(features.ExportsInfo())



trainsetdir = '/data/myAI/dataset/trainset/'
trainsetlabelpath = '/data/myAI/dataset/trainset_label.csv'
trainsetfeaturepath = '/data/myAI/dataset/features.jsonl'


extractor = extractfeature.Extractor(trainsetdir, trainsetlabelpath, trainsetfeaturepath, r)
extractor.run()

modeldir = '/data/myAI/dataset/aimodel/'
train = trainer.Trainer(trainsetfeaturepath, modeldir)
train.run()

featurelist = '/data/myAI/dataset/features.jsonl'
features = utility.readonelineFromjson(featurelist)
feature_parser = utility.FeatureType()
featureobjs = feature_parser.parsing(features)

modelpath = '/data/myAI/dataset/aimodel/GradientBoosted_model.txt'
testdir = '/data/sibal/'
outputpath = '/data/myAI/dataset/result.csv'
predict = predictor.Predictor(modelpath, testdir, featureobjs, outputpath)
predict.run()

