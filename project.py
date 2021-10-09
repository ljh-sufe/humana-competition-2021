


from pandas_profiling import ProfileReport
from dataCenter.vartype import newContiVar, newDummyVar, newDiscreVar
from dataCenter.dataCenter import dataSet, ToolFactor
from model.logisticRegModel import logisticRegressionModel
from model.lightGBMModel import lightGBMModel
from indicatorAnalyzer.analyzer import indicatorAnalyzer
import warnings

warnings.filterwarnings('ignore')



# init data
dataSet.filepath = r"C:\Users\41409\Desktop\bc\2021_Competition_Training.csv"
# dataSet.filepath = r"C:\Users\41409\Desktop\top1000.csv"



pickleFilepath = "C:/Users/41409/Desktop/bc"

dataSet.init()



# dataSet.loadHoldOut()
# dataSet.importExternalVariable()
#
# dataSet.processor()
# dataSet.varSelector()

# # dataSet.initPickleData(pickleFilepath)
# dataSet.initEvalData(ratio=0.2)


# model1 = logisticRegressionModel(lr=0.01, optimizer="ADAM", epochs=5, ifEval=True, ifQuadratic=False)
# model1.trainModel()
# model2 = lightGBMModel(ifEval=True)
# model2.trainModel()

# model1.predictProba(dataSet.holdOut)
# model2.predictProba(dataSet.holdOut)


ana = indicatorAnalyzer()
ana.analyze_twofeatures(["cons_chmi", "est_age"])

# ana.analyze_var(["atlas_pct_free_lunch14", "est_age", "cms_orig_reas_entitle_cd", "rx_gpi2_17_pmpm_cost_t_12-9-6m_b4", "rx_overall_gpi_pmpm_ct_0to3m_b4"])





model1.evalPredictProba.ravel()
model2.evalPredictProba
import numpy as np
np.corrcoef(model1.evalPredictProba.ravel(), model2.evalPredictProba)

r = 0.9
p = r * model1.evalPredictProba.ravel() + (1-r) * model2.evalPredictProba

# dataSet.evaly
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, plot_roc_curve, RocCurveDisplay
roc_auc_score(dataSet.evaly, p)

fig, ax = plt.subplots()
fpr, tpr, _ = roc_curve(dataSet.evaly, model1.evalPredictProba.ravel())
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax)
fpr, tpr, _ = roc_curve(dataSet.evaly, model2.evalPredictProba)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax)
plt.legend(["logistic", "lightgbm"], loc="lower right")
plt.title("ROC curve comparison", loc="left", fontdict={'fontsize': 20})
plt.style.use('seaborn-whitegrid')
plt.savefig("roc.png", dpi=500)
#




