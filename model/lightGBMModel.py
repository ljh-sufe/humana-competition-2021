

import numpy as np
import pandas as pd
import random
from lightgbm import LGBMClassifier
import lightgbm as lgbm
from dataCenter.dataCenter import dataSet
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc, roc_auc_score, plot_roc_curve
from dataCenter.vartype import newContiVar, newDummyVar, newDiscreVar
from matplotlib import pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class lightGBMModel():
    ifEval = None
    classifierDict = {}
    evalPredictProba = None

    def __init__(self, ifEval=None):
        self.ifEval = ifEval

    def trainModel(self):
        # validationRatio = 0.2
        # len_ = y.shape[0]
        # shuffledIndex = np.random.permutation(len_)
        # trainIndex = shuffledIndex[0: int((1 - validationRatio) * len_)]
        # evalIndex = shuffledIndex[int((1 - validationRatio) * len_):]
        #
        # X_train = X.loc[trainIndex].reset_index(drop=True)
        # y_train = y.loc[trainIndex].reset_index(drop=True)
        #
        # X_train_label0 = X_train[y_train == 0].reset_index(drop=True)
        # X_train_label1 = X_train[y_train == 1].reset_index(drop=True)
        # y_train_label0 = y_train[y_train == 0].reset_index(drop=True)
        # y_train_label1 = y_train[y_train == 1].reset_index(drop=True)
        # _1sampleLength = X_train_label1.shape[0]  # 1 more than 0
        # _0sampleLength = X_train_label0.shape[0]

        _1sampleLength = dataSet._1sampleLength
        _0sampleLength = dataSet._0sampleLength
        X_train_label0 = dataSet.X_train_label0
        X_train_label1 = dataSet.X_train_label1
        y_train_label0 = dataSet.y_train_label0
        y_train_label1 = dataSet.y_train_label1
        indexIndi = random.sample(range(0, _1sampleLength), _1sampleLength)

        indiDict = {}
        indiDict[1] = indexIndi[0: _0sampleLength]
        indiDict[2] = indexIndi[_0sampleLength: 2 * _0sampleLength]
        indiDict[3] = indexIndi[_0sampleLength * 2: 3 * _0sampleLength]
        indiDict[4] = indexIndi[_0sampleLength * 3: 4 * _0sampleLength]
        # indiDict[5] = indexIndi[_0sampleLength * 4: ]

        for i in range(1, 5):
            index = indiDict[i]
            epochX = pd.concat([X_train_label1.loc[index], X_train_label0], axis=0).reset_index(drop=True)
            epochy = pd.concat([y_train_label1.loc[index], y_train_label0], axis=0).reset_index(drop=True)
            self.oneEpochTrain(epochX, epochy, i)

        classifier = self.classifierDict[1]
        split = pd.DataFrame(index=classifier.feature_name_, columns=[1,2,3,4])
        for i in range(1, 5):
            classifier = self.classifierDict[i]
            split[i] = classifier.feature_importances_
        split.to_csv("lightGBMresult.csv")

        if self.ifEval == True:

            evalX = dataSet.evalX
            evaly = dataSet.evaly
            # for eval dataSet
            evalProba = 0
            for i in range(1, 5):
                classifier = self.classifierDict[i]

                evalProba = evalProba + classifier.predict_proba(evalX.values.astype(float))[:, 1]
            evalProba = evalProba / 4
            aucScore = roc_auc_score(evaly, evalProba)

            eval_y_pred = 1 * (evalProba > 0.5)


            tn, fp, fn, tp = confusion_matrix(evaly, eval_y_pred).ravel()
            accuracy = (tn + tp) / (tn + fp + fn + tp)
            if (tp + fn) == 0:
                recall = np.nan
            else:
                recall = (tp) / (tp + fn)
            precision = (tp) / (tp + fp)
            print("for eval  dataset, accuracy: ", round(accuracy, 4), "  recall: ", round(recall, 4), "  precision: ",
                  precision, "  auc: ", aucScore)

            print(
                pd.DataFrame(index=["predict 1", "predict 0"], columns=["true 1", "true 0"], data=[[tp, fp], [fn, tn]]))

            self.evalPredictProba = evalProba


    def oneEpochTrain(self, X, y, i):

        classifier = LGBMClassifier(boosting_type='gbdt',
                                    num_leaves=200,
                                    max_depth=8,
                                    learning_rate=0.05,
                                    n_estimators=300,
                                    min_data_in_leaf=1000,
                                    device="gpu")      ### 0.73  &  0.68

        # classifier = LGBMClassifier(boosting_type='gbdt',
        #                             num_leaves=60,
        #                             max_depth=6,
        #                             learning_rate=0.01,
        #                             n_estimators=300,
        #                             min_data_in_leaf=500,
        #                             device="gpu")     ### 0.685   0.67

        # classifier = LGBMClassifier(boosting_type='gbdt',
        #                             num_leaves=60,
        #                             max_depth=6,
        #                             learning_rate=0.05,
        #                             n_estimators=300,
        #                             min_data_in_leaf=1000,
        #                             device="gpu")      ###


        classifier.fit(X.astype(float), y)
        self.classifierDict[i] = classifier

        epoch_y_pred = classifier.predict(X.values.astype(float))
        epochProba = classifier.predict_proba(X.values.astype(float))
        aucScore = roc_auc_score(y, epochProba[:, 1])
        tn, fp, fn, tp = confusion_matrix(y, epoch_y_pred).ravel()
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        if (tp + fn) == 0:
            recall = np.nan
        else:
            recall = (tp) / (tp + fn)
        precision = (tp) / (tp + fp)
        print("for train dataset, accuracy: ", round(accuracy, 4), "  recall: ", round(recall, 4), "  precision: ",
              precision, "  auc: ", aucScore)

        print(
            pd.DataFrame(index=["predict 1", "predict 0"], columns=["true 1", "true 0"], data=[[tp, fp], [fn, tn]]))

        plt.figure()
        lgbm.plot_importance(classifier, max_num_features=30, importance_type="split")
        plt.show()
        plt.figure()
        lgbm.plot_tree(classifier)
        # plt.savefig("plot_tree.png", dpi=500)
        plt.show()
        classifier.feature_importances_
        classifier.feature_name_
        split = pd.DataFrame(index=classifier.feature_name_, columns=["split"], data=classifier.feature_importances_)
        # split.to_csv("split.csv")

        # para = pd.read_csv("para.csv")
        # linearPred = X.values.dot(para[["para"]].values)
        # linearPred = linearPred.squeeze()
        # linearPred = np.array([sigmoid(j) for j in linearPred])
        # print(np.corrcoef(epochProba[:, 1], linearPred))
        #
        #
        #
        # twoModwelProba = 0.5 * (linearPred + epochProba[:, 1])
        # roc_auc_score(y, twoModwelProba)
        # roc_auc_score(y, linearPred)
        # roc_auc_score(y, epochProba[:, 1])

    def predictProba(self, XDF):
        1
        Proba = 0
        for i in range(1, 5):
            classifier = self.classifierDict[i]

            Proba = Proba + classifier.predict_proba(XDF.values.astype(float))[:, 1]
        Proba = Proba / 4

        ID = pd.read_csv(r"C:\Users\41409\Desktop\bc\2021_Competition_Holdout.csv", usecols=["ID"])
        ID = ID[["ID"]]
        ID["SCORE"] = Proba
        ID["RANK"] = ID["SCORE"].rank(ascending=False).astype(int)

        import datetime
        today = datetime.datetime.today()
        yyyymmdd = str(today)[0:10].replace("-", "")
        ID.to_csv(r"C:\Users\41409\Desktop\bc\result\2021CaseCompetition_Ruoyuan_Gao_" + yyyymmdd + "_lightgbm.csv", index=False)