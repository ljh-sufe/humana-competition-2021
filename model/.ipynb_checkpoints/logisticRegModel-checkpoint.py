import logging
import os

import torch
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from dataCenter.dataCenter import dataSet
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from plotToExcel.plotter import scatterPlot_DF
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score, plot_roc_curve
from dataCenter.vartype import newContiVar, newDummyVar, newDiscreVar

def init_model_param(model, param):
    """只能初始化linear层的参数"""

    for x in model.modules():
        if isinstance(x, nn.Linear):
            x.weight.data = param.reshape(x.weight.data.shape)
            break

from torch import nn


class NNLR(nn.Module):
    '''
    pytorch logistic regression nn model
    '''
    def __init__(self, inputSize):
        super(NNLR, self).__init__()
        # self.BN = nn.BatchNorm1d(num_features=inputSize)
        self.lr = nn.Linear(in_features=inputSize, out_features=1)
        self.sm = nn.Sigmoid()
        self.loss = nn.BCELoss()
        self.optimizer = None

    def forward(self, x):
        # x = self.BN(x)
        x = self.lr(x)
        x = self.sm(x)
        return x



class QuadraticLR(nn.Module):
    '''
    pytorch logistic regression nn model
    '''
    def __init__(self, inputSize):
        super(QuadraticLR, self).__init__()
        # self.BN = nn.BatchNorm1d(num_features=inputSize)
        self.lr = nn.Linear(in_features=inputSize, out_features=1)
        self.blr = nn.Bilinear(in1_features=inputSize, in2_features=inputSize, out_features=1)
        self.sm = nn.Sigmoid()
        self.loss = nn.BCELoss()
        self.optimizer = None

    def forward(self, x):
        # x = self.BN(x)
        lrx = self.lr(x)
        blrx = self.blr(x, x)
        x = 0.5 * (lrx + blrx)
        x = self.sm(x)
        return x



class logisticRegressionModel():
    ''':cvar
    model we will use to train data'''
    lossList = []
    accuracyList = []
    recallList = []
    precisionList = []
    aucScoreList = []
    learning_rate = 0.001
    epochs = 50
    model = None
    ifEval = None
    ifQuadratic = None
    evalPredictProba = None


    def __init__(self, lr, epochs, optimizer, ifEval=None, ifQuadratic=None):
        self.learning_rate = lr
        self.epochs = epochs
        self.ifEval = ifEval
        self.optimizer = optimizer
        self.ifQuadratic = ifQuadratic
        self.get_para_ = None
        self.pet_paraDF = None
        pass

    def trainModel(self):

        tb = SummaryWriter()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.ifEval == True:

            evalX = dataSet.evalX
            evalX = torch.Tensor(evalX.values.astype(float)).to(device)
            evaly = dataSet.evaly
        _1sampleLength = dataSet._1sampleLength
        _0sampleLength = dataSet._0sampleLength
        X_train_label0 = dataSet.X_train_label0
        X_train_label1 = dataSet.X_train_label1
        y_train_label0 = dataSet.y_train_label0
        y_train_label1 = dataSet.y_train_label1

        #region model
        NNModel = NNLR(X_train_label0.shape[1])
        if self.ifQuadratic:
            NNModel = QuadraticLR(X_train_label0.shape[1])

        NNModel.to(device)

        # para = pd.read_csv("para.csv")
        # para = torch.Tensor(para["para"].values).to(device)
        # init_model_param(NNModel, param=para)

        if self.optimizer == "SGD":
            NNModel.optimizer = torch.optim.SGD(NNModel.parameters(), lr=self.learning_rate)
        elif self.optimizer == "ADAM":
            NNModel.optimizer = torch.optim.Adam(NNModel.parameters(), lr=self.learning_rate)
        #endregion
        scheduler = ReduceLROnPlateau(NNModel.optimizer, 'min', factor=0.5, patience=5)
        epochs = self.epochs
        for epoch in range(epochs):
            print("epoch " + str(epoch))
            # 构造平衡样本
            indexIndi = random.sample(range(0, _1sampleLength), _0sampleLength)    # indexIndicator, 从1样本中抽0样本

            epochX = pd.concat([X_train_label1.loc[indexIndi], X_train_label0], axis=0).reset_index(drop=True)
            epochy = pd.concat([y_train_label1.loc[indexIndi], y_train_label0], axis=0).reset_index(drop=True)

            # 再次打乱数据 for batch
            shuffledIndex = np.random.permutation(2 * _0sampleLength)
            epochX = torch.Tensor(epochX.loc[shuffledIndex].values.astype(float)).to(device)
            epochy = torch.Tensor(epochy.loc[shuffledIndex].values.astype(float)).to(device)

            self.BatchTrain(NNModel, epochX, epochy)

            tb.add_scalar("training_Loss", self.lossList[epoch], epoch)    # 绘制loss曲线
            tb.add_scalar("training_AUC", self.aucScoreList[epoch], epoch)  # 绘制auc曲线
            tb.add_scalar("LR", NNModel.optimizer.param_groups[0]["lr"], epoch)  # 绘制学习率

            if epoch%10 == 5:
                for x in NNModel.parameters():
                    break
                para = x.cpu().detach().numpy()
                a = pd.DataFrame(index=X_train_label0.columns, columns=["para"], data=para.T)

                a = a.sort_values(by="para", ascending=False)
                print(pd.concat([a.head(20).reset_index(), a.tail(20).reset_index()], axis=0))



            if self.ifEval == True:
                NNModel.eval()
                eval_y_pred = NNModel(evalX)
                val_loss = NNModel.loss(eval_y_pred.reshape(eval_y_pred.shape[0],), torch.Tensor(evaly.values).to(device))
                print("学习率为", NNModel.optimizer.param_groups[0]["lr"])
                eval_y_pred = eval_y_pred.cpu().detach().numpy()

                self.evalPredictProba = eval_y_pred

                aucScore = roc_auc_score(evaly, eval_y_pred)
                tb.add_scalar("eval_AUC", aucScore, epoch)  # 绘制auc曲线
                label_y_pred = 1 * (eval_y_pred > 0.5)

                tn, fp, fn, tp = confusion_matrix(evaly, label_y_pred).ravel()
                accuracy = (tn + tp) / (tn + fp + fn + tp)
                if (tp + fn) == 0:
                    recall = np.nan
                else:
                    recall = (tp) / (tp + fn)
                precision = (tp) / (tp + fp)
                print("for eval dataset, accuracy: ", round(accuracy, 4), "  recall: ", round(recall, 4), "  precision: ", precision, "  auc: ", aucScore)
                scheduler.step(val_loss)
            else:
                pass
        self.model = NNModel

        eval_y_pred = pd.DataFrame(eval_y_pred, columns=["y_pred"])
        scatterPlot_DF(filepath="logistic_eval.xlsx", DF=eval_y_pred, columns=["y_pred"])
        tb.close()


        for x in NNModel.parameters():
            break
        para = x.cpu().detach().numpy()
        a = pd.DataFrame(index=X_train_label0.columns, columns=["para"], data=para.T)
        self.get_para_ = para
        self.pet_paraDF = a
        self.compute_standard_deviation()



        a.to_csv("para.csv")


    def BatchTrain(self, model, X, y):

        lossList = []
        accuracyList = []
        recallList = []
        precisionList = []
        aucList = []
        batch_size = 4096 * 25
        model = model
        model.train()  ####################
        optimizer = model.optimizer
        batches = int(y.shape[0] / batch_size)+1
        for i in range(0, batches):
            batchX = X[i * batch_size: (i + 1) * batch_size]
            batchy = y[i * batch_size: (i + 1) * batch_size]
            batch_y_pred = model(batchX).reshape(batchX.shape[0])  # 根据逻辑回归模型拟合出的y值

            loss = model.loss(batch_y_pred, batchy)  # 计算损失函数
            # loss = loss + 0.02 * torch.norm(model.lr.weight, p=2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lossItem = loss.item()
            y_pred = batch_y_pred.cpu().detach().numpy()
            y_true = batchy.cpu().detach().numpy().astype(int)
            aucScore = roc_auc_score(y_true, y_pred)

            y_pred = 1*(y_pred > 0.5)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            accuracy = (tn+tp) / (tn+fp+fn+tp)
            if (tp+fn) == 0:
                recall = np.nan
            else:
                recall = (tp) / (tp + fn)
            precision = (tp) / (tp + fp)


            lossList.append(lossItem)
            accuracyList.append(accuracy)
            recallList.append(recall)
            precisionList.append(precision)
            aucList.append(aucScore)

        print("loss: ", round(np.mean(lossList), 4), "  accuracy: ", round(np.mean(accuracyList), 4), "  recall: ", round(np.mean(recallList), 4), "  precision: ", round(np.mean(precisionList), 4), "  auc: ", round(np.mean(aucList), 4))
        self.lossList.append(np.mean(lossList))
        self.accuracyList.append(np.mean(accuracyList))
        self.recallList.append(np.mean(recallList))
        self.precisionList.append(np.mean(precisionList))
        self.aucScoreList.append(np.mean(aucList))

    def predictProba(self, XDF):
        '''X dataframe'''

        model = self.model
        predictX = torch.Tensor(XDF.values.astype(float)).to("cuda")
        proba = model(predictX)
        proba = proba.cpu().detach().numpy()
        # ID = pd.read_pickle(r"C:\Users\41409\Desktop\bc\2021_Competition_Holdout_pickle")
        ID = pd.read_csv(r"C:\Users\41409\Desktop\bc\2021_Competition_Holdout.csv", usecols=["ID"])
        ID = ID[["ID"]]
        ID["SCORE"] = proba
        ID["RANK"] = ID["SCORE"].rank(ascending=False).astype(int)



        import datetime
        today = datetime.datetime.today()
        yyyymmdd = str(today)[0:10].replace("-", "")
        ID.to_csv(r"C:\Users\41409\Desktop\bc\result\2021CaseCompetition_Ruoyuan_Gao_" + yyyymmdd+ ".csv", index=False)






    def compute_standard_deviation(self):
        _1sampleLength = dataSet._1sampleLength
        _0sampleLength = dataSet._0sampleLength
        X_train_label0 = dataSet.X_train_label0
        X_train_label1 = dataSet.X_train_label1
        y_train_label0 = dataSet.y_train_label0
        y_train_label1 = dataSet.y_train_label1

        para = self.get_para_
        model = self.model


        wholePredict = None
        finalXTVX = None
        for i in tqdm(range(0, 40)):
            # print(i)
            tempData = dataSet.X.iloc[25000*i : 25000*(i+1)]
            torchTempData = torch.Tensor(tempData.values.astype(float)).to("cuda")
            tempPredict = self.model(torchTempData)
            # tempPredict = tempPredict.detach().cpu().numpy()
            tempPredict = tempPredict.reshape(tempPredict.shape[0])
            XTVX = torch.mm(torchTempData.T, torch.diag(tempPredict))
            XTVX = torch.mm(XTVX, torchTempData)
            XTVX = XTVX.cpu().detach().numpy()
            if np.isnan(XTVX).sum() > 0:
                logging.ERROR("出现nan！")
                break
            if finalXTVX is None:
                # wholePredict = tempPredict
                finalXTVX = XTVX
            else:
                # wholePredict = np.vstack((wholePredict, tempPredict))
                finalXTVX = finalXTVX + XTVX
        import numpy.linalg as lin
        Pv = lin.pinv(finalXTVX, rcond = 0.0000001)
        Pv = np.diag(Pv)
        paraDF = self.pet_paraDF.copy()
        paraDF["std"] = np.sqrt(Pv)
        paraDF["lb"] = paraDF["para"] - 2 * paraDF["std"]
        paraDF["ub"] = paraDF["para"] + 2 * paraDF["std"]
        paraDF["0leqlb"] = 1 * (paraDF["lb"] < 0)
        paraDF["0sequb"] = 1 * (paraDF["ub"] > 0)
        paraDF["0inbound"] = paraDF["0leqlb"] * paraDF["0sequb"]
        paraDF = paraDF[["para", "std", "lb", "ub", "0inbound"]]
        paraDF[paraDF["0inbound"] == 0].sort_values(by="para")

        paraDF.to_csv("logistic_result.csv")

        # 由于样本量太大，所以需要抽样计算sd
