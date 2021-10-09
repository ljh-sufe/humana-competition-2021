
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.linalg as li
import logging
import torch
from scipy import stats
from scipy.stats import chi2_contingency
from dataCenter.all0var import all0var

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

from sklearn.preprocessing import KBinsDiscretizer, FunctionTransformer


sickVarList = ["auth_3mth_bh_acute_mean_los", "bh_ip_snf_net_paid_pmpm_cost_6to9m_b4",
               "bh_ip_snf_net_paid_pmpm_cost_9to12m_b4", "bh_ip_snf_mbr_resp_pmpm_cost_3to6m_b4", "bh_ip_snf_net_paid_pmpm_cost",
               "bh_ip_snf_mbr_resp_pmpm_cost_6to9m_b4", "bh_ip_snf_net_paid_pmpm_cost_6to9m_b4",
               "bh_ip_snf_net_paid_pmpm_cost_3to6m_b4", "bh_ip_snf_net_paid_pmpm_cost_9to12m_b4",
               "bh_ip_snf_mbr_resp_pmpm_cost_9to12m_b4", "ccsp_065_pmpm_ct", "bh_ip_snf_net_paid_pmpm_cost_0to3m_b4"]


from pandas_profiling import ProfileReport

class dataSet():

    filepath = ""
    pickleFilePath = ""
    categoricalVar = []
    continuousVar = []
    discreteVar = []
    dummyVar = []
    numLEQ0 = None
    dependentVar = ""
    KellyVarTypeDict = None
    vaccineData = None
    externalTrainData_cdc = None
    externalHoldOutData_cdc = None
    X = None
    y = None

    holdOut = None

    evalX = None
    evaly = None

    X_train_label0 = None
    X_train_label1 = None
    y_train_label0 = None
    y_train_label1 = None
    _1sampleLength = None
    _0sampleLength = None


    @staticmethod
    def init():
        print("\n")
        logging.info("start reading file.")
        # sepecify data type
        a = pd.read_csv(r"C:\Users\41409\Documents\wustlCourses\business_competition\TypeOfVariables.csv")
        KellyVarTypeDict = {}
        converter = {"continuous": "Float64",
                     "categorical": "object",
                     "discrete": "Int64"}
        for i in range(0, a.shape[0]):
            type = a.loc[i][1]
            KellyVarTypeDict[a.loc[i][0]] = converter[type]
        KellyVarTypeDict["cons_estinv30_rc"] = "Float64"
        KellyVarTypeDict["atlas_foodhub16"] = "Int64"

        dataSet.KellyVarTypeDict = KellyVarTypeDict
        # read file
        # vaccineData = pd.read_csv(dataSet.filepath,
        #                           low_memory=False,
        #                           na_values=["*", "NaN"],
        #                           engine="c",
        #                           dtype=KellyVarTypeDict)
        vaccineData = pd.read_pickle(r"C:\Users\41409\Desktop\bc\2021_Competition_Training_pickle")
        vaccineData = vaccineData.drop(["Unnamed: 0", "ID", "src_div_id"], axis=1)
        vaccineData = vaccineData.drop(all0var, axis=1)    # 删除全为0的变量
        logging.info("finish reading file.")
        # profile = ProfileReport(vaccineData, minimal=True)
        # profile.to_file("output.html")
        # vaccineData.to_pickle(r"C:\Users\41409\Desktop\bc\2021_Competition_Training_pickle")

        #generate mean

        # zipDF = vaccineData[["zip_cd", "covid_vaccination"]]
        # zipDF["cityZip"] = zipDF["zip_cd"].map(lambda x: x[0:3])
        #
        # zipDF["covid_vaccination"] = zipDF["covid_vaccination"].map(lambda  x: 0 if x == "vacc" else 1)

        #region prepare city area mean
        # cityMean = zipDF.groupby("cityZip").apply(lambda x: x["covid_vaccination"].mean())
        # cityCount = zipDF.groupby("cityZip").apply(lambda x: x.shape[0])
        # areaMean = zipDF.groupby("zip_cd").apply(lambda x: x["covid_vaccination"].mean())
        # areaCount =  zipDF.groupby("zip_cd").apply(lambda x: x.shape[0])
        # city = pd.concat([cityMean, cityCount], axis=1)
        # area = pd.concat([areaMean, areaCount], axis=1)

        # city = pd.read_csv("cityMean.csv", dtype={"cityZip": "string"}).set_index("cityZip")
        # area = pd.read_csv("areaMean.csv", dtype={"zip_cd": "string"}).set_index("zip_cd")
        # zipDF["Mean"] = zipDF["zip_cd"].map(lambda x: area["0"].loc[x] if area["1"].loc[x] >= 10 else -1)
        # areaMean = zipDF.apply(lambda x: city["0"].loc[x["cityZip"]] if x["Mean"] == -1 else x["Mean"], axis=1)
        # vaccineData["areaMean"] = areaMean.astype("Float64")
        #endregion

        dataSet.trainZipcd = vaccineData[["zip_cd"]]
        vaccineData = vaccineData.drop(["zip_cd"], axis=1)
        vaccineData = vaccineData.drop(["race_cd", "sex_cd"], axis=1)
        # vaccineData = vaccineData.drop(sickVarList, axis=1)
        # make numeric data that stored in obj type to numeric, however this will make some categorical data, (1,2,3,...) also numeric

        dataSet.vaccineData = vaccineData

        varTypeDF = dataSet.vaccineData.apply(lambda x: x.dtype)
        dataSet.discreteVar = varTypeDF[varTypeDF == "Int64"].index.to_list()
        dataSet.continuousVar = varTypeDF[varTypeDF == "Float64"].index.to_list()
        dataSet.categoricalVar = varTypeDF[varTypeDF == "object"].index.to_list()

        dataSet.categoricalVar.remove("covid_vaccination")

        dataSet.dependentVar = "covid_vaccination"



    @staticmethod
    def importExternalVariable():

        # cdc data
        cdc_data = pd.read_excel("./dataCenter/CDC_data.xlsx")
        cdc_data = cdc_data.groupby(by="zip3").apply(lambda x: x.mean())
        cdc_data = cdc_data[['Popttl', 'Series_Complete', 'Administered_Dose1_Recip', 'Series_Complete_Pop_Pct',
                             'Administered_Dose1_Pop_Pct', 'Estimated_hesitant', 'Estimated_hesitant_or_unsure',
                             'Estimated_strongly_hesitant', 'SVI_Index ', 'CVAC_level']]
        trainZip = dataSet.trainZipcd.copy()
        trainZip["zip3"] = trainZip["zip_cd"].map(lambda x: int(str(x)[0:3]))

        trainNewdata = trainZip.set_index("zip3").join(cdc_data)
        trainNewdata = trainNewdata.reset_index()[['Popttl', 'Series_Complete', 'Administered_Dose1_Recip', 'Series_Complete_Pop_Pct',
                             'Administered_Dose1_Pop_Pct', 'Estimated_hesitant', 'Estimated_hesitant_or_unsure',
                             'Estimated_strongly_hesitant', 'SVI_Index ', 'CVAC_level']]
        trainNewdata = (trainNewdata - trainNewdata.mean()) / trainNewdata.std()
        dataSet.externalTrainData_cdc = trainNewdata

        holdOutzip = dataSet.holdOutZipcd.copy()
        holdOutzip["zip3"] = holdOutzip["zip_cd"].map(lambda x: int(str(x)[0:3]))
        holdOutNewData = holdOutzip.set_index("zip3").join(cdc_data)
        holdOutNewData = holdOutNewData.reset_index()[
            ['Popttl', 'Series_Complete', 'Administered_Dose1_Recip', 'Series_Complete_Pop_Pct',
             'Administered_Dose1_Pop_Pct', 'Estimated_hesitant', 'Estimated_hesitant_or_unsure',
             'Estimated_strongly_hesitant', 'SVI_Index ', 'CVAC_level']]
        holdOutNewData = (holdOutNewData - holdOutNewData.mean()) / holdOutNewData.std()
        dataSet.externalHoldOutData_cdc = holdOutNewData

        # education data

        edu_data = pd.read_excel("datacenter/Education_by_zipcode.xlsx")
        edu_data = edu_data.drop(["FIPS Code", "Zip Code", "State", "Area name"], axis=1)
        edu_data = edu_data.groupby("zipCode3").apply(lambda x: x.mean())
        eduNewdata = trainZip.set_index("zip3").join(edu_data)
        eduNewdata = eduNewdata.drop(["zip_cd", "zipCode3"], axis=1)
        eduNewdata = (eduNewdata - eduNewdata.mean()) / eduNewdata.std()
        dataSet.externalTrain_EduData = eduNewdata.reset_index(drop=True)

        holdOutedu = trainZip.set_index("zip3").join(edu_data)
        holdOutedu = holdOutedu.drop(["zip_cd", "zipCode3"], axis=1)
        holdOutedu = (holdOutedu - holdOutedu.mean()) / holdOutedu.std()
        dataSet.externalHoldOut_EduData = holdOutedu.reset_index(drop=True)

    @staticmethod
    def processor():
        logging.info("begin processing data")
        # first we need to dummy it! if there is NA in the data, it will be ignored,
        # so there will no NA value in dummyDF
        vaccineData = dataSet.vaccineData
        vaccineData = pd.concat([vaccineData, dataSet.externalTrainData_cdc], axis=1)
        holdOut = dataSet.holdOut
        holdOut = pd.concat([holdOut, dataSet.externalHoldOutData_cdc], axis=1)

        dataSet.continuousVar.extend(dataSet.externalHoldOutData_cdc.columns.to_list())

        
        varList = dataSet.continuousVar + dataSet.discreteVar + dataSet.categoricalVar
        X = pd.concat([vaccineData[varList], holdOut])


        # a = (X == 0).sum()
        # varsManyZero = a[(a / 1500000) > 0.1].index.to_list()
        # varsManyZeroDF = X[varsManyZero]
        # varsManyZeroDF = varsManyZeroDF.fillna(varsManyZeroDF.median())
        # varsManyZeroDF = 1 * (varsManyZeroDF == 0)
        # varsManyZeroDF.columns = ["I_" + x for x in varsManyZero]
        # logging.info("生成了新的01变量，指示某些连续变量取值是否为0")


        # dataSet.categoricalVar.remove("mabh_seg")      # for creating new var
        # est_age_mabh_seg = ToolFactor.newFeature(X, "est_age", "mabh_seg")
        # X = X.drop(["mabh_seg"], axis=1)
        # dataSet.dummyVar.extend(est_age_mabh_seg.columns.to_list())

        logging.info("begin processing 2-value data")
        #region two value data
        catVar = dataSet.categoricalVar
        dummyDF = X[catVar]
        valueCounts = dummyDF.apply(lambda x: x.value_counts().shape[0], axis=0)
        twoValuesVar = valueCounts[valueCounts == 2].index.to_list()

        twoValueDF = X[twoValuesVar]
        # encoding it
        encoder = twoValueDF.apply(lambda x: x.value_counts()[1], axis=0)
        twoValueDF = 1 * (twoValueDF == encoder)

        dataSet.twoValuesVar = twoValuesVar
        #endregion

        #region multivalue data
        logging.info("begin processing multi-value data")
        readyForDummyVar = []
        for var in catVar:
            if var not in twoValuesVar:
                readyForDummyVar.append(var)
        dummyDF = X[readyForDummyVar]
        dummyDF = pd.get_dummies(dummyDF, drop_first=True)
        dataSet.dummyVar.extend(dummyDF.columns.to_list())
        #endregion

        #region continuous daat   use column median to fill the NA value for continuous value
        logging.info("begin processing continuous and discrete data")
        contiVar = dataSet.continuousVar
        contiDF = X[contiVar]
        contiDF = contiDF.fillna(contiDF.median())
        contiDF = (contiDF - contiDF.mean()) / contiDF.std()
        #endregion

        #region  discrete data for discrete value use median to fill NA
        disVar = dataSet.discreteVar
        disDF = X[disVar]
        disDF = disDF.fillna(disDF.median())
        disDF = (disDF - disDF.mean()) / disDF.std()
        #endregion
        # concat them
        # X = pd.concat([contiDF, disDF, twoValueDF, dummyDF, est_age_mabh_seg], axis=1)
        # X = pd.concat([contiDF, disDF, twoValueDF, dummyDF, varsManyZeroDF], axis=1)
        X = pd.concat([contiDF, disDF, twoValueDF, dummyDF], axis=1)
        # X0 = pd.concat([contiDF, disDF], axis=1)
        # X0 = ToolFactor.symmetric_orthogonal_transform(X0)
        # X = pd.concat([X0, twoValueDF, dummyDF], axis=1)
        # X = ToolFactor.symmetric_orthogonal_transform(X)


        trainDataLen = vaccineData.shape[0]
        dataSet.X = X.iloc[0:trainDataLen, :]
        dataSet.holdOut = X.iloc[trainDataLen : , :]


        # dataSet.X["ccsp_065_pmpm_ct"] = 1

        # dataSet.X.to_csv(r"C:\Users\41409\Desktop\allVar.csv")
        # dataSet.X.head(1000).to_csv(r"C:\Users\41409\Desktop\allVarTop1000.csv")

        # prepare y
        dataSet.dependentVar = "covid_vaccination"
        dataSet.y = vaccineData[dataSet.dependentVar].map(lambda x: 0 if x == "vacc" else 1)

        logging.info("finish processing data!")


    @staticmethod
    def initPickleData(pickleFilePath):

        dataSet.X = pd.read_pickle(pickleFilePath + "/X_selected")
        dataSet.y = pd.read_pickle(pickleFilePath + "/y_processed")
        dataSet.holdOut = pd.read_pickle(pickleFilePath + "/holdOut_selected")

    @staticmethod
    def varSelector():

        X = dataSet.X
        y = dataSet.y
        initL = X.shape[1]
        logging.info("input number of variables: " + str(initL))
        data = pd.concat([X, y], axis=1)
        # 如果样本是二分类变量，使用两样本T检验

        # 将数据以是否接种疫苗为标准分组

        data_vac0 = data[data["covid_vaccination"] == 0]
        data_vac1 = data[data["covid_vaccination"] == 1]

        pvalues_contiVar = pd.DataFrame(index=dataSet.continuousVar, columns=["pValue"])
        for var in dataSet.continuousVar:
            result = stats.stats.ttest_ind(data_vac0[var], data_vac1[var])
            pvalues_contiVar.loc[var]["pValue"] = result.pvalue

        dataSet.newContiVar = pvalues_contiVar[pvalues_contiVar["pValue"] < 0.05].index.to_list()

        pvalues_discreVar = pd.DataFrame(index=dataSet.discreteVar, columns=["pValue"])
        for var in dataSet.discreteVar:
            result = stats.stats.ttest_ind(data_vac0[var], data_vac1[var])
            pvalues_discreVar.loc[var]["pValue"] = result.pvalue

        dataSet.newDiscreVar = pvalues_discreVar[pvalues_discreVar["pValue"] < 0.05].index.to_list()
        # In[60]:

        # 对二值型变量进行卡方检验
        pvalues_dummyVar = pd.DataFrame(index=dataSet.dummyVar, columns=["pValue"])
        for var in dataSet.dummyVar:
            # 创建交叉表
            data_table = pd.crosstab(data[var], data["covid_vaccination"])
            # 计算卡方统计值与p值
            result = chi2_contingency(data_table)
            pvalues_dummyVar.loc[var]["pValue"] = result[1]

        dataSet.newDummyVar = pvalues_dummyVar[pvalues_dummyVar["pValue"] < 0.05].index.to_list()

        newVar = dataSet.newContiVar +  dataSet.newDiscreVar +  dataSet.newDummyVar

        dataSet.newVar = newVar
        dataSet.X = dataSet.X[newVar]
        dataSet.holdOut = dataSet.holdOut[newVar]
        outL = newVar.__len__()
        logging.info("ouput number of variables: " + str(outL))
        # In[72]:
        1
        # pvalueDF_ttest = pd.DataFrame(pvalues_ttest)
        # pvalueDF_chi = pd.DataFrame(pvalues_chi)
        # columnsDF = pd.DataFrame(data_no.columns)
        # resultDF = pd.concat([columnsDF, pvalueDF_ttest, pvalueDF_chi], axis=1, keys=["varname", "ttest", "chisquare"])
        #
        # # In[74]:
        #
        # print(resultDF)



        # var = "total_bh_copay_pmpm_cost_t_9-6-3m_b4_New"
        # plt.hist(data_no[var])
        # plt.hist(data_yes[var])
        # plt.show()
        # X[var]

    @staticmethod
    def loadHoldOut():
        # city = pd.read_csv("cityMean.csv", dtype={"cityZip": "string"}).set_index("cityZip")
        # area = pd.read_csv("areaMean.csv", dtype={"zip_cd": "string"}).set_index("zip_cd")


        KellyVarTypeDict = dataSet.KellyVarTypeDict
        
        # holdOut = pd.read_csv(r"C:\Users\41409\Desktop\bc\2021_Competition_Holdout.csv",
        #                      low_memory=False,
        #                      na_values=["*", "NaN"],
        #                      engine="c",
        #                      dtype=KellyVarTypeDict)
        holdOut = pd.read_pickle(r"C:\Users\41409\Desktop\bc\2021_Competition_Holdout_pickle123")
        # def get_area_value(x):
        #     try:
        #         return area["0"].loc[x] if area["1"].loc[x] >= 10 else -1
        #     except:
        #         return -1

        # zipDF = holdOut[["zip_cd"]]
        # zipDF["cityZip"] = zipDF["zip_cd"].map(lambda x: x[0:3])
        #
        # zipDF["Mean"] = zipDF["zip_cd"].map(lambda x: get_area_value(x))
        # areaMean = zipDF.apply(lambda x: city["0"].loc[x["cityZip"]] if x["Mean"] == -1 else x["Mean"], axis=1)
        # holdOut["areaMean"] = areaMean.astype("Float64")

        dataSet.holdOutZipcd = holdOut[["zip_cd"]]
        holdOut = holdOut.drop(["Unnamed: 0", "ID", "src_div_id"], axis=1)
        holdOut = holdOut.drop(["zip_cd"], axis=1)
        holdOut = holdOut.drop(["race_cd", "sex_cd"], axis=1)
        holdOut = holdOut.drop(all0var, axis=1)  # 删除全为0的变量

        logging.info("finish reading holdout")
        dataSet.holdOut = holdOut



    @staticmethod
    def initEvalData(ratio=0.2):
        1
        X, y = dataSet.X, dataSet.y
        validationRatio = ratio
        len_ = y.shape[0]
        shuffledIndex = np.random.permutation(len_)
        trainIndex = shuffledIndex[0: int((1 - validationRatio) * len_)]
        evalIndex = shuffledIndex[int((1 - validationRatio) * len_):]

        X_train = X.loc[trainIndex].reset_index(drop=True)
        y_train = y.loc[trainIndex].reset_index(drop=True)

        X_train_label0 = X_train[y_train == 0].reset_index(drop=True)
        X_train_label1 = X_train[y_train == 1].reset_index(drop=True)
        y_train_label0 = y_train[y_train == 0].reset_index(drop=True)
        y_train_label1 = y_train[y_train == 1].reset_index(drop=True)
        _1sampleLength = X_train_label1.shape[0]  # 1 more than 0
        _0sampleLength = X_train_label0.shape[0]

        evalX = X.loc[evalIndex].reset_index(drop=True)
        evaly = y.loc[evalIndex].reset_index(drop=True)
        dataSet.evalX = evalX
        dataSet.evaly = evaly

        dataSet.X_train_label0 = X_train_label0
        dataSet.X_train_label1 = X_train_label1
        dataSet.y_train_label0 = y_train_label0
        dataSet.y_train_label1 = y_train_label1
        dataSet._1sampleLength = _1sampleLength
        dataSet._0sampleLength = _0sampleLength



        


class ToolFactor():
    @staticmethod
    def upper_lower(x, lower, upper):
        if lower <= x <= upper:
            return x
        elif x < lower:
            return lower
        elif x > upper:
            return upper
        else:
            return None

    @staticmethod
    def pullback(factor):
        quantiles_lower = factor.quantile(0.01)
        quantiles_upper = factor.quantile(0.99)
        return factor.map(lambda x: ToolFactor.upper_lower(x, quantiles_lower, quantiles_upper))

    @staticmethod
    def newFeature(X, groupVar, dependentVar):

        '''以groupVar为分组变量。dependentVar是一个分类变量，生成融合特征的dummy变量'''

        age = X[groupVar]

        if groupVar not in dataSet.categoricalVar:
            # 5 groups
            ma = max(age)
            mi = min(age)
            range_ = ma - mi
            bins = [mi + x * range_/5  for x in range(0, 6)]
            labels = [i for i in range(0, 5)]
            # transformer = FunctionTransformer(
            #     pd.cut, kw_args={'bins': bins, 'labels': labels, 'retbins': False}
            # )
            groupAge = pd.cut(age, bins=bins, labels=labels, retbins=False)
        else:
            groupAge = age

        depend = X[dependentVar]


        a = pd.concat([groupAge, depend], axis=1)
        b = a.values
        a = a.apply(lambda x: str(x[0])+str(x[1]) , axis=1)


        logging.info("finish generating new feature " + groupVar + "&" + dependentVar)
        return pd.get_dummies(a)

    @staticmethod
    def symmetric_orthogonal_transform(F):
        '''传入一个矩阵，输出对称正交后的矩阵'''

        F_sample = F.sample(100000).astype(float)
        M = F_sample.T.dot(F_sample)
        D = np.diag(li.eig(M)[0])
        D = np.abs(D)
        U = li.eig(M)[1]
        Dx = np.diag(li.eig(M)[0] ** (-0.5))
        S = U.dot(Dx).dot(U.T)

        F_new = F.dot(S)


        M = F.T.dot(F)
        D = np.diag(li.eig(M)[0])
        D = np.abs(D)
        U = li.eig(M)[1]
        Dx = np.diag(li.eig(M)[0] ** (-0.5))
        S = U.dot(Dx).dot(U.T)

        F_new = F.dot(S)




        return F_new
