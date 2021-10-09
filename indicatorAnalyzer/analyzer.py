
from dataCenter.dataCenter import dataSet
from plotToExcel.plotter import scatterPlot_DF, excelOutput

class indicatorAnalyzer():
    variables_to_analyze = None


    def analyze_var(self, varList):

        exceloutputer = excelOutput()

        vaccineData = dataSet.vaccineData
        filename = ""
        for x in varList:
            filename = filename + "_" + x
        dataDict = {}
        for var in varList:
            dataDict[var] = vaccineData[[var, dataSet.dependentVar]]

        exceloutputer.var_analyzer_workbook(filepath="C:\\Users\\41409\\Desktop\\bc\\result\\anavar_"+ filename +".xlsx", dataDict=dataDict)


    def analyze_twofeatures(self, varList):

        exceloutputer = excelOutput()
        vaccineData = dataSet.vaccineData
        DF = vaccineData[varList + [dataSet.dependentVar]]
        filename = varList[0] + varList[1]
        exceloutputer.twofeatureCompare_workbook(DF, filepath="C:\\Users\\41409\\Desktop\\bc\\result\\twofeature_"+ filename +".xlsx")