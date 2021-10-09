
from dataCenter.dataCenter import dataSet
from plotToExcel.plotter import scatterPlot_DF, excelOutput

class indicatorAnalyzer():
    variables_to_analyze = None


    def analyze_var(self, varList):

        exceloutputer = excelOutput()

        vaccineData = dataSet.vaccineData

        dataDict = {}
        for var in varList:
            dataDict[var] = vaccineData[[var, dataSet.dependentVar]]

        exceloutputer.var_analyzer_workbook(filepath="Varanalyze.xlsx", dataDict=dataDict)


    def analyze_twofeatures(self, varList):

        exceloutputer = excelOutput()
        vaccineData = dataSet.vaccineData
        DF = vaccineData[varList + [dataSet.dependentVar]]

        exceloutputer.twofeatureCompare_workbook(DF, filepath=r"C:\Users\41409\Desktop\bc\result\twovar.xlsx")