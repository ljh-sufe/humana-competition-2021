import logging

import xlsxwriter
import string
import pandas as pd
from dataCenter.dataCenter import dataSet
import logging
import warnings

warnings.filterwarnings('ignore')

def plot_scatter_(filepath, dataDict, datalen ):
    workbook = xlsxwriter.Workbook(filepath)
    worksheet = workbook.add_worksheet()
    chart = workbook.add_chart({"type": "scatter",
                                'subtype': 'straight_with_markers'})
    worksheet.write_column("A1", [i for i in range(1, datalen)])
    i = 0
    uppercase = string.ascii_uppercase[1:]
    for key in dataDict:
        colIndicator = uppercase[i]+"1"
        worksheet.write_column(colIndicator, dataDict[key])
        i = i + 1

        chart.add_series({
            "name": key,
            "categories": "Sheet1!$A$1:$A$" + str(datalen),
            "values": "=Sheet1!$"+colIndicator+"$1:$"+colIndicator + str(datalen),
            "marker": {
                'type': 'square',
                "size": 2,
            },
            "line": {
                "width": 1,
            },
        })


    # chart.set_style(14)
    worksheet.insert_chart(uppercase[dataDict.__len__()+3] + "1" ,chart)
    workbook.close()


def scatterPlot_DF(filepath, DF, columns):
    DF = DF.sort_values(by=columns[0])
    datalen = DF.shape[0]
    dataDict = {}
    for col in columns:
        dataDict[col] = DF[col]

    plot_scatter_(filepath=filepath, dataDict=dataDict, datalen=datalen)




class excelOutput():

    workbook = None

    def var_analyzer_workbook(self, dataDict, filepath):

        workbook = xlsxwriter.Workbook(filepath)
        self.workbook = workbook
        for key in dataDict.keys():
            DF = dataDict[key]
            sheetname = key
            if key in dataSet.continuousVar or key in dataSet.discreteVar:
                self.plot_conti_distribution_(DF, sheetname)
            elif key in dataSet.categoricalVar:
                self.plot_categorical_distribution_(DF, sheetname)
            else:
                logging.error("错误的变量名 :"+key)
                pass

        workbook.close()


    def twofeatureCompare_workbook(self, DF, filepath):

        workbook = xlsxwriter.Workbook(filepath)
        self.workbook = workbook
        sheetname = DF.columns[0] + "_" + DF.columns[1]
        sheetname = sheetname[0:31]
        self.plot_twofeatures_(DF, sheetname)


        workbook.close()




    def plot_conti_distribution_(self, DF, sheetname):
        '''DF有两列，第一列是连续自变量，第二列是分组变量'''

        varDF = DF
        var = varDF.columns[0]
        bins = 20
        yname = dataSet.dependentVar
        varRange = varDF[var].max() - varDF[var].min()
        width = varRange / bins
        binsList = [varDF[var].min()]
        temp_ = [binsList.append(varDF[var].min() + (i+1)*width) for i in range(bins)]

        binsList = [round(x, 3) for x in binsList]

        frequency = pd.cut(varDF[var], bins=bins, labels=binsList[1:])
        frequency = pd.concat([frequency, varDF[yname]], axis=1)
        groupVacc = frequency[frequency[yname] == "vacc"][var]
        groupNO_vacc = frequency[frequency[yname] == "no_vacc"][var]

        groupVacc_freq = groupVacc.value_counts() / groupVacc.shape[0]
        groupVacc_freq = groupVacc_freq.reset_index().sort_values(by="index")[var]
        groupNO_vacc_freq = groupNO_vacc.value_counts() / groupNO_vacc.shape[0]
        groupNO_vacc_freq = groupNO_vacc_freq.reset_index().sort_values(by="index")[var]
        index = binsList[1:]

        yinterval = round(max(max(groupVacc_freq), max(groupNO_vacc_freq)) / 5 + 0.02, 3)
        workbook = self.workbook

        worksheet = workbook.add_worksheet(sheetname)

        #  hist chart
        chart1 = workbook.add_chart({"type": "column",})
        chart1.set_chartarea({
            'border': {'none': True},
        })
        chart1.set_y_axis({
            'line': {'none': True},
            'major_unit': yinterval,
            'min': 0,
        })
        chart1.set_plotarea({
            'layout': {
                'x': 0.15,
                'y': 0.25,
                'width': 0.65,
                'height': 0.50,
            }
        })
        worksheet.write_column("A1", index)
        worksheet.write_column("B1", groupVacc_freq)
        worksheet.write_column("C1", groupNO_vacc_freq)

        chart1.add_series({
            "name": "vacc",
            "categories": "="+sheetname+"!$A$1:$A$" + str(bins),
            "values": "="+sheetname+"!$B$1:$B$" + str(bins),
            "overlap": 100,
            "gap": 0,
            "fill": {
                "color": "red",
                "transparency": 70,
            },
        })

        chart1.add_series({
            "name": "no_vacc",
            "categories": "="+sheetname+"!$A$1:$A$" + str(bins),
            "values": "="+sheetname+"!$C$1:$C$" + str(bins),
            "fill": {
                "color": "blue",
                "transparency": 70,
            },
        })





        # trend chart
        chart = workbook.add_chart({"type": "line",  })
        chart.set_chartarea({
            'border': {'none': True},
        })
        chart.set_y_axis({
            'line': {'none': True},
            'major_unit': yinterval,
        })
        chart.set_plotarea({
            'layout': {
                'x': 0.15,
                'y': 0.25,
                'width': 0.65,
                'height': 0.50,
            }
        })
        worksheet.write_column("A1", index)
        worksheet.write_column("B1", groupVacc_freq)
        worksheet.write_column("C1", groupNO_vacc_freq)

        chart.add_series({
            "name": "vacc",
            "categories": "=" + sheetname + "!$A$1:$A$" + str(bins),
            "values": "=" + sheetname + "!$B$1:$B$" + str(bins),
            'line':   {'width': 1, "color": "red",},
            'smooth': True,
            "fill": {
                "color": "red",
                "transparency": 70,
            },
        })

        chart.add_series({
            "name": "no_vacc",
            "categories": "=" + sheetname + "!$A$1:$A$" + str(bins),
            "values": "=" + sheetname + "!$C$1:$C$" + str(bins),
            'line':   {'width': 1, "color": "blue",},
            'smooth': True,
            "fill": {
                "color": "blue",
                "transparency": 70,
            },
        })
        chart.combine(chart1)
        title = var
        chart.set_title({
            'name': title,
            'overlay': True,
            'layout': {
                'x': 0.05,
                'y': 0.05,
            },
            "name_font": {"size": 15}
        })
        chart.set_size({'width': 450, 'height': 350})
        worksheet.insert_chart("F1", chart)



        self.workbook = workbook


    def plot_categorical_distribution_(self, DF, sheetname):

        varDF = DF
        varDF = varDF.dropna(axis=0)
        var = sheetname
        yname = dataSet.dependentVar

        groupVacc = varDF[varDF[yname] == "vacc"][var]
        groupNO_vacc = varDF[varDF[yname] == "no_vacc"][var]

        groupVacc_freq = groupVacc.value_counts() / groupVacc.shape[0]

        groupNO_vacc_freq = groupNO_vacc.value_counts() / groupNO_vacc.shape[0]
        yinterval = round(max(max(groupVacc_freq), max(groupNO_vacc_freq)) / 5 + 0.02, 3)
        index = groupVacc_freq.index.to_list()
        bins = index.__len__()


        workbook = self.workbook
        sheetname = sheetname[0:31]
        worksheet = workbook.add_worksheet(sheetname)

        # hist plot
        chart1 = workbook.add_chart({"type": "column", })
        chart1.set_chartarea({
            'border': {'none': True},
        })
        chart1.set_y_axis({
            'line': {'none': True},
            'major_unit': yinterval,
        })
        chart1.set_plotarea({
            'layout': {
                'x': 0.15,
                'y': 0.25,
                'width': 0.65,
                'height': 0.50,
            }
        })

        worksheet.write_column("A1", index)
        worksheet.write_column("B1", groupVacc_freq)
        worksheet.write_column("C1", groupNO_vacc_freq)

        chart1.add_series({
            "name": "vacc",
            "categories": "='"+sheetname + "'!$A$1:$A$" + str(bins),
            "values": "='"+sheetname + "'!$B$1:$B$" + str(bins),
            "overlap": 100,
            "gap": 0,
            "fill": {
                "color": "red",
                "transparency": 70,
            },
        })
        chart1.add_series({
            "name": "no_vacc",
            "categories": "='"+sheetname + "'!$A$1:$A$" + str(bins),
            "values": "='"+sheetname + "'!$C$1:$C$" + str(bins),
            "fill": {
                "color": "blue",
                "transparency": 70,
            },
        })



        # trend plot


        title = var
        chart1.set_title({
            'name': title,
            'overlay': True,
            'layout': {
                'x': 0.05,
                'y': 0.05,
            },
            "name_font": {"size": 15}
        })
        chart1.set_size({'width': 450, 'height': 350})
        worksheet.insert_chart("F1", chart1)

        self.workbook = workbook


    def plot_twofeatures_(self, DF, sheetname):
        1
        var0 = DF.columns[0]     # x
        var1 = DF.columns[1]     # y
        DF[[var0, var1]] = DF[[var0, var1]].fillna(DF[[var0, var1]].median(), axis=0)

        DF = pd.concat([DF[DF[dataSet.dependentVar] == "no_vacc"].sample(10000), DF[DF[dataSet.dependentVar] == "vacc"].sample(10000)], axis=0)

        if var1 in dataSet.categoricalVar:
            yinterval = None
        else:
            yinterval = round(max(DF[var1].max(), DF[var1].max() / 5 + 0.02), 3)
        if var0 in dataSet.categoricalVar:
            1
            Xinterval = DF[var0].value_counts(dropna=False).index.to_list()

            Xinterval = pd.DataFrame(index=Xinterval, data=[i for i in range(0, len(Xinterval))])
            DF[var0] = DF[var0].map(lambda x: Xinterval.loc[x].values[0])
        else:
            Xinterval = round(max(DF[var0].max(), DF[var0].max() / 5 + 0.02), 3)

        DFvacc = DF[DF[dataSet.dependentVar] == "vacc"]
        DFno_vacc = DF[DF[dataSet.dependentVar] == "no_vacc"]

        workbook = self.workbook
        chart = workbook.add_chart({"type": "scatter"})
        chart.set_chartarea({
            'border': {'none': True},
        })
        chart.set_y_axis({
            'line': {'none': True},
            # 'major_unit': None,
        })
        # chart.set_x_axis({
        #     'major_unit': Xinterval,
        # })
        chart.set_plotarea({
            'layout': {
                'x': 0.15,
                'y': 0.25,
                'width': 0.65,
                'height': 0.5,
            }
        })

        workbook.nan_inf_to_errors = True
        sheetname = sheetname[0:31]
        worksheet = workbook.add_worksheet(sheetname)
        worksheet.write_column("A1", DFvacc[var0])    # vacc x
        worksheet.write_column("B1", DFvacc[var1])    # vacc y
        binsx = DFvacc.shape[0]
        worksheet.write_column("C1", DFno_vacc[var0])
        worksheet.write_column("D1", DFno_vacc[var1])
        binsy = DFno_vacc.shape[0]
        if var0 in dataSet.categoricalVar:
            worksheet.write_column("N1", Xinterval.index)
            worksheet.write_column("O1", Xinterval.iloc[:, 0])

        chart.add_series({
            "name": "vacc",
            "categories": "='" + sheetname + "'!$A$1:$A$" + str(binsx),
            "values": "='" + sheetname + "'!$B$1:$B$" + str(binsx),
            "marker": {
                'type': 'square',
                'size': 2,
                'border': {'none': True},
                "fill": {
                    "color": "blue",
                    "transparency": 60,
                },
            }
        })

        chart.add_series({
            "name": "no_vacc",
            "categories": "='" + sheetname + "'!$C$1:$C$" + str(binsy),
            "values": "='" + sheetname + "'!$D$1:$D$" + str(binsy),
            "marker": {
                'type': 'square',
                'size': 2,
                'border': {'none': True},
                "fill": {
                    "color": "red",
                    "transparency": 60,
                },
            }
        })
        title = var0 + "(x) & " + var1 + "(y)"
        chart.set_title({
            'name': title,
            'overlay': True,
            'layout': {
                'x': 0.05,
                'y': 0.05,
            },
            "name_font": {"size": 15}
        })
        chart.set_size({'width': 450, 'height': 350})
        worksheet.insert_chart("F1", chart)
        self.workbook = workbook
