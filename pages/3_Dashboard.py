import streamlit as st
import pandas as pd
from pyecharts.charts import *
import pyecharts.options as opts
from pyecharts.globals import ThemeType
from pyecharts.globals import SymbolType

st.set_page_config(page_title = "AnalytiXpress", page_icon = "./assets/logo.png")

st.markdown("# Dashboard Designer")
st.sidebar.header("Dashboard Designer")

'''
Chart Plotting Functions
'''    
def plotBar(chartTheme, themeDict, df, xAxis, yAxisName, yAxis, chartTitle, titleColor, xAxisName, XColor, YColor, labelColor, legendColor, columnColor):
    if chartTheme != None:
        chart = (
            Bar(init_opts = opts.InitOpts(width = "800px", height = "400px", theme = themeDict[chartTheme]))
            .add_xaxis(list(df[xAxis]))
            .add_yaxis(yAxisName, list(df[yAxis]))
        )
    else:
        chart = (
            Bar(init_opts = opts.InitOpts(width = "800px", height = "400px"))
            .add_xaxis(list(df[xAxis]))
            .add_yaxis(yAxisName, list(df[yAxis]), color = columnColor)
        )
    chart.set_global_opts(title_opts = opts.TitleOpts(title = chartTitle, title_textstyle_opts = opts.TextStyleOpts(color = titleColor)), 
                          xaxis_opts = opts.AxisOpts(name = xAxisName, name_textstyle_opts = opts.TextStyleOpts(color = XColor), axislabel_opts = opts.LabelOpts(color = labelColor)),
                          yaxis_opts = opts.AxisOpts(name = yAxisName, name_textstyle_opts = opts.TextStyleOpts(color = YColor), axislabel_opts = opts.LabelOpts(color = labelColor)),
                          legend_opts = opts.LegendOpts(textstyle_opts = opts.TextStyleOpts(color = legendColor)),
                          toolbox_opts = opts.ToolboxOpts(is_show = True)
                        )
    return chart

def plotStackBar(chartTheme, themeDict, df, xAxis, yAxisNames, yAxes, chartTitle, titleColor, xAxisName, XColor, YColor, labelColor, legendColor, columnColors):
    # Pre-processing           
    dataLists = []
    totals = df[yAxes].sum(axis = 1).tolist()
    for yAxis in yAxes:
        dataList = [{"value": value, "percent": value / total} for value, total in zip(df[yAxis], totals)]
        dataLists.append(dataList)
    
    # Plot chart
    if chartTheme != None:
        chart = Bar(init_opts = opts.InitOpts(theme = themeDict[chartTheme], width = "800px", height = "400px"))
    else:
        chart = Bar(init_opts = opts.InitOpts(width = "800px", height = "400px"))
    
    chart.add_xaxis(list(df[xAxis]))
    
    for i in range(len(yAxes)):
        if chartTheme != None:
            chart.add_yaxis(yAxisNames[i], dataLists[i], stack="stack1")
        else:
            chart.add_yaxis(yAxisNames[i], dataLists[i], stack="stack1", color = columnColors[i])
    
    chart.set_global_opts(
        title_opts = opts.TitleOpts(title = chartTitle, title_textstyle_opts = opts.TextStyleOpts(color = titleColor)),
        xaxis_opts = opts.AxisOpts(name = xAxisName, name_textstyle_opts = opts.TextStyleOpts(color = XColor), axislabel_opts = opts.LabelOpts(color = labelColor)),
        yaxis_opts = opts.AxisOpts(axislabel_opts = opts.LabelOpts(color = YColor)),
        legend_opts = opts.LegendOpts(textstyle_opts = opts.TextStyleOpts(color = legendColor)),
        toolbox_opts = opts.ToolboxOpts(is_show = True)
    )
    return chart

def plotLine(chartTheme, themeDict, df, xAxis, yAxisName, yAxis, chartTitle, titleColor, xAxisName, XColor, YColor, labelColor, legendColor, columnColor, isArea):
    if chartTheme != None:
        chart = (
            Line(init_opts = opts.InitOpts(width = "800px", height = "400px", theme = themeDict[chartTheme]))
            .add_xaxis(list(df[xAxis]))    
        )
        if isArea:
            chart.add_yaxis(yAxisName, list(df[yAxis]), is_fill = True, area_opacity = 0.3, is_smooth = True)
        else:
            chart.add_yaxis(yAxisName, list(df[yAxis]))
    else:
        chart = (
            Line(init_opts = opts.InitOpts(width = "800px", height = "400px"))
            .add_xaxis(list(df[xAxis]))
        )
        if isArea:
            chart.add_yaxis(yAxisName, list(df[yAxis]), color = columnColor, is_fill = True, area_opacity = 0.3, is_smooth = True)
        else:
            chart.add_yaxis(yAxisName, list(df[yAxis]), color = columnColor)
    chart.set_global_opts(title_opts = opts.TitleOpts(title = chartTitle, title_textstyle_opts = opts.TextStyleOpts(color = titleColor)),
                          xaxis_opts = opts.AxisOpts(name = xAxisName, name_textstyle_opts = opts.TextStyleOpts(color = XColor), axislabel_opts = opts.LabelOpts(color = labelColor)),
                          yaxis_opts = opts.AxisOpts(name = yAxisName, name_textstyle_opts = opts.TextStyleOpts(color = YColor), axislabel_opts = opts.LabelOpts(color = labelColor)),
                          legend_opts = opts.LegendOpts(textstyle_opts = opts.TextStyleOpts(color = legendColor)),
                          toolbox_opts = opts.ToolboxOpts(is_show = True)
                        )
    return chart

def plotPie(chartTheme, themeDict, df, xAxis, yAxis, chartTitle, titleColor, legendColor, isRing, isRose, columnColors):
    if chartTheme != None:
        chart = (
            Pie(init_opts = opts.InitOpts(width = "800px", height = "400px", theme = themeDict[chartTheme]))
        )
    else:
        chart = (
            Pie(init_opts=opts.InitOpts(width = "800px", height = "400px"))
            .set_colors(columnColors)
        )
    if isRing and not isRose:
        chart.add("", [list(z) for z in zip(df[xAxis], df[yAxis])], radius = [30, 70])
    elif isRose and not isRing:
        chart.add("", [list(z) for z in zip(df[xAxis], df[yAxis])], rosetype = "area")
    elif isRing and isRose:
        chart.add("", [list(z) for z in zip(df[xAxis], df[yAxis])], radius = [30, 70], rosetype = "area")
    else:
        chart.add("", [list(z) for z in zip(df[xAxis], df[yAxis])])
    chart.set_global_opts(title_opts = opts.TitleOpts(title = chartTitle, title_textstyle_opts = opts.TextStyleOpts(color = titleColor)),
                          legend_opts = opts.LegendOpts(textstyle_opts = opts.TextStyleOpts(color = legendColor)),
                          toolbox_opts = opts.ToolboxOpts(is_show = True)
                        )
    return chart

def plotScatter(chartTheme, themeDict, df, xAxis, yAxisName, yAxis, chartTitle, titleColor, xAxisName, XColor, YColor, labelColor, legendColor, columnColor):
    if chartTheme != None:
        chart = (
            Scatter(init_opts = opts.InitOpts(width = "800px", height = "400px", theme = themeDict[chartTheme]))
            .add_xaxis(list(df[xAxis]))
            .add_yaxis(yAxisName, list(df[yAxis]))
        )
    else:
        chart  =  (
            Scatter(init_opts = opts.InitOpts(width = "800px", height = "400px"))
            .add_xaxis(list(df[xAxis]))
            .add_yaxis(yAxisName, list(df[yAxis]), color = columnColor)
        )
    chart.set_global_opts(title_opts = opts.TitleOpts(title = chartTitle, title_textstyle_opts = opts.TextStyleOpts(color = titleColor)),
                          xaxis_opts = opts.AxisOpts(name = xAxisName, name_textstyle_opts = opts.TextStyleOpts(color = XColor), axislabel_opts = opts.LabelOpts(color = labelColor)),
                          yaxis_opts = opts.AxisOpts(name = yAxisName, name_textstyle_opts = opts.TextStyleOpts(color = YColor), axislabel_opts = opts.LabelOpts(color = labelColor)),
                          legend_opts = opts.LegendOpts(textstyle_opts = opts.TextStyleOpts(color = legendColor)),
                          toolbox_opts = opts.ToolboxOpts(is_show = True)
                        )
    return chart

def plotEffectScatter(chartTheme, themeDict, df, xAxis, yAxisName, yAxis, chartTitle, titleColor, xAxisName, XColor, YColor, labelColor, legendColor, columnColor, effectType):
    if chartTheme != None:
        chart = (
            EffectScatter(init_opts = opts.InitOpts(width = "800px", height = "400px", theme = themeDict[chartTheme]))
            .add_xaxis(list(df[xAxis]))
            .add_yaxis(yAxisName, list(df[yAxis]), symbol = effectType)
        )
    else:
        chart  =  (
            EffectScatter(init_opts = opts.InitOpts(width = "800px", height = "400px"))
            .add_xaxis(list(df[xAxis]))
            .add_yaxis(yAxisName, list(df[yAxis]), color = columnColor, symbol = effectType)
        )
    chart.set_global_opts(title_opts = opts.TitleOpts(title = chartTitle, title_textstyle_opts = opts.TextStyleOpts(color = titleColor)),
                          xaxis_opts = opts.AxisOpts(name = xAxisName, name_textstyle_opts = opts.TextStyleOpts(color = XColor), axislabel_opts = opts.LabelOpts(color = labelColor)),
                          yaxis_opts = opts.AxisOpts(name = yAxisName, name_textstyle_opts = opts.TextStyleOpts(color = YColor), axislabel_opts = opts.LabelOpts(color = labelColor)),
                          legend_opts = opts.LegendOpts(textstyle_opts = opts.TextStyleOpts(color = legendColor)),
                          toolbox_opts = opts.ToolboxOpts(is_show = True)
                        )
    return chart

def plotBar3D(chartTheme, themeDict, df, xAxis, yAxis, zAxis, chartTitle, titleColor, xAxisName, yAxisName, zAxisName, XColor, YColor, ZColor):        
    data = []
    for i in range(len(df)):
        for j in range(len(df[yAxis].unique())):
            data.append([i, j, int(df[zAxis].iloc[i])])
    
    if chartTheme != None:
        chart = (
            Bar3D(init_opts = opts.InitOpts(width = "800px", height = "400px", theme = themeDict[chartTheme]))
        )
    else:
        chart  =  (
            Bar3D(init_opts = opts.InitOpts(width = "800px", height = "400px")) 
        )
    chart.add(
        "",
        data,
        xaxis3d_opts = opts.Axis3DOpts(type_ = "category", data = list(df[xAxis]), name = xAxisName, textstyle_opts = opts.TextStyleOpts(color = XColor)),
        yaxis3d_opts = opts.Axis3DOpts(type_ = "category", data = list(df[yAxis]), name = yAxisName, textstyle_opts = opts.TextStyleOpts(color = YColor)),
        zaxis3d_opts = opts.Axis3DOpts(type_ = "value", name = zAxisName, textstyle_opts = opts.TextStyleOpts(color = ZColor)),
        grid3d_opts = opts.Grid3DOpts(width = 100, depth = 100)
    )
    chart.set_global_opts(
        visualmap_opts = opts.VisualMapOpts(max_ = max(df[zAxis])),
        title_opts = opts.TitleOpts(title = chartTitle, title_textstyle_opts = opts.TextStyleOpts(color = titleColor)),
        toolbox_opts = opts.ToolboxOpts(is_show = True)
    )
    return chart

def plotLine3D(chartTheme, themeDict, df, xAxis, yAxis, zAxis, chartTitle, titleColor, xAxisName, yAxisName, zAxisName, XColor, YColor, ZColor):
    data = []
    for i in range(len(df)):
        for j in range(len(df[yAxis].unique())):
            data.append([i, j, int(df[zAxis].iloc[i])])

    if chartTheme != None:
        chart = (
            Line3D(init_opts = opts.InitOpts(width = "800px", height = "400px", theme = themeDict[chartTheme]))
        )
    else:
        chart  =  (
            Line3D(init_opts = opts.InitOpts(width = "800px", height = "400px"))
        )
    chart.add(
        "",
        data,
        xaxis3d_opts = opts.Axis3DOpts(type_ = "category", data = list(df[xAxis]), name = xAxisName, textstyle_opts = opts.TextStyleOpts(color = XColor)),
        yaxis3d_opts = opts.Axis3DOpts(type_ = "category", data = list(df[yAxis]), name = yAxisName, textstyle_opts = opts.TextStyleOpts(color = YColor)),
        zaxis3d_opts = opts.Axis3DOpts(type_ = "value", name = zAxisName, textstyle_opts = opts.TextStyleOpts(color = ZColor)),
        grid3d_opts = opts.Grid3DOpts(width = 100, depth = 100)
    )
    chart.set_global_opts(
        visualmap_opts = opts.VisualMapOpts(max_ = max([i[2] for i in data])),
        title_opts = opts.TitleOpts(title = chartTitle, title_textstyle_opts = opts.TextStyleOpts(color = titleColor)),
        toolbox_opts = opts.ToolboxOpts(is_show = True)
    )
    return chart

def plotScatter3D(chartTheme, themeDict, df, xAxis, yAxis, zAxis, chartTitle, titleColor, xAxisName, yAxisName, zAxisName, XColor, YColor, ZColor):
    data = []
    for i in range(len(df)):
        data.append([int(df[xAxis].iloc[i]), int(df[yAxis].iloc[i]), int(df[zAxis].iloc[i])])

    if chartTheme != None:
        chart = (
            Scatter3D(init_opts = opts.InitOpts(width = "800px", height = "400px", theme = themeDict[chartTheme]))
        )
    else:
        chart = (
            Scatter3D(init_opts = opts.InitOpts(width = "800px", height = "400px"))
        )
    chart.add(
        "",
        data,
        xaxis3d_opts = opts.Axis3DOpts(type_ = "value", name = xAxisName, textstyle_opts = opts.TextStyleOpts(color = XColor)),
        yaxis3d_opts = opts.Axis3DOpts(type_ = "value", name = yAxisName, textstyle_opts = opts.TextStyleOpts(color = YColor)),
        zaxis3d_opts = opts.Axis3DOpts(type_ = "value", name = zAxisName, textstyle_opts = opts.TextStyleOpts(color = ZColor)),
        grid3d_opts = opts.Grid3DOpts(width = 100, depth = 100)
    )
    chart.set_global_opts(
        visualmap_opts = opts.VisualMapOpts(max_ = max(df[zAxis])),
        title_opts = opts.TitleOpts(title = chartTitle, title_textstyle_opts = opts.TextStyleOpts(color = titleColor)),
        toolbox_opts = opts.ToolboxOpts(is_show = True)
    )
    return chart

def plotFunnel(chartTheme, themeDict, df, xAxis, yAxis, chartTitle, titleColor, legendColor, columnColors):
    if chartTheme != None:
        chart = (
            Funnel(init_opts = opts.InitOpts(width = "800px", height = "400px", theme = themeDict[chartTheme]))
        )
    else:
        chart = (
            Funnel(init_opts = opts.InitOpts(width = "800px", height = "400px"))
            .set_colors(columnColors)
        )
    chart.add("funnel", [list(z) for z in zip(list(df[xAxis]), list(df[yAxis]))])
    chart.set_global_opts(title_opts = opts.TitleOpts(title=chartTitle, title_textstyle_opts = opts.TextStyleOpts(color = titleColor)),
                          legend_opts = opts.LegendOpts(textstyle_opts = opts.TextStyleOpts(color = legendColor)),
                          toolbox_opts = opts.ToolboxOpts(is_show = True))
    return chart

def plotGauge(chartTheme, themeDict, key, value, chartTitle, titleColor, legendColor, labelColor, columnColor):
    if chartTheme != None:
        chart = (
            Gauge(init_opts = opts.InitOpts(width = "800px", height = "400px", theme = themeDict[chartTheme]))
        )
    else:
        chart = (
            Gauge(init_opts = opts.InitOpts(width = "800px", height = "400px"))
            .set_colors(columnColor)
        )
    chart.add("", [(key, value)], title_label_opts = opts.LabelOpts(color = labelColor), detail_label_opts = opts.LabelOpts(color = legendColor))
    chart.set_global_opts(title_opts = opts.TitleOpts(title = chartTitle, title_textstyle_opts = opts.TextStyleOpts(color = titleColor)),
                          legend_opts = opts.LegendOpts(is_show = False),
                          toolbox_opts = opts.ToolboxOpts(is_show = True)
                        )
    return chart

def plotHeatmap(chartTheme, themeDict, df, xAxis, yAxis, values, chartTitle, titleColor):
    xAxisData = df[xAxis].unique().tolist()
    yAxisData = df[yAxis].unique().tolist()
    data = []
    for _, row in df.iterrows():
        xIndex = xAxisData.index(row[xAxis])
        yIndex = yAxisData.index(row[yAxis])
        value = row[values]
        data.append([xIndex, yIndex, value])

    if chartTheme != None:
        chart = (
            HeatMap(init_opts = opts.InitOpts(width = "800px", height = "400px", theme = themeDict[chartTheme]))
        )
    else:
        chart = (
            HeatMap(init_opts = opts.InitOpts(width = "800px", height = "400px"))
        )
    chart.add_xaxis(xAxisData)
    chart.add_yaxis("heatmap", yAxisData, data)
    chart.set_global_opts(title_opts = opts.TitleOpts(title = chartTitle, title_textstyle_opts = opts.TextStyleOpts(color = titleColor)),
                          toolbox_opts = opts.ToolboxOpts(is_show = True), visualmap_opts = opts.VisualMapOpts())
    return chart

def plotCandlestick(chartTheme, themeDict, df, xAxis, yAxisName, yAxis, chartTitle, titleColor, xAxisName, XColor, YColor, labelColor, legendColor):
    if chartTheme != None:
        chart = (
            Kline(init_opts = opts.InitOpts(width = "800px", height = "400px", theme = themeDict[chartTheme]))
        )
    else:
        chart = (
            Kline(init_opts = opts.InitOpts(width = "800px", height = "400px"))
        )
    chart.add_xaxis(list(df[xAxis]))
    chart.add_yaxis("candlestick", list(df[yAxis]))
    chart.set_global_opts(title_opts = opts.TitleOpts(title = chartTitle, title_textstyle_opts = opts.TextStyleOpts(color=titleColor)),
                          xaxis_opts = opts.AxisOpts(name = xAxisName, name_textstyle_opts = opts.TextStyleOpts(color = XColor), axislabel_opts = opts.LabelOpts(color = labelColor)),
                          yaxis_opts = opts.AxisOpts(name = yAxisName, name_textstyle_opts = opts.TextStyleOpts(color = YColor), axislabel_opts = opts.LabelOpts(color = labelColor)),
                          legend_opts = opts.LegendOpts(textstyle_opts = opts.TextStyleOpts(color = legendColor)),
                          toolbox_opts=opts.ToolboxOpts(is_show = True))
    return chart

def plotLiquid(chartTheme, themeDict, value, chartTitle, titleColor, valueColor):
    if chartTheme != None:
        chart = (
            Liquid(init_opts=opts.InitOpts(width="800px", height="400px", theme=themeDict[chartTheme]))
            .add("liquid", [value / 100])
        )
    else:
        chart = (
            Liquid(init_opts=opts.InitOpts(width="800px", height="400px"))
            .add("liquid", [value / 100], color = valueColor)
        )
    chart.set_global_opts(title_opts = opts.TitleOpts(title = chartTitle, title_textstyle_opts = opts.TextStyleOpts(color = titleColor)),
                          legend_opts = opts.LegendOpts(is_show = False),
                          toolbox_opts = opts.ToolboxOpts(is_show = True))
    return chart

def plotMap(chartTheme, themeDict, xAxis, yAxis, chartTitle, titleColor, xAxisName, yAxisName, XColor, YColor, labelColor, legendColor):
    if chartTheme != None:
        chart = (
            Map(init_opts=opts.InitOpts(width="800px", height="400px", theme=themeDict[chartTheme]))
        )
    else:
        chart = (
            Map(init_opts=opts.InitOpts(width="800px", height="400px"))
        )
    chart.add("map", [list(z) for z in zip(list(df[xAxis]), list(df[yAxis]))], "world")
    chart.set_global_opts(title_opts=opts.TitleOpts(title=chartTitle, title_textstyle_opts=opts.TextStyleOpts(color=titleColor)),
                          visualmap_opts=opts.VisualMapOpts(max_=df[yAxis].max()),
                          xaxis_opts=opts.AxisOpts(name=xAxisName, name_textstyle_opts=opts.TextStyleOpts(color=XColor), axislabel_opts=opts.LabelOpts(color=labelColor)),
                          yaxis_opts=opts.AxisOpts(name=yAxisName, name_textstyle_opts=opts.TextStyleOpts(color=YColor), axislabel_opts=opts.LabelOpts(color=labelColor)),
                          legend_opts=opts.LegendOpts(textstyle_opts=opts.TextStyleOpts(color=legendColor)),
                          toolbox_opts=opts.ToolboxOpts(is_show=True)
                        )
    return chart

def plotRadar(chartTheme, themeDict, df, xAxis, yAxis, chartTitle, titleColor, labelColor, legendColor, columnColor, maxValue):
    indicators = [opts.RadarIndicatorItem(name=key, max_=maxValue) for key in df[xAxis].tolist()]
    data = [df[yAxis].tolist()]

    if chartTheme in themeDict:
        chart = (
            Radar(init_opts = opts.InitOpts(width = "800px", height = "400px", theme = themeDict[chartTheme]))
            .add(
                series_name = chartTitle,
                data = data
            )
        )
    else:
        chart = (
            Radar(init_opts = opts.InitOpts(width = "800px", height = "400px"))
            .add(
                series_name = chartTitle,
                data = data,
                linestyle_opts = opts.LineStyleOpts(color = columnColor)
            )
        )
    chart.add_schema(
        schema = indicators,
        splitarea_opt = opts.SplitAreaOpts(
            is_show = True, areastyle_opts = opts.AreaStyleOpts(opacity = 1)
        ),
        textstyle_opts = opts.TextStyleOpts(color = labelColor)
    )
    chart.set_series_opts(label_opts = opts.LabelOpts(is_show = False))
    chart.set_global_opts(
        title_opts = opts.TitleOpts(title = chartTitle, title_textstyle_opts = opts.TextStyleOpts(color = titleColor)),
        legend_opts = opts.LegendOpts(textstyle_opts = opts.TextStyleOpts(color = legendColor)),
        toolbox_opts = opts.ToolboxOpts(is_show = True)
    )
    return chart

def plotWordCloud(chartTheme, themeDict, df, xAxis, yAxisName, yAxis, chartTitle, titleColor, legendColor):
    data = [(word, freq) for word, freq in zip(df[xAxis], df[yAxis])]

    if chartTheme != None:
        chart = (
            WordCloud(init_opts = opts.InitOpts(width = "800px", height = "400px", theme = themeDict[chartTheme]))
        )
    else:
        chart = (
            WordCloud(init_opts = opts.InitOpts(width = "800px", height = "400px"))
        )
    chart.add(series_name = yAxisName, data_pair = data, word_size_range = [20, 100])
    chart.set_global_opts(title_opts = opts.TitleOpts(title = chartTitle, title_textstyle_opts = opts.TextStyleOpts(color = titleColor)),
                          legend_opts = opts.LegendOpts(textstyle_opts = opts.TextStyleOpts(color = legendColor)),
                          toolbox_opts = opts.ToolboxOpts(is_show = True)
                        )
    return chart

def plotTimelineBar(chartTheme, themeDict, df, xAxis, yAxisName, yAxis, chartTitle, titleColor, xAxisName, XColor, YColor, labelColor, legendColor, columnColor, times):
    timeline = Timeline(init_opts = opts.InitOpts(width="800px", height="370px"))
    
    for time in list(df[times]):
        dfTime = df[df[times] == time]
        if chartTheme != None:
            chart = (
                Bar(init_opts = opts.InitOpts(theme = themeDict[chartTheme]))
                .add_xaxis(list(dfTime[xAxis]))
                .add_yaxis(yAxisName, list(dfTime[yAxis]))
            )
        else:
            chart = (
                Bar()
                .add_xaxis(list(dfTime[xAxis]))
                .add_yaxis(yAxisName, list(dfTime[yAxis]), color = columnColor)
            )
        chart.set_global_opts(title_opts = opts.TitleOpts(title = chartTitle, title_textstyle_opts = opts.TextStyleOpts(color = titleColor)), 
                              xaxis_opts = opts.AxisOpts(name = xAxisName, name_textstyle_opts = opts.TextStyleOpts(color = XColor), axislabel_opts = opts.LabelOpts(color = labelColor)),
                              yaxis_opts = opts.AxisOpts(name = yAxisName, name_textstyle_opts = opts.TextStyleOpts(color = YColor), axislabel_opts = opts.LabelOpts(color = labelColor)),
                              legend_opts = opts.LegendOpts(textstyle_opts = opts.TextStyleOpts(color = legendColor)),
                              toolbox_opts = opts.ToolboxOpts(is_show = True)
                            )
        timeline.add(chart, time)
    
    return timeline

def plotTimelineLine(chartTheme, themeDict, df, xAxis, yAxisName, yAxis, chartTitle, titleColor, xAxisName, XColor, YColor, labelColor, legendColor, columnColor, times):
    timeline = Timeline(init_opts = opts.InitOpts(width="800px", height="370px"))
    
    for time in list(df[times]):
        dfTime = df[df[times] == time]
        if chartTheme != None:
            chart = (
                Line(init_opts = opts.InitOpts(theme = themeDict[chartTheme]))
                .add_xaxis(list(dfTime[xAxis]))
                .add_yaxis(yAxisName, list(dfTime[yAxis]))
            )
        else:
            chart = (
                Line()
                .add_xaxis(list(dfTime[xAxis]))
                .add_yaxis(yAxisName, list(dfTime[yAxis]), color = columnColor)
            )
        chart.set_global_opts(title_opts = opts.TitleOpts(title = chartTitle, title_textstyle_opts = opts.TextStyleOpts(color = titleColor)), 
                              xaxis_opts = opts.AxisOpts(name = xAxisName, name_textstyle_opts = opts.TextStyleOpts(color = XColor), axislabel_opts = opts.LabelOpts(color = labelColor)),
                              yaxis_opts = opts.AxisOpts(name = yAxisName, name_textstyle_opts = opts.TextStyleOpts(color = YColor), axislabel_opts = opts.LabelOpts(color = labelColor)),
                              legend_opts = opts.LegendOpts(textstyle_opts = opts.TextStyleOpts(color = legendColor)),
                              toolbox_opts = opts.ToolboxOpts(is_show = True)
                            )
        timeline.add(chart, time)
    return timeline

def plotTimelinePie(chartTheme, themeDict, df, xAxis, yAxisName, yAxis, chartTitle, titleColor, labelColor, legendColor, times, columnColors):
    timeline = Timeline(init_opts=opts.InitOpts(width="800px", height="370px"))
    
    for time in list(df[times]):
        dfTime = df[df[times] == time]
        if chartTheme != None:
            chart = (
                Pie(init_opts = opts.InitOpts(theme = themeDict[chartTheme]))
            )
        else:
            chart = (
                Pie()
                .set_colors(columnColors)
            )
        chart.add("", [list(z) for z in zip(dfTime[xAxis], dfTime[yAxis])])
        chart.set_global_opts(title_opts = opts.TitleOpts(title = chartTitle, title_textstyle_opts = opts.TextStyleOpts(color = titleColor)), 
                            legend_opts = opts.LegendOpts(textstyle_opts = opts.TextStyleOpts(color = legendColor)),
                            toolbox_opts = opts.ToolboxOpts(is_show = True)
                        )
        chart.set_series_opts(label_opts = opts.LabelOpts(color = labelColor))
        timeline.add(chart, time)
    return timeline

def plotTitle(titleName, titleColor, fontSize, bgColor):
    chart = (
        Pie(init_opts = opts.InitOpts(bg_color = bgColor))
        .set_global_opts(title_opts = opts.TitleOpts(title = titleName, pos_left = 'center', pos_top = 'center',
                                                   title_textstyle_opts = opts.TextStyleOpts(color = titleColor, font_size = fontSize, background_color = bgColor)))
        )
    return chart


# To set chart colors through loop
def getColColors():
    colorn = st.number_input("Number of colors to set")
    columnColors = []
    for i in range(colorn):
        yyColor = st.color_picker("Choose y-axis colors %d"%i, "#ffffff")
        columnColors.append(yyColor)
    return columnColors

    
# Get the data from session_state
if 'df' in st.session_state:
    st.markdown("## View the current dataframe")
    df = st.session_state.df
    st.write(df)
    
    # Get the numberic columns
    dfDTypes = df.dtypes
    numColumns = []
    for col in df.columns:
        if dfDTypes[col] in ["float64", "int64"]:
            numColumns.append(col)
    
    '''
    Chart Configuration Part
    '''    
    # Select total number of charts and which chart to configure
    chartNum = st.sidebar.slider("Select total number of charts for the dashboard", 2, 15)
    currentChart = st.sidebar.slider("Select which chart to configure", 1, chartNum)
    
    # Store the configured charts in seesion state
    if 'configuredCharts' not in st.session_state:
        st.session_state.configuredCharts = {i: None for i in range(1, chartNum + 1)}
    
    # Select the chart type
    chartTypes = ["Bar", "Stack Bar", "Line", "Pie", "Scatter", "Effect Scatter", "Bar 3D", "Line 3D", "Scatter 3D", "Radar", "Candlestick", 
                  "Heatmap", "Map", "World Cloud", "Funnel", "Gauge", "Liquid", "Timeline Bar", "Timeline Line", "Timeline Pie", "Dashboard Title"]
    chartType = st.sidebar.selectbox("Select Chart Type", chartTypes)
        
    # Chart configuration form
    if chartType == "Dashboard Title":
        with st.sidebar.form("title_config_form"):
            titleName = st.text_input("Dashboard Title", "My Dashboard")
            fontSize = st.number_input("Set the font size")
            titleColor = st.color_picker("Choose a title color", "#ffffff")
            bgColor = st.color_picker("Choose a background color", "#ffffff")
            submitButton = st.form_submit_button("Create Title")      
    else:
        with st.sidebar.form("chart_config_form"):
            # Title settings
            chartTitle = st.text_input("Chart Title", "My Chart")
            titleColor = st.color_picker("Choose a title color", "#ffffff")
            
            # X-axis settings
            if chartType != "Gauge":
                if chartType == "Scatter 3D":
                    xAxis = st.selectbox("Select X-axis column", numColumns)
                else:
                    xAxis = st.selectbox("Select X-axis column", df.columns)
                xAxisName = st.text_input("X-axis Name", xAxis)
                XColor = st.color_picker("Choose a X-axis color", "#ffffff")
            
            # Y-axis settings
            if chartType != "Stack Bar" and chartType != "Gauge":
                yAxis = st.selectbox("Select Y-axis column", numColumns)
                yAxisName = st.text_input("Y-axis Name", yAxis)
                YColor = st.color_picker("Choose a Y-axis color", "#ffffff")
            
            # Chart color settings
            chartColorType = st.selectbox("Select Chart Color Type", ["Theme", "Color"])
            chartTheme, columnColor = None, None
            if chartColorType == "Theme":
                Themes = ["Chalk Style", "Dark Style", "Essos Style", "Infographic Style", "Light Style", "Macarons Style", "Purple-Passion Style", "Roma Style", "Romantic Style", "Shine Style", "Vintage Style", "Walden Style", "Westeros Style", "White Style", "Wonderland Style"]
                chartTheme = st.selectbox("Select Chart Type", Themes)
            else:
                columnColor = st.color_picker("Choose a color")
                
            # Chart options for specific chart types
            if chartType == "Stack Bar":
                ys = st.multiselect("Select all the Y-axis columns", df.columns)
                YColor = st.color_picker("Choose a Y-axis color", "#ffffff")
                columnColors = []
                if chartTheme == None:
                    for i in range(len(ys)):
                        ysColor = st.color_picker("Choose y-axis colors %d"%i, "#ffffff")
                        columnColors.append(ysColor)
            elif chartType == "Line":
                isArea = st.checkbox("Fill the lines as area chart")
            elif chartType == "Pie":
                isRing = st.checkbox("Ring-Type Chart")
                isRose = st.checkbox("Rose-Type Chart")
                columnColors = getColColors()
            elif chartType == "Effect Scatter":
                effectTypes = [SymbolType.ARROW, SymbolType.DIAMOND, SymbolType.RECT, SymbolType.ROUND_RECT, SymbolType.TRIANGLE]
                effectType = st.selectbox("Select an effect type", effectTypes)
            elif chartType == "Bar 3D" or chartType == "Line 3D" or chartType == "Scatter 3D":
                zAxis = st.selectbox("Select Z-axis column", numColumns)
                zAxisName = st.text_input("Z-axis Name", zAxis)
                ZColor = st.color_picker("Choose a Z-axis color", "#ffffff")
            elif chartType == "Funnel":
                columnColors = getColColors()
            elif chartType == "Gauge":
                key = st.text_input("Input the metric name", "Percentage")
                value = st.number_input("Input the percentage")
            elif chartType == "Heatmap":
                values = st.selectbox("Select heat values column", df.columns)
            elif chartType == "Liquid":
                value = st.number_input("Input the Value")
                valueColor = st.color_picker("Choose a liquid color", "#ffffff")
            elif chartType == "Radar":
                maxValue = st.number_input("Set the maximum Value")
            elif chartType == "Timeline Bar" or chartType == "Timeline Line" or chartType == "Timeline Pie":
                times = st.selectbox("Select a time column", df.columns)
                if chartType == "Timeline Pie":
                    columnColors = getColColors()
            
            # Set color of the labels and legends
            labelColor = st.color_picker("Choose a label color", "#ffffff")
            legendColor = st.color_picker("Choose a legend color", "#ffffff")    
            
            submitButton = st.form_submit_button("Configure Chart")

    '''
    Chart Plotting Part
    '''    
    # Plot the chart
    if submitButton:
        themeDict = {"Chalk Style": ThemeType.CHALK, "Dark Style": ThemeType.DARK, "Essos Style": ThemeType.ESSOS, "Infographic Style": ThemeType.INFOGRAPHIC, 
                     "Light Style": ThemeType.LIGHT, "Macarons Style": ThemeType.MACARONS, "Purple-Passion Style": ThemeType.PURPLE_PASSION, "Roma Style": ThemeType.ROMA, 
                     "Romantic Style": ThemeType.ROMANTIC, "Shine Style": ThemeType.SHINE, "Vintage Style": ThemeType.VINTAGE, "Walden Style": ThemeType.WALDEN, 
                     "Westeros Style": ThemeType.WESTEROS, "White Style": ThemeType.WHITE, "Wonderland Style": ThemeType.WONDERLAND}
        
        # Create Chart
        try:
            match chartType:
                case "Bar":
                    chart = plotBar(chartTheme, themeDict, df, xAxis, yAxisName, yAxis, chartTitle, titleColor, xAxisName, XColor, YColor, labelColor, legendColor, columnColor)
                case "Stack Bar":
                    chart = plotStackBar(chartTheme, themeDict, df, xAxis, ys, ys, chartTitle, titleColor, xAxisName, XColor, YColor, labelColor, legendColor, columnColors)
                case "Line":
                    chart = plotLine(chartTheme, themeDict, df, xAxis, yAxisName, yAxis, chartTitle, titleColor, xAxisName, XColor, YColor, labelColor, legendColor, columnColor, isArea)
                case "Pie":
                    chart = plotPie(chartTheme, themeDict, df, xAxis, yAxis, chartTitle, titleColor, legendColor, isRing, isRose)
                case "Scatter":
                    chart = plotScatter(chartTheme, themeDict, df, xAxis, yAxisName, yAxis, chartTitle, titleColor, xAxisName, XColor, YColor, labelColor, legendColor, columnColor)
                case "Effect Scatter":
                    chart = plotEffectScatter(chartTheme, themeDict, df, xAxis, yAxisName, yAxis, chartTitle, titleColor, xAxisName, XColor, YColor, labelColor, legendColor, columnColor, effectType)
                case "Bar 3D":
                    chart = plotBar3D(chartTheme, themeDict, df, xAxis, yAxis, zAxis, chartTitle, titleColor, xAxisName, yAxisName, zAxisName, XColor, YColor, ZColor)
                case "Line 3D":
                    chart = plotLine3D(chartTheme, themeDict, df, xAxis, yAxis, zAxis, chartTitle, titleColor, xAxisName, yAxisName, zAxisName, XColor, YColor, ZColor)
                case "Scatter 3D":
                    chart = plotScatter3D(chartTheme, themeDict, df, xAxis, yAxis, zAxis, chartTitle, titleColor, xAxisName, yAxisName, zAxisName, XColor, YColor, ZColor)
                case "Funnel":
                    chart = plotFunnel(chartTheme, themeDict, df, xAxis, yAxis, chartTitle, titleColor, legendColor, columnColors)
                case "Gauge":
                    chart = plotGauge(chartTheme, themeDict, key, value, chartTitle, titleColor, legendColor, labelColor, columnColor)
                case "Heatmap":
                    chart = plotHeatmap(chartTheme, themeDict, df, xAxis, yAxis, values, chartTitle, titleColor)
                case "Candlestick":
                    chart = plotCandlestick(chartTheme, themeDict, df, xAxis, yAxisName, yAxis, chartTitle, titleColor, xAxisName, XColor, YColor, labelColor, legendColor)
                case "Liquid":
                    chart = plotLiquid(chartTheme, themeDict, value, chartTitle, titleColor, valueColor)
                case "Map":
                    chart = plotMap(chartTheme, themeDict, xAxis, yAxis, chartTitle, titleColor, xAxisName, yAxisName, XColor, YColor, labelColor, legendColor)
                case "Radar":
                    chart = plotRadar(chartTheme, themeDict, df, xAxis, yAxis, chartTitle, titleColor, labelColor, legendColor, columnColor, maxValue)
                case "World Cloud":
                    chart = plotWordCloud(chartTheme, themeDict, df, xAxis, yAxisName, yAxis, chartTitle, titleColor, legendColor)
                case "Timeline Bar":
                    chart = plotTimelineBar(chartTheme, themeDict, df, xAxis, yAxisName, yAxis, chartTitle, titleColor, xAxisName, XColor, YColor, labelColor, legendColor, columnColor, times)
                case "Timeline Line":
                    chart = plotTimelineLine(chartTheme, themeDict, df, xAxis, yAxisName, yAxis, chartTitle, titleColor, xAxisName, XColor, YColor, labelColor, legendColor, columnColor, times)
                case "Timeline Pie":
                    chart = plotTimelinePie(chartTheme, themeDict, df, xAxis, yAxisName, yAxis, chartTitle, titleColor, labelColor, legendColor, times, columnColors)
                case "Dashboard Title":
                    chart = plotTitle(titleName, titleColor, fontSize, bgColor)
                    
        #Handle Exceptions
        except Exception as e:
            st.warning(f"An error occurred: {e}")
            st.warning("Please adjust the chart configurations.")
            
        # Store each configured chart
        # st.session_state.configuredCharts.append(chart)
        st.session_state.configuredCharts[currentChart] = chart
        
        st.markdown("## Render the Chart")
        # Convert the chart to HTML string
        charthtml = chart.render_embed()
        # Embed the HTML content in Streamlit
        st.components.v1.html(charthtml, width = 800, height = 400)
        # Download the chart
        chart.render("AnalytiXpress_Dashboard_Chart_%d.html"%(currentChart))
        st.write("Chart Downloaded Successfully!")
      

    '''
    Dashboard Design Part
    '''        
    st.markdown("## Dashboard Design")
    # Check if all charts are created
    emptyChartNum = sum(1 for value in st.session_state.configuredCharts.values() if value is None)
    if emptyChartNum == 0:
        st.sidebar.write("All charts are configured, free to create dashboard!")
        st.markdown("""
                    ### Dashboard Design
                        * Freely drag & drop each chart component to adjust positions and sizes.
                    ### Save and download the dashboard
                        * Click the 'Save Config' button to donwload the config json file.
                        * Move the downloaded 'chart_config.json' file to this streamlit app file path. 
                        * Upload the 'chart_config.json' file to check the json file status.
                        * The final dashboard will be automatically created and downloaded in the same file path on your local device.
                    """)
        
        page = Page(layout = Page.DraggablePageLayout)
        for pageChart in list(st.session_state.configuredCharts.values()):
            page.add(pageChart)
        # Convert the page to HTML string
        dashboard = page.render_embed()
        # Embed the HTML content in Streamlit
        st.components.v1.html(dashboard, width = 1920, height = 1080)
        # Download the dashboard
        page.render("dashboard.html")
        
        # Config the dashboard page
        uploadedJSON = st.file_uploader("Upload the dashboard config json file", type = ["json"])
        if uploadedJSON is not None:
            # Download the final dashboard
            page.save_resize_html("dashboard.html", cfg_file = uploadedJSON.name, dest = "dashboard_resized.html")
        st.markdown("Dashboard Downloaded Successfully! The file name is 'dashboard_adjusted.html'. Open it and view it in your local file folder!")
        
    else:
        st.sidebar.write(f"Please configure the remaining {emptyChartNum} charts.")
        st.write("Note: This section will be actived only if all the charts are configured!")


# Show Warning
else:
    st.warning("Please upload and edit data first!")
