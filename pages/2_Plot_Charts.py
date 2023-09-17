import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title = "AnalytiXpress", page_icon = "./assets/logo.png")

st.markdown("# Data Visualization")
st.sidebar.header("Data Visualization")
st.write("Using the data from 'Data Lab' to make visualization charts!")

def setChartColor(coloredColumn, colorNum, colorType, colorList, params):
    if coloredColumn:
        params["color"] = coloredColumn
        if colorNum != 0:
            if colorType == "Discrete":
                params["color_discrete_sequence"] = colorList
            else:
                params["color_continuous_scale"] = colorList
    return params

def setDiscreteChartColor(coloredColumn, colorNum, colorList, params):
    if coloredColumn:
        params["color"] = coloredColumn
        if colorNum != 0:
            params["color_discrete_sequence"] = colorList
    return params

def colorConfig(columns):
    coloredColumn, colorNum, colorType, colorList = None, None, None, None 
    coloredColumn = st.selectbox("Select column for colored group", columns)
    st.write("* Notice: You have to click the 'Render Chart' button first to active the next step color configuration!")
    if coloredColumn == "None": 
        coloredColumn = None
    else:
        colorNum = st.selectbox("Select the number of colors to set", list(range(11)))
        if colorNum == 0:
            st.write("No color to set!")
        else:
            colorType = st.selectbox("Select the type of colored data", ["Discrete", "Continuous"])
            colorList = []
            for i in range(colorNum):
                colorList.append(st.color_picker("Color %d: " % (i+1)))
    return coloredColumn, colorNum, colorType, colorList
      
def sizeConfig(columns):           
    sizedColumn = st.selectbox("Select column for different sizes", columns)
    if sizedColumn == "None": 
        sizedColumn = None
    return sizedColumn

def hoverDataConfig(columns):
    hoverData = st.multiselect("Hover Data Column", columns)
    if hoverData == ["None"]: 
        hoverData = None
    return hoverData

# Get the data from session_state
if 'df' in st.session_state:
    df = st.session_state.df
    st.write(df)
    
    # Chart Configs
    # Select Chart Type
    chartTypesList = ["Scatter", "Scatter 3D", "Bar", "Horizontal Bar", "Line", "Line 3D", "Pie", "Histogram", "Area", "Box", "Violin", 
                      "Treemap", "Sunburst", "Funnel", "Scatter Polar", "Bar Polar", "Line Polar", "Heatmap", "Density Heatmap",
                      "Gantt", "Sankey", "Candlestick", "Waterfall", "Indicators", "Gauge", "Bullet", "Mapbox Choropleth Map", 
                      "Scatters on Map", "Lines on Map", "Mapbox Density Heatmap"]
    chartType = st.sidebar.selectbox("Select Chart Type", chartTypesList)
    
    # Whether SHow color, size, hover data chart configs
    colorChartList = ["Scatter", "Scatter 3D", "Bar", "Horizontal Bar", "Line", "Line 3D", "Pie", "Histogram", "Area", "Box", "Violin", "Treemap", 
                      "Sunburst", "Funnel", "Scatter Polar", "Bar Polar", "Line Polar", "Gantt", "Mapbox Choropleth Map", "Scatters on Map", "Lines on Map"]
    sizeChartList = ["Scatter", "Scatter 3D", "Scatter Polar", "Scatters on Map"]
    hoverDataChartList = ["Scatter", "Scatter 3D", "Bar", "Horizontal Bar", "Line", "Line 3D", "Pie", "Histogram", "Area", "Box", "Violin", 
                      "Treemap", "Sunburst", "Funnel", "Scatter Polar", "Bar Polar", "Line Polar", "Density Heatmap",
                      "Gantt", "Mapbox Choropleth Map", "Scatters on Map", "Lines on Map", "Mapbox Density Heatmap"]
    showColor, showSize, showHoverData = False, False, False
    if chartType in colorChartList:
        showColor = True
    if chartType in sizeChartList:
        showSize = True
    if chartType in hoverDataChartList:
        showHoverData = True
    
    # The column list for user to select later on
    columns = df.columns.tolist()
    columns.insert(0, "None")
    
    with st.form("chart_config_form"):
        # Required Options
        chartTitle = st.text_input("Chart Title", "My Chart")
        
        if chartType == "Scatter 3D" or chartType == "Line 3D":
            xAxis = st.selectbox("Select X-axis column", df.columns)
            xAxisName = st.text_input("X-axis Name", xAxis)
            yAxis = st.selectbox("Select Y-axis column(s)", df.columns)
            yAxisName = st.text_input("Y-axis Name", yAxis)
            zAxis = st.selectbox("Select Z-axis column", df.columns)
            zAxisName = st.text_input("Z-axis Name", zAxis)
        elif chartType == "Treemap" or chartType == "Sunburst":
            path = st.multiselect("Select hierarchy column (from top to bottom)", df.columns)
            yAxis = st.selectbox("Select Y-axis column(s)", df.columns)
        elif chartType == "Heatmap":
            st.write("* No other chart configs to set, the heatmap has been rendered!")
        elif chartType == "Gantt":
            startColumn = st.selectbox("Select start date column", columns)
            endColumn = st.selectbox("Select end date column", columns)
        elif chartType == "Sankey":
            sourceColumn = st.selectbox("Select source column", columns)
            targetColumn = st.selectbox("Select target column", columns)
            valueColumn = st.selectbox("Select value column", columns)
        elif chartType == "Candlestick":
            xAxis = st.selectbox("Select X-axis column", df.columns)
            openColumn = st.selectbox("Select open column", columns)
            closeColumn = st.selectbox("Select close column", columns)
            highColumn = st.selectbox("Select high column", columns)
            lowColumn = st.selectbox("Select low column", columns)
        elif chartType == "Indicators" or chartType == "Gauge" or chartType == "Bullet":
            valueValue = st.number_input("Indicator Value")
            targetValue = st.number_input("Target Value")
            valueColor = st.color_picker("Value Color")
            if chartType == "Gauge" or chartType == "Bullet":
                xRangeStart = st.number_input("Range Start Value")
                xRangeEnd = st.number_input("Range End Value")
        elif chartType == "Mapbox Choropleth Map":
            locationColumn = st.selectbox("Select location column", columns)
        elif chartType == "Scatters on Map" or chartType == "Lines on Map" or chartType == "Mapbox Density Heatmap":
            latColumn = st.selectbox("Select latitude column", columns)
            lonColumn = st.selectbox("Select longitude column", columns)
        else:   
            xAxis = st.selectbox("Select X-axis column", df.columns)
            xAxisName = st.text_input("X-axis Name", xAxis)
            yAxis = st.selectbox("Select Y-axis column(s)", df.columns)
            yAxisName = st.text_input("Y-axis Name", yAxis)
        
        # Optional Options
        if showColor:
            coloredColumn, colorNum, colorType, colorList = colorConfig(columns)
        if showSize:
            sizedColumn = sizeConfig(columns)
        if showHoverData:
            hoverData = hoverDataConfig(columns)
            
        # Submit the form
        submitButton = st.form_submit_button("Render Chart")
            
    if submitButton:
        try:
            match chartType:
                case "Scatter":
                    # Set the parameters
                    scatterParams = {
                        "x": xAxis,
                        "y": yAxis,
                        "labels": {xAxis: xAxisName, yAxis: yAxisName},
                        "title": chartTitle
                    }
                    scatterParams = setChartColor(coloredColumn, colorNum, colorType, colorList, scatterParams)
                    if sizedColumn:
                        scatterParams["size"] = sizedColumn
                    if hoverData:
                        scatterParams["hover_data"] = hoverData
                    # Create the scatter plot
                    fig = px.scatter(df, **scatterParams)

                case "Scatter 3D":
                    # Set the parameters
                    scatter3DParams = {
                        "x": xAxis,
                        "y": yAxis,
                        "z": zAxis,
                        "labels": {xAxis: xAxisName, yAxis: yAxisName, zAxis: zAxisName},
                        "title": chartTitle
                    }
                    scatter3DParams = setChartColor(coloredColumn, colorNum, colorType, colorList, scatter3DParams)
                    if hoverData:
                        scatter3DParams["hover_data"] = hoverData
                    if sizedColumn:
                        scatter3DParams["size"] = sizedColumn
                    # Create the scatter 3D plot
                    fig = px.scatter_3d(df, **scatter3DParams)
                    
                case "Bar":
                    # Set the parameters
                    barParams = {
                        "x": xAxis,
                        "y": yAxis,
                        "labels": {xAxis: xAxisName, yAxis: yAxisName},
                        "title": chartTitle
                    }
                    barParams = setChartColor(coloredColumn, colorNum, colorType, colorList, barParams)
                    if hoverData:
                        barParams["hover_data"] = hoverData
                    # Create the bar plot
                    fig = px.bar(df, **barParams)
             
                case "Horizontal Bar":
                    # Set the parameters
                    horizontalBarParams = {
                        "x": yAxis,
                        "y": xAxis,
                        "labels": {xAxis: xAxisName, yAxis: yAxisName},
                        "title": chartTitle,
                        "orientation": "h"
                    }
                    horizontalBarParams = setChartColor(coloredColumn, colorNum, colorType, colorList, horizontalBarParams)
                    if hoverData:
                        horizontalBarParams["hover_data"] = hoverData
                    # Create the horizontal bar plot
                    fig = px.bar(df, **horizontalBarParams)

                case "Line":
                    # Set the parameters
                    lineParams = {
                        "x": xAxis,
                        "y": yAxis,
                        "labels": {xAxis: xAxisName, yAxis: yAxisName},
                        "title": chartTitle
                    }
                    lineParams = setDiscreteChartColor(coloredColumn, colorNum, colorList, lineParams)
                    if hoverData:
                        lineParams["hover_data"] = hoverData
                    # Create the line plot
                    fig = px.line(df, **lineParams)

                case "Line 3D":
                    # Set the parameters
                    line3DParams = {
                        "x": xAxis,
                        "y": yAxis,
                        "z": zAxis,
                        "labels": {xAxis: xAxisName, yAxis: yAxisName, zAxis: zAxisName},
                        "title": chartTitle
                    }
                    line3DParams = setDiscreteChartColor(coloredColumn, colorNum, colorList, line3DParams)
                    if hoverData:
                        line3DParams["hover_data"] = hoverData
                    # Create the line 3D plot
                    fig = px.line_3d(df, **line3DParams)

                case "Pie":
                    # Set the parameters
                    pieParams = {
                        "names": xAxis,
                        "values": yAxis,
                        "title": chartTitle
                    }
                    pieParams = setDiscreteChartColor(coloredColumn, colorNum, colorList, pieParams)
                    if hoverData:
                        pieParams["hover_data"] = hoverData
                    # Create the pie plot
                    fig = px.pie(df, **pieParams)

                case "Histogram":
                    # Set the parameters
                    histogramParams = {
                        "x": xAxis,
                        "title": chartTitle
                    }
                    histogramParams = setDiscreteChartColor(coloredColumn, colorNum, colorList, histogramParams)
                    if hoverData:
                        histogramParams["hover_data"] = hoverData
                    # Create the histogram plot
                    fig = px.histogram(df, **histogramParams)

                case "Area":
                    # Set the parameters
                    areaParams = {
                        "x": xAxis,
                        "y": yAxis,
                        "labels": {xAxis: xAxisName, yAxis: yAxisName},
                        "title": chartTitle
                    }
                    areaParams = setDiscreteChartColor(coloredColumn, colorNum, colorList, areaParams)
                    if hoverData:
                        areaParams["hover_data"] = hoverData
                    # Create the area plot
                    fig = px.area(df, **areaParams)

                case "Box":
                    # Set the parameters
                    boxParams = {
                        "x": xAxis,
                        "y": yAxis,
                        "labels": {xAxis: xAxisName, yAxis: yAxisName},
                        "title": chartTitle
                    }
                    boxParams = setDiscreteChartColor(coloredColumn, colorNum, colorList, boxParams)
                    if hoverData:
                        boxParams["hover_data"] = hoverData
                    # Create the box plot
                    fig = px.box(df, **boxParams)
            
                case "Violin":
                    # Set the parameters
                    violinParams = {
                        "x": xAxis,
                        "y": yAxis,
                        "labels": {xAxis: xAxisName, yAxis: yAxisName},
                        "title": chartTitle
                    }
                    violinParams = setDiscreteChartColor(coloredColumn, colorNum, colorList, violinParams)
                    if hoverData:
                        violinParams["hover_data"] = hoverData
                    # Create the violin plot
                    fig = px.violin(df, **violinParams)
            
                case "Treemap":
                    # Set the parameters
                    treemapParams = {
                        "path": path,
                        "values": yAxis,
                        "title": chartTitle
                    }
                    treemapParams = setChartColor(coloredColumn, colorNum, colorType, colorList, treemapParams)
                    if hoverData:
                        treemapParams["hover_data"] = hoverData
                    # Create the treemap plot
                    fig = px.treemap(df, **treemapParams)

                case "Sunburst":
                    # Set the parameters
                    sunburstParams = {
                        "path": path,
                        "values": yAxis,
                        "title": chartTitle
                    }
                    sunburstParams = setChartColor(coloredColumn, colorNum, colorType, colorList, sunburstParams)
                    if hoverData:
                        sunburstParams["hover_data"] = hoverData
                    # Create the sunburst plot
                    fig = px.sunburst(df, **sunburstParams)

                case "Funnel":
                    # Set the parameters
                    funnelParams = {
                        "x": xAxis,
                        "y": yAxis,
                        "title": chartTitle
                    }
                    funnelParams = setDiscreteChartColor(coloredColumn, colorNum, colorList, funnelParams)
                    if hoverData:
                        funnelParams["hover_data"] = hoverData
                    # Create the funnel plot
                    fig = px.funnel(df, **funnelParams)

                case "Scatter Polar":
                    # Set the parameters
                    polarScatterParams = {
                        "r": yAxis,
                        "theta": xAxis,
                        "title": chartTitle
                    }
                    polarScatterParams = setChartColor(coloredColumn, colorNum, colorType, colorList, polarScatterParams)
                    if hoverData:
                        polarScatterParams["hover_data"] = hoverData
                    if sizedColumn:
                        polarScatterParams["size"] = sizedColumn
                    # Create the scatter polar plot
                    fig = px.scatter_polar(df, **polarScatterParams)

                case "Bar Polar":
                    # Set the parameters
                    polarBarParams = {
                        "r": yAxis,
                        "theta": xAxis,
                        "title": chartTitle
                    }
                    polarBarParams = setChartColor(coloredColumn, colorNum, colorType, colorList, polarBarParams)
                    if hoverData:
                        polarBarParams["hover_data"] = hoverData
                    # Create the bar polar plot
                    fig = px.bar_polar(df, **polarBarParams)

                case "Line Polar":
                    # Set the parameters
                    polarParams = {
                        "r": yAxis,
                        "theta": xAxis,
                        "title": chartTitle
                    }
                    polarParams = setDiscreteChartColor(coloredColumn, colorNum, colorList, polarParams)
                    if hoverData:
                        polarParams["hover_data"] = hoverData
                    # Create the line polar plot
                    fig = px.line_polar(df, **polarParams)

                case "Heatmap":
                    # Set the parameters
                    dtypes = df.dtypes
                    numCols = []
                    for i in df.columns:
                        if dtypes[i] == "float64" or dtypes[i] == "int64":
                            numCols.append(i)
                    corr = df[numCols].corr()
                    # Create the heatmap plot
                    fig = plt.figure(figsize = (10, 10))
                    ax = sns.heatmap(corr, xticklabels = corr.columns.values, yticklabels = corr.columns.values)
                    ax.set_title(chartTitle)
                    st.pyplot(fig)

                case "Density Heatmap":
                    # Set the parameters
                    densityHeatmapParams = {
                        "x": xAxis,
                        "y": yAxis,
                        "title": chartTitle
                    }
                    if hoverData:
                        densityHeatmapParams["hover_data"] = hoverData
                    # Create the density heatmap plot
                    fig = px.density_heatmap(df, **densityHeatmapParams)

                case "Gantt":
                    # Set the parameters
                    ganttParams = {
                        "start": startColumn,
                        "end": endColumn,
                        "title": chartTitle
                    }
                    ganttParams = setChartColor(coloredColumn, colorNum, colorType, colorList, ganttParams)
                    if hoverData:
                        ganttParams["hover_data"] = hoverData
                    # Create the gantt plot
                    fig = px.timeline(df, **ganttParams)

                case "Sankey":    
                    # Extract unique nodes and create a mapping
                    uniqueNodes = list(pd.concat([df[sourceColumn], df[targetColumn]]).unique())
                    nodeMap = {node: i for i, node in enumerate(uniqueNodes)}
                    sources = df[sourceColumn].map(nodeMap).tolist()
                    targets = df[targetColumn].map(nodeMap).tolist()

                    # Create the sankey plot
                    fig = go.Figure(data=[go.Sankey(
                        # Set the parameters
                        valueformat = ".0f",
                        valuesuffix = "TWh",
                        node = dict(
                            pad = 15,
                            thickness = 15,
                            line = dict(color = "black", width = 0.5),
                            label = uniqueNodes,
                        ),
                        link = dict(
                            source = sources,
                            target = targets,
                            value = df[valueColumn].tolist(),
                            label = df[valueColumn].tolist(),
                        )
                    )])
                    fig.update_layout(title_text = chartTitle, font_size = 14)

                case "Candlestick":
                    # Create the candlestick plot
                    fig = go.Figure(data=[go.Candlestick(
                        # Set the parameters
                        x = df[xAxis],
                        open = df[openColumn],
                        high = df[highColumn],
                        low = df[lowColumn],
                        close = df[closeColumn]
                    )])
                    fig.update_layout(title = chartTitle)
                    
                case "Waterfall":
                    # Create the waterfall plot
                    fig = go.Figure(go.Waterfall(
                        # Set the parameters
                        name = chartTitle, 
                        orientation = "v",
                        measure = ["relative" for _ in range(df.shape[0])],
                        x = df[xAxis],
                        textposition = "outside",
                        text = df[yAxis],
                        y = df[yAxis],
                        connector = {"line": {"color": "#ffffff"}},
                    ))

                case "Indicators":
                    # Create the indicator plot
                    fig = go.Figure(go.Indicator(
                        # Set the parameters
                        mode = "number+delta",
                        value = valueValue,
                        delta = {'position': "top", 'reference': targetValue},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {'bar': {'color': valueColor}}
                    ))

                case "Gauge":
                    # Create the gauge plot
                    fig = go.Figure(go.Indicator(
                        # Set the parameters
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        value = valueValue,
                        mode = "gauge+number+delta",
                        title = {'text': chartTitle},
                        delta = {'reference': targetValue},
                        gauge = {'axis': {'range': [xRangeStart, xRangeEnd]},
                                 'bar': {'color': valueColor}}
                    ))

                case "Bullet":
                    # Create the bullet plot
                    fig = go.Figure(go.Indicator(
                        # Set the parameters
                        mode = "number+gauge+delta", value = valueValue,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        delta = {'reference': targetValue, 'position': "top"},
                        title = {'text':"<b>%s</b>"%chartTitle, 'font': {"size": 14}},
                        gauge = {
                            'shape': "bullet",
                            'axis': {'range': [xRangeStart, xRangeEnd]},
                            'bar': {'color': valueColor}}
                    ))  

                case "Mapbox Choropleth Map":
                    # Set the parameters
                    choroplethParams = {
                        "geojson": locationColumn,
                        "locations": locationColumn,
                        "title": chartTitle
                    }
                    choroplethParams = setChartColor(coloredColumn, colorNum, colorType, colorList, choroplethParams)
                    if hoverData:
                        choroplethParams["hover_data"] = hoverData
                    # Create the choropleth map plot
                    fig = px.choropleth_mapbox(df, **choroplethParams)

                case "Scatters on Map":
                    # Set the parameters
                    bubbleMapParams = {
                        "lat": latColumn,
                        "lon": lonColumn,
                        "title": chartTitle
                    }
                    bubbleMapParams = setChartColor(coloredColumn, colorNum, colorType, colorList, bubbleMapParams)
                    if sizedColumn:
                        bubbleMapParams["size"] = sizedColumn
                    if hoverData:
                        bubbleMapParams["hover_data"] = hoverData
                    # Create the scatter map plot
                    fig = px.scatter_mapbox(df, **bubbleMapParams)

                case "Lines on Map":
                    # Set the parameters
                    lineMapParams = {
                        "lat": latColumn,
                        "lon": lonColumn,
                        "title": chartTitle
                    }
                    lineMapParams = setDiscreteChartColor(coloredColumn, colorNum, colorList, lineMapParams)
                    if hoverData:
                        lineMapParams["hover_data"] = hoverData
                    # Create the line map plot
                    fig = px.line_mapbox(df, **lineMapParams)

                case "Mapbox Density Heatmap":
                    latColumn = st.selectbox("Select latitude column", columns)
                    lonColumn = st.selectbox("Select longitude column", columns)
                    # Set the parameters
                    heatmapParams = {
                        "lat": latColumn,
                        "lon": lonColumn,
                        "title": chartTitle
                    }
                    if hoverData:
                        heatmapParams["hover_data"] = hoverData
                    # Create the density map plot
                    fig = px.density_mapbox(df, **heatmapParams)

            if chartType != "Heatmap":
                st.plotly_chart(fig)

        #Handle Exceptions
        except Exception as e:
            st.warning(f"An error occurred: {e}")
            st.warning("Please adjust the chart configurations.")

# Show Warning
else:
    st.warning("Please upload and edit data first!")




