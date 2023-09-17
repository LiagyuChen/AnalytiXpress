import streamlit as st
import pandas as pd

st.set_page_config(page_title = "AnalytiXpress", page_icon = "./assets/logo.png")

st.markdown("# Data Lab")
st.sidebar.header("Data Lab")


'''
Dataset Operations by Pandas library
'''
# Inspection Options for the whole dataset
def Inspection(df):
    # Sidebar for checking dataset details
    st.sidebar.header("Dataset Inspection")

    if st.sidebar.checkbox("Display Statistical Information"):
        st.write(df.describe())

    if st.sidebar.checkbox("Display Data Types"):
        st.write(df.dtypes)

# 7 Operation types for user to select
def MissingData(tool, columns):    
    global df
    match tool:
        case "Drop Missing Values":
            st.markdown("### Drop Missing Values")
            column = st.selectbox("Select column: ", columns, index = 0)
            if column == "All Columns":
                df = df.dropna()
                st.write("Result: ")
                st.write(df)
                st.write("Notice: all the changes are auto-saved!")
            elif column == "None":
                st.write("Your need to select the column(s) !")
            else: 
                df = df[column].dropna()
                st.write("Result: ")
                st.write(df)
                st.write("Notice: all the changes are auto-saved!")
            
        case "Fill Missing Values":
            st.markdown("### Fill Missing Values")
            column = st.selectbox("Select column: ", columns, index = 0)
            fill_value = st.text_input("Fill Missing values with:")
            if column == "All Columns":
                df = df.fillna(fill_value)
                st.write("Result: ")
                st.write(df)
                st.write("Notice: all the changes are auto-saved!")
            elif column == "None":
                st.write("Your need to select the column(s) !")
            else: 
                df = df[column].fillna(fill_value)
                st.write("Result: ")
                st.write(df)
                st.write("Notice: all the changes are auto-saved!")
            
        case "Missing Value Columns":
            st.markdown("### Missing Value Columns")
            column = st.selectbox("Select column: ", columns, index = 0)
            st.write("Result: ")
            if column == "All Columns":
                st.write(df.isna())
            elif column == "None":
                st.write("Your need to select the column(s) !")
            else: 
                st.write(df[column].isna())
            
        case "Non-Missing Value Columns":
            st.markdown("### Non-Missing Value Columns")
            column = st.selectbox("Select column: ", columns, index = 0)
            st.write("Result: ")
            if column == "All Columns":
                st.write(df.notna())
            elif column == "None":
                st.write("Your need to select the column(s) !")
            else: 
                st.write(df[column].notna())
                
        case _:
            st.markdown("#### Please select an operation tool!")
            
def Transformation(tool, columns):
    global df
    match tool:
        case "Rename Columns":
            st.markdown("### Rename Columns")
            column = st.selectbox("Select column: ", columns, index = 0)
            newName = st.text_input("New Column Name:")
            if column == "All Columns":
                st.write("Notice: can only rename one column at one time!")
            elif column == "None":
                st.write("Your need to select the column(s) !")
            else: 
                df = df.rename(columns = {column: newName})
                st.write("Result: ")
                st.write(df)
                st.write("Notice: all the changes are auto-saved!")
            
        case "Set Index":
            st.markdown("### Set Index")
            column = st.selectbox("Select column: ", columns, index = 0)
            if column == "All Columns":
                st.write("Notice: can only rename one column at one time!")
            elif column == "None":
                st.write("Your need to select the column(s) !")
            else: 
                df = df.set_index(column)
                st.write("Result: ")
                st.write(df)
                st.write("Notice: all the changes are auto-saved!")
            
        case "Reset Index":
            st.markdown("### Reset Index")
            df = df.reset_index()
            st.write("Result: ")
            st.write(df)
            st.write("Notice: all the changes are auto-saved!")
                   
        case "Transform Data Type":
            st.markdown("### Transform Data Type")
            column = st.selectbox("Select column: ", columns, index=0)
            newType = st.selectbox("Select new type:", ["float", "int", "str"], index = 2)
            if column == "All Columns":
                for col in df.columns:
                    df[col] = df[col].astype(newType)
                st.write("Result: ")
                st.write(df)
                st.write("Notice: all the changes are auto-saved!")
            elif column == "None":
                st.write("Your need to select the column(s) !")
            else: 
                df[column] = df[column].astype(newType)
                st.write("Result: ")
                st.write(df)
                st.write("Notice: all the changes are auto-saved!")
            
        case "Melt Columns":
            st.markdown("### Melt Columns")
            idVars = st.multiselect("Select ID variables:", df.columns)
            valueVars = st.multiselect("Select Value Variables:", df.columns)
            if idVars and valueVars:
                df = df.melt(id_vars = idVars, value_vars = valueVars)
                st.write("Result: ")
                st.write(df)
                st.write("Notice: All the changes are auto-saved! The old version dataframe is now abandoned!")
         
        case "Reshape Dataframe through Pivot":
            st.markdown("### Reshape Dataframe through Pivot")
            index = st.selectbox("Select index column:", df.columns, index = 0)
            columns = st.selectbox("Select columns:", df.columns, index = 1)
            values = st.selectbox("Select values:", df.columns, index = 2)
            if index and columns and values:
                df = df.pivot(index = index, columns = columns, values = values)
                st.write(df)
                st.write("Notice: All the changes are auto-saved! The old version dataframe is now abandoned!")
            
        case "Explode List Columns":
            st.markdown("### Explode List Columns")
            column = st.selectbox("Select column: ", columns, index = 0)
            if column == "All Columns":
                st.write("Notice: can only explode one column at one time!")
            elif column == "None":
                st.write("Your need to select the column(s) !")
            else: 
                df = df.explode(column)
                st.write("Result: ")
                st.write(df)
                st.write("Notice: All the changes are auto-saved!")
                
        case "Replace Values":
            st.markdown("### Replace Values")
            oldValue = st.text_input("Replace value:")
            newValue = st.text_input("With value:")
            if oldValue and newValue:
                df = df.replace(oldValue, newValue)
                st.write("Result: ")
                st.write(df)
                st.write("Notice: All the changes are auto-saved!")
                     
        case _:
            st.markdown("#### Please select an operation tool!")
    
def Filtering(tool, columns):
    global df
    match tool:
        case "Duplicated Values":
            st.markdown("### Duplicated Values")
            st.write("Duplicated Rows: ")
            st.write(df[df.duplicated()])
            
        case "Check data is in one column":
            st.markdown("### Check data is in one column")
            column = st.selectbox("Select column: ", columns, index = 0)
            if column == "All Columns":
                valuesCheck = st.text_input("Enter values to check (comma-separated):")
                if valuesCheck:
                    values = [val.strip() for val in valuesCheck.split(",")]
                    st.write("Result: ")
                    st.write(df.isin(values))
                    st.write("Notice: all the changes are auto-saved!")
            elif column == "None":
                st.write("Your need to select the column(s) !")
            else: 
                valuesCheck = st.text_input("Enter values to check (comma-separated):")
                if valuesCheck:
                    values = [val.strip() for val in valuesCheck.split(",")]
                    st.write("Result: ")
                    st.write(df[column].isin(values))
                    st.write("Notice: all the changes are auto-saved!")
                    
        case "Values Between":
            st.markdown("### Values Between")
            column = st.selectbox("Select column: ", columns, index = 0)
            if column == "All Columns":
                st.write("Notice: can only select one column at one time!")
            elif column == "None":
                st.write("Your need to select the column(s) !")
            else: 
                dfDTypes = df.dtypes
                if dfDTypes[column] == "float64" or dfDTypes[column] == "int64":
                    lowerBound = st.number_input("Enter lower bound:")
                    upperBound = st.number_input("Enter upper bound:")
                elif dfDTypes[column] == "datetime64[ns]":
                    lowerBound = pd.to_datetime(st.date_input("Enter lower bound:")) 
                    upperBound = pd.to_datetime(st.date_input("Enter upper bound:"))
                else:
                    lowerBound = st.text_input("Enter lower bound:")
                    upperBound = st.text_input("Enter upper bound:")
                st.write("Satisfactory Data: ")
                st.write(df[df[column].between(lowerBound, upperBound)])
                     
        case _:
            st.markdown("#### Please select an operation tool!")
    
def Aggregation(tool):
    global df
    
    # Get the numberic columns
    dfDTypes = df.dtypes
    numColumns = ["None"]
    for col in df.columns:
        if dfDTypes[col] in ["float64", "int64"]:
            numColumns.append(col)
    
    match tool:
        case "Group by Columns":
            st.markdown("### Group by Columns")
            cols = st.multiselect("Select columns to group by:", df.columns)
            st.write(cols)
            if cols:
                funcCol = st.multiselect("Select Aggregation Function Columns: ", numColumns)
                func = st.selectbox("Select Aggregation Function: ", ["Sum", "Count", "Max", "Min", "Mean"], index = 0)
                match func:
                    case "Sum":
                        df = df.groupby(cols)[funcCol].sum().reset_index()
                    case "Count":
                        df = df.groupby(cols)[funcCol].count().reset_index()
                    case "Max":
                        df = df.groupby(cols)[funcCol].max().reset_index()
                    case "Min":
                        df = df.groupby(cols)[funcCol].min().reset_index()
                    case "Mean":
                        df = df.groupby(cols)[funcCol].mean().reset_index()
                st.write("Result: ")
                st.write(df)
                st.write("Notice: all the changes are auto-saved! The old version dataframe is now abandoned!")
            
        case "Pivot Table":
            st.markdown("### Pivot Table")
            if len(df.columns) < 3:
                st.markdown("Note: This operation require at least 3 columns in your dataset!")
            else:
                indexCol = st.selectbox("Select index column for pivot table:", df.columns, index = 0)
                columnsCol = st.selectbox("Select columns for pivot table:", df.columns, index = 1)
                valuesCol = st.selectbox("Select values for pivot table:", numColumns, index = 2)
                aggFunc = st.selectbox("Select aggregation function:", ["mean", "sum", "count"])
                df = df.pivot_table(index = indexCol, columns = columnsCol, values = valuesCol, aggfunc = aggFunc)
                st.write("Result: ")
                st.write(df)
                st.write("Notice: all the changes are auto-saved! The old version dataframe is now abandoned!")
              
        case "Count":
            st.markdown("### Count")
            column = st.selectbox("Select column: ", numColumns, index = 0)
            if column == "None":
                st.write("Your need to select the column(s) !")
            else: 
                st.write("Result: ")
                st.write(df[column].count())
            
        case "Sum":
            st.markdown("### Sum")
            column = st.selectbox("Select column: ", numColumns, index = 0)
            if column == "None":
                st.write("Your need to select the column(s) !")
            else: 
                st.write("Result: ")
                st.write(df[column].sum())
         
        case "Mean Value":
            st.markdown("### Mean Value")
            column = st.selectbox("Select column: ", numColumns, index = 0)
            if column == "None":
                st.write("Your need to select the column(s) !")
            else: 
                st.write("Result: ")
                st.write(df[column].mean())
            
        case "Median Value":
            st.markdown("### Median Value")
            column = st.selectbox("Select column: ", numColumns, index = 0)
            if column == "None":
                st.write("Your need to select the column(s) !")
            else: 
                st.write("Result: ")
                st.write(df[column].median())
                
        case "Minimum Value":
            st.markdown("### Minimum Value")
            column = st.selectbox("Select column: ", numColumns, index = 0)
            if column == "None":
                st.write("Your need to select the column(s) !")
            else: 
                st.write("Result: ")
                st.write(df[column].min())
                     
        case "Maximum Value":
            st.markdown("### Maximum Value")
            column = st.selectbox("Select column: ", numColumns, index = 0)
            if column == "None":
                st.write("Your need to select the column(s) !")
            else: 
                st.write("Result: ")
                st.write(df[column].max())
        
        case "Standard Deviation":
            st.markdown("### Standard Deviation")
            column = st.selectbox("Select column: ", numColumns, index = 0)
            if column == "None":
                st.write("Your need to select the column(s) !")
            else: 
                st.write("Result: ")
                st.write(df[column].std())
                           
        case _:
            st.markdown("#### Please select an operation tool!")
    
def Sorting(tool, columns):
    global df
    match tool:
        case "Sort Values":
            st.markdown("### Sort Values")
            column = st.selectbox("Select column to sort by: ", columns, index = 0)
            if column == "All Columns":
                st.write("Notice: can only sort by one column ot one time!")
            elif column == "None":
                st.write("Your need to select the column(s) !")
            else: 
                df = df.sort_values(by = column)
                st.write("Result: ")
                st.write(df)
                st.write("Notice: all the changes are auto-saved!")
            
        case "Sort Index":
            st.markdown("### Sort Index")
            df = df.sort_index()
            st.write("Result: ")
            st.write(df)
            st.write("Notice: all the changes are auto-saved!")
                     
        case _:
            st.markdown("#### Please select an operation tool!")
    
def Joining(tool):
    global df
    match tool:
        case "Merge Dataframe":
            st.markdown("### Merge Dataframe")
            file2 = st.file_uploader("Upload another dataset for merging", type=["csv", "xls", "xlsx"])
            if file2:
                if file2.name.endswith('.csv'):
                    df2 = pd.read_csv(file2)
                else:
                    df2 = pd.read_excel(file2)
                columnList = []
                columnList.extend(df.columns.intersection(df2.columns))
                onCol = st.selectbox("Select column to merge on:", columnList, index = 0)
                df = df.merge(df2, on = onCol)
                st.write("Result: ")
                st.write(df)
                st.write("Notice: all the changes are auto-saved!")
            
        case "Join Dataframe":
            st.markdown("### Join Dataframe")
            file2 = st.file_uploader("Upload another dataset for joining", type=["csv", "xls", "xlsx"])
            if file2:
                if file2.name.endswith('.csv'):
                    df2 = pd.read_csv(file2)
                else:
                    df2 = pd.read_excel(file2)
                df = df.join(df2, lsuffix='_left', rsuffix='_right')
                st.write("Result: ")
                st.write(df)
                st.write("Notice: all the changes are auto-saved!")
                     
        case _:
            st.markdown("#### Please select an operation tool!")

def Removal(tool, columns):
    global df
    match tool:
        case "Drop Columns":
            st.markdown("### Drop Columns")
            column = st.selectbox("Select column: ", columns, index = 0)
            if column == "All Columns":
                df = df.drop(columns = df.columns.tolist())
                st.write("Result: ")
                st.write(df)
                st.write("Notice: all the changes are auto-saved!")
            elif column == "None":
                st.write("Your need to select the column(s) !")
            else: 
                df = df.drop(columns = [column])
                st.write("Result: ")
                st.write(df)
                st.write("Notice: all the changes are auto-saved!")
            
        case "Drop Duplicates":
            st.markdown("### Drop Duplicates")
            column = st.selectbox("Select column: ", columns, index = 0)
            if column == "All Columns":
                df = df.drop_duplicates()
                st.write("Result: ")
                st.write(df)
                st.write("Notice: all the changes are auto-saved!")
            elif column == "None":
                st.write("Your need to select the column(s) !")
            else: 
                df = df.drop_duplicates(subset = column)
                st.write("Result: ")
                st.write(df)
                st.write("Notice: all the changes are auto-saved!")
                
        case _:
            st.markdown("#### Please select an operation tool!")


'''
Update the value of df and editedDF in session_state to avoid reloading dataset
'''
uploaded_file = None
if 'df' not in st.session_state or st.session_state.df.empty:
    st.session_state.df = pd.DataFrame({})
    uploaded_file = st.file_uploader("Choose to upload a file", type = ["csv", "xls", "xlsx"])
    st.write("uploaded_file: ", uploaded_file)
    print("uploaded_file: ", uploaded_file)

if uploaded_file is not None:
    filetype = uploaded_file.name.split(".")[-1]
    if filetype == "csv":
        st.session_state.df = pd.read_csv(uploaded_file)
    else:
        st.session_state.df = pd.read_excel(uploaded_file)
    st.session_state.editedDF = st.session_state.df.copy()
   
# Save edits
def saveEdits():
    st.session_state.df = st.session_state.editedDF.copy()

# Short notation
df = st.session_state.df


'''
Page Content
'''
st.markdown("## Daraframe")
st.markdown("### Data Editor")
st.markdown("* You can freely change data contents inside the following dataframe!")
df = st.data_editor(df)
st.markdown("### View Current Data")
st.write(df)
st.write("Dataframe Shape: ", df.shape)

# Get the columns of the dataframe
columns = df.columns.tolist()
columns.append("All Columns")
columns.insert(0, "None")

# Show Inspections options in the sidebar
Inspection(df)

# Choose a operation type
operations = ["Handling Missing Data", "Data Transformation", "Data Filtering", "Data Aggregation", "Data Sorting", "Data Joining", "Data Removal"]
operationType = st.sidebar.selectbox("Choose a type of data operation", operations, on_change = saveEdits)

# Define operation lists for each operation type
missings = ["None", "Drop Missing Values", "Fill Missing Values", "Missing Value Columns", "Non-Missing Value Columns"]
transformations = ["None", "Rename Columns", "Set Index", "Reset Index", "Transform Data Type", "Melt Columns", "Reshape Dataframe through Pivot", "Explode List Columns", "Replace Values"]
filterings = ["None", "Duplicated Values", "Check data is in one column", "Values Between"]
aggregations = ["None", "Group by Columns", "Pivot Table", "Count", "Sum", "Mean Value", "Median Value", "Minimum Value", "Maximum Value", "Standard Deviation"]
sortings = ["None", "Sort Values", "Sort Index"]
joinings = ["None", "Merge Dataframe", "Join Dataframe"]
removals = ["None", "Drop Columns", "Drop Duplicates"]

# Execute the corresponding function
match operationType:
    case "Handling Missing Data":
        operationTool = st.sidebar.selectbox("Choose a operation tool", missings, index = 0, on_change = saveEdits)
        MissingData(operationTool, columns)
    case "Data Transformation":
        operationTool = st.sidebar.selectbox("Choose a operation tool", transformations, index = 0, on_change = saveEdits)
        Transformation(operationTool, columns)
    case "Data Filtering":
        operationTool = st.sidebar.selectbox("Choose a operation tool", filterings, index = 0, on_change = saveEdits)
        Filtering(operationTool, columns)
    case "Data Aggregation":
        operationTool = st.sidebar.selectbox("Choose a operation tool", aggregations, index = 0, on_change = saveEdits)
        Aggregation(operationTool)
    case "Data Sorting":
        operationTool = st.sidebar.selectbox("Choose a operation tool", sortings, index = 0, on_change = saveEdits)
        Sorting(operationTool, columns)
    case "Data Joining":
        operationTool = st.sidebar.selectbox("Choose a operation tool", joinings, index = 0, on_change = saveEdits)
        Joining(operationTool)
    case "Data Removal":
        operationTool = st.sidebar.selectbox("Choose a operation tool", removals, index = 0, on_change = saveEdits)
        Removal(operationTool, columns)
    case _:
        st.write("Sorry, we don't support this operation!")
st.session_state.editedDF = df
    


