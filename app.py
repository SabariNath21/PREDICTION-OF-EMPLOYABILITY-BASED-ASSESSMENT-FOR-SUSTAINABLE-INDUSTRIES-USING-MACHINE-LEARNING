import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
rawdata_1=pd.read_csv("D:\\MY PROJECTS\\FINAL YEAR PROJECT\\PROJECT\\Raw_data\\layoffs.csv")
rawdata_2=pd.read_csv("D:\\MY PROJECTS\\FINAL YEAR PROJECT\\PROJECT\\Raw_data\\Startups1.csv")
rawdata_3=pd.read_csv("D:\\MY PROJECTS\\FINAL YEAR PROJECT\\PROJECT\\Raw_data\\naukri.csv")
data_1 = pd.read_csv("D:\\MY PROJECTS\\FINAL YEAR PROJECT\\PROJECT\\Pre_processed_data\\Cleaned_naukri.csv")
rawdata=pd.read_csv("D:\\MY PROJECTS\\FINAL YEAR PROJECT\\PROJECT\\Raw_data\\emp_attrition.csv")
data = pd.read_csv("D:\\MY PROJECTS\\FINAL YEAR PROJECT\\PROJECT\\Pre_processed_data\\Employee Attrition.csv")
data_numeric= data.drop(['Department','EducationField','Gender','JobRole','MaritalStatus'],axis = 1)
data_numeric_short=data_numeric.drop(['BusinessTravel','attrition encode','gender encode','marital status encode','Human Resources',
                                      'Life Sciences','Marketing','Medical','Other','Technical Degree'],axis = 1)
dataset=pd.DataFrame({'Age':[0.0],'JobLevel':[0.0],'MonthlyIncome':[0.0],'NumCompaniesWorked':[0.0],'StockOptionLevel':[0.0],'TotalWorkingYears':[0.0],'Business Travel encode':[0.0]})

def main():
    st.sidebar.header('Employability Visualization and Prediction!')
    page = st.sidebar.selectbox("What do you want?", ["Dataset","Dashboard","Employability Analysis","Prediction"], index=0, key='page_selector')
    if page == "Dataset":
        st.image('image1.png')
        st.title("Descriptive Analysis")
        st.header("Here you can see both raw dataset and after pre-processing")
        if st.checkbox("Showing Employee Attrition raw dataset?", key='attrition_raw'):
            show_rawdata=st.slider("How much raw data do you want to see?" , 1,100,5, key='attrition_raw_slider')
            st.write(rawdata.sample(show_rawdata))
        if st.checkbox("Show layoff raw dataset?", key='layoff_raw'):
            show_rawdata=st.slider("How much raw data do you want to see?" , 1,100,5, key='layoff_raw_slider')
            st.write(rawdata_1.sample(show_rawdata))
        if st.checkbox("Show Startup raw dataset?", key='startup_raw'):
            show_rawdata=st.slider("How much raw data do you want to see?" , 1,100,5, key='startup_raw_slider')
            st.write(rawdata_2.sample(show_rawdata))
        if st.checkbox("Show Job raw dataset?", key='job_raw'):
            show_rawdata=st.slider("How much raw data do you want to see?" , 1,100,5, key='job_raw_slider')
            st.write(rawdata_3.sample(show_rawdata))
        if st.checkbox("Show Employee Attrition dataset after pre-processing?", key='attrition_processed'):
            show_data=st.slider("How much data do you want to see?" , 1,100,5, key='attrition_processed_slider')
            st.write(data.sample(show_data))
        if st.checkbox("Show Job dataset after pre-processing?", key='job_processed'):
            show_data=st.slider("How much data do you want to see?" , 1,100,5, key='job_processed_slider')
            st.write(data_1.sample(show_data))
        if st.button("Primary Analysis", key='primary_analysis'):
            st.write("Attrition balance:")
            st.write(pd.DataFrame(data['Attrition'].value_counts()))
            proportion = data.Attrition.value_counts()[0] / data.Attrition.value_counts()[1]
            st.write("So, the data proportion is %.2f" %proportion , ": 1")
            st.write("list of global layoff:")
            st.write(pd.DataFrame(rawdata_1['country'].value_counts()))
            st.write("list of Indian startups based on cities:")
            st.write(pd.DataFrame(rawdata_2['City'].value_counts()))

            
    elif page == "Dashboard":
        st.image('image2.png')
        st.title("Data Visualization")
        st.header("Pros and cons of Employability:")
        st.write('Please select at least one visualization.')
        st.write("Positive Impact: Growth of Startup companies:")
        if st.checkbox("Geographic Distribution of Indian Startups"):
            def plot_bar_chart():
                city_counts = rawdata_2['City'].value_counts()
                fig, ax = plt.subplots()
                ax.bar(city_counts.index, city_counts.values)
                ax.set_xticklabels(city_counts.index, rotation=90)
                ax.set_xlabel('City')
                ax.set_ylabel('Number of Startups')
                ax.set_title('Geographic Distribution of Indian Startups')
                st.pyplot(fig)
            plot_bar_chart()
        if st.checkbox("Top 10 Industries by Number of Startups"):
            def plot_top_industries_by_count():
                top_industries = rawdata_2['Industries'].value_counts().nlargest(10)
                fig, ax = plt.subplots(figsize=(10,6))
                sns.barplot(x=top_industries.values, y=top_industries.index, palette="Blues_r")
                ax.set_xlabel("Number of Startups")
                ax.set_ylabel("Industry")
                ax.set_title("Top 10 Industries by Number of Startups")
                st.pyplot(fig)
            plot_top_industries_by_count()
        if st.checkbox("Top 10 Industries by Total Funding Amount"):
            def plot_top_industries_by_funding():
                funding_by_industry = rawdata_2.groupby('Industries')['Funding Amount in $'].sum().nlargest(10)
                fig, ax = plt.subplots(figsize=(10,6))
                sns.barplot(x=funding_by_industry.values, y=funding_by_industry.index, palette="Greens_r")
                ax.set_xlabel("Total Funding Amount ($)")
                ax.set_ylabel("Industry")
                ax.set_title("Top 10 Industries by Total Funding Amount")
                st.pyplot(fig)
            plot_top_industries_by_funding()
        if st.checkbox("Startups According to year:"):
            def plot_total_funding_by_year(df):
                rawdata_2['Starting Year'] = pd.to_datetime(rawdata_2['Starting Year'], format='%Y')
                ts_df = rawdata_2.groupby('Starting Year')['Funding Amount in $'].sum()
                fig, ax = plt.subplots()
                ax.plot(ts_df.index, ts_df.values)
                ax.set_xlabel('Year')
                ax.set_ylabel('Total Funding Amount in $')
                ax.set_title('Startup Investments in India')
                st.pyplot(fig)
            plot_total_funding_by_year(rawdata_2)
        if st.checkbox("correlation heatmap"):
            def display_correlation_heatmap(rawdata_2):
                corr =rawdata_2.corr()
                fig, ax = plt.subplots()
                sns.heatmap(corr, annot=True, cmap='coolwarm')
                st.pyplot(fig)
            display_correlation_heatmap(rawdata_2)
        st.write("Negative Impact: Global layoff companies:")
        df_top_20 = rawdata_1.sort_values(by='total_laid_off', ascending=False).head(20)
        if st.checkbox("Total Laid Off by Company (Top 20)"):
            def plot_bar_chart():
                fig, ax = plt.subplots(figsize=(12, 8))
                df_top_20.plot.bar(x='company', y='total_laid_off', ax=ax, color='steelblue')
                ax.set_title('Total Laid Off by Company (Top 20)', fontsize=14)
                ax.set_xlabel('Company', fontsize=12)
                ax.set_ylabel('Total Laid Off', fontsize=12)
                ax.tick_params(axis='both', labelsize=10)
                st.pyplot(fig)
            plot_bar_chart()
        if st.checkbox("Correlation Chart (Top 20)"):
            def plot_correlation_chart():
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(df_top_20.corr(), annot=True, cmap='coolwarm', ax=ax)
                ax.set_title('Correlation Chart (Top 20)', fontsize=14)
                st.pyplot(fig)
            plot_correlation_chart()
        if st.checkbox("Total Laid Off Over Time (Top 20)"):
            def plot_line_chart():
                fig, ax = plt.subplots(figsize=(12, 8))
                df_top_20.groupby('date').sum().plot.line(y='total_laid_off', ax=ax, color='darkorange')
                ax.set_title('Total Laid Off Over Time (Top 20)', fontsize=14)
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Total Laid Off', fontsize=12)
                ax.tick_params(axis='both', labelsize=10)
                st.pyplot(fig)
            plot_line_chart()
        if st.checkbox("Percentage Laid Off by Company (Top 20)"):
            def plot_horizontal_bar_chart():
                fig, ax = plt.subplots(figsize=(12, 8))
                df_top_20.plot.barh(x='company', y='percentage_laid_off', ax=ax, color='green')
                ax.set_title('Percentage Laid Off by Company (Top 20)', fontsize=14)
                ax.set_xlabel('Percentage Laid Off', fontsize=12)
                ax.set_ylabel('Company', fontsize=12)
                ax.tick_params(axis='both', labelsize=10)
                st.pyplot(fig)
            plot_horizontal_bar_chart()
        st.write('Please select at least one visualization.')
        st.header("Pre-processing Attrition Analysis")
        data_type = st.selectbox("What type of data you want to visualize and analyze?", ["Numerical","Categorical"])
        if data_type == "Numerical":
            if st.checkbox("Factors for consideration"):
                feature = st.selectbox('Choose category : ',data_numeric_short.columns)
                analytics(feature)
                
                
            if st.checkbox("numerical data visualization"):
                show_columns_x= st.selectbox("Select your x axis" , data_numeric.columns)
                show_columns_y= st.selectbox("Select your y axis", data_numeric.columns)
                show_color=st.selectbox("What data do you want as color?",data_numeric.columns)
                if st.button("Let's go!") :
                    with st.spinner('Wait a sec'):
                        time.sleep(2)
                        st.success('Here it is!')
                        fig = px.bar(data, x=show_columns_x, y=show_columns_y, title='Distribution of attrition data')
                        st.plotly_chart(fig)
        
        if data_type == "Categorical":
            subject = st.radio('Choose category : ',['Department','EducationField',
                                                    'Gender','JobRole',
                                                    'MaritalStatus'])
            
            if st.button('Visualize!'):
                with st.spinner('Wait a sec'):
                    time.sleep(2)
                    st.success('Here it is!')
                    viz1(subject)

    elif page == "Employability Analysis":
        data_1["job_title"] = data_1["job_title"].astype(str)
        options = ["Job Title", "Key Skills", "Industry Type"]
        option = st.selectbox("Search by:", options)
        if option == "Job Title":
           search_term = "job_title"
        elif option == "Key Skills":
           search_term = "key_skills"
        else:
           search_term = "industry_type"
        query = st.text_input(f"Enter {search_term} to search:")
        if query:
           filtered_data = data_1[data_1[search_term].str.contains(query, case=False, na=False)]
        else:
           filtered_data = data_1
        company_count = filtered_data.groupby("company")["job_title"].count().reset_index()
        sorted_data = company_count.sort_values("job_title", ascending=False)
        st.write("Top companies and their job postings")
        st.write(sorted_data.head(20))
        job_count = filtered_data.groupby("job_title")["company"].count().reset_index()
        sorted_jobs = job_count.sort_values("company", ascending=False)
        st.write("Most available jobs and their occurrences")
        st.write(sorted_jobs.head(20))
   
            
    elif page == "Prediction":
        st.image('image3.png')
        st.title("Employabiliy Prediction")
        st.header('predict whether an employee will get dismissal from job or not')
        
        name = st.text_input('Write employee name:')
        
        age=st.slider('Age :',17,61,18)
        dataset.Age[0]=age
        
        joblevel=st.slider ('What is his/her job level?', 1,6,2)
        dataset.JobLevel[0]=joblevel
        
        income = st.slider('How much his/her income?' ,20000,200000,25000)
        dataset.MonthlyIncome[0]=income
        
        numcompaniesworked=st.slider('How many company has he/she been working before?' , 0,10,1)
        dataset.NumCompaniesWorked[0]=numcompaniesworked
        
        stock=st.slider ('How many stock does he/she has in this company?',0,4,1)
        dataset.StockOptionLevel[0]=stock
       
        total_working_years=st.slider('How long he/she has been working? ',0,50,1)
        dataset.TotalWorkingYears[0]=total_working_years
        
        bistrip= st.radio('How often does he/she travel for business reason? ',['Never','Rarely','Often'])
        if bistrip=='Never':
            dataset['Business Travel encode'][0]=0.0
        elif bistrip=='Rarely':
            dataset['Business Travel encode'][0]=1.0
        else:
            dataset['Business Travel encode'][0]=2.0
        
        x = data[['TotalWorkingYears','Age','MonthlyIncome','JobLevel','StockOptionLevel','NumCompaniesWorked','Business Travel encode']]
        y = data['attrition encode']  
        x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.3,random_state=45) 
        predict = st.selectbox("What model do you want to use?", ['Logistic Regression','Naive Bayes','Decision Tree','Random Forest'])
        if predict=='Logistic Regression':
            log_modelling(name)
        elif predict=='Naive Bayes':
            naivebayes_modelling(name)
        elif predict=='Decision Tree':
            decisiontree_modelling(name)
        elif predict=='Random Forest':
            randomforest_modelling(name)
           

### This is the start of function for data analytics
def analytics(feature):
    feature_head=data_numeric_short.sort_values(by = feature ).head(147)
    feature_tail=data_numeric_short.sort_values(by = feature ).tail(147)
    top_10= (feature_head[feature_head['Attrition'] == 'Yes'].count()[0]/147)*100
    bottom_10= (feature_tail[feature_tail['Attrition'] == 'Yes'].count()[0]/147)*100
    st.write("So there are " , top_10 , " % of job loss employee in top 10% rank and " , bottom_10 , " % of job loss employee in bottom 10% rank ")
    if (top_10 < 20.0) & (bottom_10 < 20.0) :
        st.write("It means " , feature ," doesn't really affect people to get job loss")
    elif (top_10 >= 20.0) :
        st.write("It means " , feature ," is one of the factor of attrition")
    elif (top_10 < 20.0) & (bottom_10 >=20.0) :
        st.write("It means " , feature ," is one of the factor of attrition but the correlation is negative")
### This is the end of function for data analytics
        
### This is the start of function for categorical data visualization    
def viz1(subject):       
    if subject=='Department':
        subject_x = 'Department_x'
        subject_y = 'Department_y'
    elif subject=='EducationField':
        subject_x = 'EducationField_x'
        subject_y = 'EducationField_y'
    elif subject=='Gender':
        subject_x = 'Gender_x'
        subject_y = 'Gender_y'
    elif subject=='JobRole':
        subject_x = 'JobRole_x'
        subject_y = 'JobRole_y'
    elif subject=='MaritalStatus':
        subject_x = 'MaritalStatus_x'
        subject_y = 'MaritalStatus_y'
        
    resign = pd.DataFrame(data.groupby([subject]).sum()['attrition encode'].reset_index())
    sub_all = pd.DataFrame(data[subject].value_counts().reset_index())

    sub_data = pd.merge(resign,sub_all,right_on='index', left_on= subject)
    sub_data['Not get job loss'] = sub_data[subject_y] - sub_data['attrition encode']
    sub_data['job loss Percentage'] =  (sub_data['attrition encode']/sub_data[subject_y])*100
    sub_data['Not get job loss percentage'] = (sub_data['Not get job loss']/sub_data[subject_y])*100

    
    plt.figure(figsize=[15,10])
    plt.bar(x=sub_data['index'], height=sub_data['attrition encode'])
    plt.bar(x=sub_data['index'], height=(sub_data['Not get job loss']),bottom=sub_data['attrition encode'])
    
    plt.title("{} vs Number of Employee who job loss or not".format(subject))
    plt.legend(['job loss','Not get job loss'])
    plt.xlabel(subject)
    plt.ylabel("Number of People")
    st.pyplot() 
    pd.options.display.float_format = '{:,.2f}'.format
    st.write(sub_data[[subject_x,subject_y,'job loss Percentage','Not get job loss percentage']].sort_values('Resign Percentage'))
###This is the end of function for categorical data visualization

###This is the definition of variable that we're using    
x = data[['TotalWorkingYears','Age','MonthlyIncome','JobLevel','StockOptionLevel','NumCompaniesWorked','Business Travel encode']]
y = data['attrition encode']  
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.3,random_state=45) 
###This is the end of variable definition 

###This is the start of function for logistic regression
def log_modelling(name):
    model_lr = LogisticRegression()
    model_lr.fit(x_train, y_train)
    accuracy = model_lr.score(x_test,y_test)
    st.write("Your employee with name " , name , " has ",  int(model_lr.predict_proba(dataset)[0][1] *100), "% chances to get job loss")
    st.write('This logistic regression model has accuracy :',accuracy*100,'%') 
###This is the end of function for logistic regression
    
###This is the start of function for naive bayes theorem
def naivebayes_modelling(name):
    model_nb = GaussianNB()
    model_nb.fit(x_train, y_train)
    accuracy1 = model_nb.score(x_test,y_test)
    st.write("Your employee with name " , name , " has ",  int(model_nb.predict_proba(dataset)[0][1] *100), "% chances to get job loss")
    st.write('This naive bayes model has accuracy :',accuracy1*100,'%') 
###This is the end of function for naive bayes theorem
    
###This is the start of function for decision tree 
def decisiontree_modelling(name):
    model_dt = DecisionTreeClassifier()
    model_dt.fit(x_train, y_train)
    accuracy2 = model_dt.score(x_test,y_test)
    if model_dt.predict(dataset)[0] == 0:
        status = ('will not get job loss')
    else:
        status = st.write('will get job loss')
    st.write("Your employee with name " , name , status)
    st.write('This decision tree model has accuracy :',accuracy2*100,'%') 
    st.write("Oh yeah and just in case you want to check whether feature I used is also important based on feature importance in decision tree, here you go :")
    featimp = pd.DataFrame(list(model_dt.feature_importances_), columns = ['Importances'])
    featcol = pd.DataFrame(list(x_train.columns), columns = ['Parameter'])
    featimp = featimp.join(featcol)
    featimp = pd.DataFrame(featimp.sort_values(by = ['Importances'], ascending = False))
    st.write("Feature importances : \n" , featimp)
###This is the end of function for decision tree

###This is the start of function for random forest
def randomforest_modelling(name):
    model_rf = RandomForestClassifier()
    model_rf.fit(x_train, y_train)
    accuracy3 = model_rf.score(x_test,y_test)
    if model_rf.predict(dataset)[0] == 0:
        status = ('will not get job loss')
    else:
        status = st.write('will get job loss')
    st.write("Your employee with name " , name , status)
    st.write('This random forest model has accuracy :',accuracy3*100,'%') 
    st.write("Oh yeah and just in case you want to check whether feature I used is also important based on feature importance in random forest, here you go :")
    
    feat_importances = pd.Series(model_rf.feature_importances_, index=x.columns)
    feat_importances = feat_importances.nlargest(20)
    fig, ax = plt.subplots()
    feat_importances.plot(kind='barh', ax=ax)
    st.pyplot(fig)
###This is the end of function for random forest
    
main()