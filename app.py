import streamlit as st 
import pandas as pd 
import numpy as np 

# Data Viz Pkg 
import matplotlib.pyplot as plt 
import matplotlib  
matplotlib.use('Agg')
import seaborn as sns 

# ML Pkg
from sklearn import model_selection
from sklearn.linear_model import  LogisticRegression
from sklearn.neighbors import  KNeighborsClassifier	
from sklearn.naive_bayes  import GaussianNB
from sklearn.svm import SVC


def main():

	st.title('Visualizer')

	activities = ["EDA","Plot","Model Building","About"]

	choice = st.sidebar.selectbox("Select Activity",activities)

	if choice == 'EDA' : 
		
		st.subheader("Exploratory Data Analysis")
		data = st.file_uploader("Upload Dataset",type = ["csv","txt","xls"])		
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())

			if st.checkbox("Show shape"):
				st.write(df.shape)

			if st.checkbox("Show Columns"):
				all_columns = df.columns.to_list()
				st.write(all_columns)

			if st.checkbox("Select Columns To Show"):
				selected_columns = st.multiselect("Select Columns",all_columns)
				new_df = df[selected_columns]
				st.dataframe(new_df)

			if st.checkbox("Show Summary"):
				st.write(df.describe())

			if st.checkbox("Show Value Counts"):
				st.write(df.iloc[:,-1].value_counts()) # select the target columns..assuming last columns is target columns


	elif choice == 'Plot' : 
		st.subheader("Data Visualization")

		data = st.file_uploader("Upload Dataset",type = ["csv","txt","xls"])		
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())

		if st.checkbox("Correlation with Seaborn"):
			st.write(sns.heatmap(df.corr(),annot = True))
			st.pyplot()

		if st.checkbox("Pie chart"):
			all_columns = df.columns.to_list()
			columns_to_plot = st.selectbox("Select Column",all_columns)
			pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct = "%1.1f%%")
			st.write(pie_plot)
			st.pyplot()

		all_columns_names = df.columns.to_list()
		type_of_plot = st.selectbox("Select type of Plot",["area","bar","line","hist","box","kde"])
		selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

		if st.button("Generate Plot"):
			st.success("Generating Customizable Plot {} for {}".format(type_of_plot,selected_columns_names))

			if type_of_plot == 'area':
				collect_data = df[selected_columns_names]
				st.area_chart(collect_data)				
			elif type_of_plot == 'bar':
				collect_data = df[selected_columns_names]
				st.bar_chart(collect_data)

			elif type_of_plot == 'line':
				collect_data = df[selected_columns_names]
				st.line_chart(collect_data)

			elif type_of_plot:
				collect_data = df[selected_columns_names].plot(kind = type_of_plot)
				st.write(collect_data)
				st.pyplot()

	elif choice == 'Model Building' : 
		st.subheader("Building ML Model")
		data = st.file_uploader("Upload Dataset",type = ["csv","txt","xls"])		
		if data is not None:
			df = pd.read_csv(data)
			st.dataframe(df.head())

			X = df.iloc[:,0:-1]
			y = df.iloc[:,-1]
			seed = 7
			
			# Model Building : Assuming last column is target columns
			models = []
			models.append(("LR",LogisticRegression()))
			models.append(("LDA",LinearDiscriminantAnalysis()))
			models.append(("KNN",KNeighborsClassifier()))
			models.append(("CART",DecisionTreeClassifier()))
			models.append(("NB",GaussianNB()))
			models.append(("SVM",SVC()))

			# Evaluate each model in turn

			# List
			model_names = []
			model_mean = []
			model_std = []
			all_models = []
			scoring = 'accuracy'

			for name,model in models:
				kfold = model_selection.KFold(n_splits=10,random_state=seed)
				cv_results = model_selection.cross_val_score(model,X,Y,cv = kfold,scoring=scoring)
				model_names.append(name)
				model_mean.append(cv_results.mean())
				model_std.append(cv_results.std())

				accuracy_results = {"model_name":name,"model_accuracy":cv_results.mean(),"standard_deviation":cv_results.std()}
	


	elif choice == 'About' : 
		st.subheader("About")


if __name__ == '__main__':
	main()


