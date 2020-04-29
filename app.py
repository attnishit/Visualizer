import streamlit as st 
import pandas as pd 
import numpy as np 

# Data Viz Pkg 
import matplotlib.pyplot as plt 
import matplotlib  
matplotlib.use('Agg')
import seaborn as sns 

# ML Pkg




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

	elif choice == 'Model Building' : 
		st.subheader("Building ML Model")

	elif choice == 'About' : 
		st.subheader("About")


if __name__ == '__main__':
	main()


