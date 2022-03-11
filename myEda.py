import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from urllib.request import urlopen
import json
import plotly.express as px
import random

random.seed(7)
plt.rcParams.update({'font.size': 22})

class EDA:
	def __init__(self,data):
		self.data=data
	def preprocessing(self):

		threshold = len(self.data) * .99
		prdata = self.data.dropna(thresh=threshold, axis=1)
		prdata['loan_status'].replace({'Does not meet the credit policy. Status:Fully Paid': 'Fully Paid', 'Does not meet the credit policy. Status:Charged Off': 'Charged Off'}, inplace=True)
		#prdata = prdata.sample(frac=0.7)
		self.data = prdata

	def boxplots(self,x,y):
		sns.set_style("whitegrid")

		plt.figure(figsize=(20,8))
		sns.boxplot(x=self.data[x], y=self.data[y])


	def pubrec_simpledist(self, x, hue, col=False):
		plt.figure(figsize=(20,8))
		sns.displot(data=self.data, x=x, hue=hue,col=col, kind="kde").set(xlim=(0,5))

	def mapdist_loan_amt(self):
		loan = self.data.groupby('addr_state').sum()['loan_amnt'].reset_index()
		loan['loan_amnt'] = loan['loan_amnt']/1000000

		fig = px.choropleth(loan,
		    locationmode="USA-states",
		    locations='addr_state', 
		    color='loan_amnt',
		    color_continuous_scale="Viridis",
		    scope="usa",
		    labels={'loan_amnt':'Total Loan in Million $'},
		    title = "Total Loan Amount in Million $"
		)
		fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

		# Improve the legend
		fig.update_layout(coloraxis_colorbar=dict(
		    thicknessmode="pixels", thickness=10,
		    yanchor="top", y=0.8,
		    ticks="outside"
		))
		fig.show(renderer='notebook')
		fig.write_html("./choropleth-map-loanamt-python.html")



	def mapdist_pur_loan_amt(self,purpose):
		loan = self.data[self.data['purpose']==purpose].groupby('addr_state').sum()['loan_amnt'].reset_index()
		loan['loan_amnt'] = loan['loan_amnt']/1000000

		fig = px.choropleth(loan,
		    locationmode="USA-states",
		    locations='addr_state', 
		    color='loan_amnt',
		    color_continuous_scale="Viridis",
		    scope="usa",
		    labels={'loan_amnt':'Total Education Loan in Million $'},
		    title = "Total Loan Amount in Million $"
		)
		fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

		# Improve the legend
		fig.update_layout(coloraxis_colorbar=dict(
		    thicknessmode="pixels", thickness=10,
		    yanchor="top", y=0.8,
		    ticks="outside"
		))
		fig.show(renderer='notebook')
		fig.write_html("./"+purpose+"-choropleth-map-loanamt-python.html")


	# def mapdist_vac_loan_amt(self):
	# 	vac = self.data[self.data['purpose']=='vacation'].groupby('addr_state').sum()['loan_amnt'].reset_index()
	# 	vac['loan_amnt'] = vac['loan_amnt']/1000000
	# 	fig = px.choropleth(self.data, 
	# 	    locationmode="USA-states",
	# 	    locations='addr_state', 
	# 	    color='loan_amnt',
	# 	    color_continuous_scale="Viridis",
	# 	    scope="usa",
	# 	    labels={'loan_amnt':'Total Vacation Loan in Million $'},
	# 	    title = "Total Vacation Loan in Million $"
	# 	)
	# 	fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

	# 	# Improve the legend
	# 	fig.update_layout(coloraxis_colorbar=dict(
	# 	    thicknessmode="pixels", thickness=10,
	# 	    yanchor="top", y=0.8,
	# 	    ticks="outside"
	# 	))
	# 	# fig.show(renderer='notebook')
	# 	fig.write_html("./choropleth-map-vac-python.html")

	def logcountplot(self,x,hue=None):
		plt.figure(figsize=(20,8))
		sns.set_style("whitegrid")
		countpl = sns.countplot(x=x, data=self.data, hue=hue)
		countpl.set_yscale("log")
		_ = countpl.set(xlabel=x, ylabel="Count in Log Scale")

	def comparedist(self, x, hue, col=False):
		plt.figure(figsize=(20,8))
		sns.displot(data=self.data, x=x, hue=hue,col=col, kind="kde",col_wrap=3, facet_kws=dict(sharex=False))

	def datedist(self, x, hue, col=False):
		self.data['year'] = pd.DatetimeIndex(self.data[x]).year
		plt.figure(figsize=(20,8))
		sns.displot(data=self.data, x='year', hue=hue,col=col, kind="kde")


	def incomedist(self):
		df = self.data[self.data['annual_inc'] <= 250000]
		sns.displot(data=df, x='annual_inc', hue='loan_status', bins=80,  kde=True, palette='viridis', height=7, aspect=1).set(title="Annual Income less than 250000");

	def fico_loan(self):
		df = self.data[['fico_range_high','loan_status','fico_range_low']]
		df['fico'] = (df['fico_range_high']+df['fico_range_low'])//2
		plt.figure(figsize=(20,7), dpi=300)
		sns.displot(data=df, x='fico', hue='loan_status', bins=100, height=4, aspect=3, kde=True, palette='viridis')

	def comparehist(self,x,hue):
		sns.displot(data=self.data, x=x, hue=hue, bins=80,  kde=True, palette='viridis', height=7, aspect=1)

	def simpledist(self, x, hue, col=False):
		plt.figure(figsize=(20,8))
		sns.displot(data=self.data, x=x, hue=hue,col=col, kind="kde")

	def dti_loanst(self, x, hue, col=False):
		plt.figure(figsize=(20,8))
		sns.displot(data=self.data, x=x, hue=hue,col=col, kind="kde").set(xlim=(0,150))

	def fico_date(self):
		df = self.data[['fico_range_high','fico_range_low','earliest_cr_line']]
		df['fico'] = (df['fico_range_high']+df['fico_range_low'])//2
		df['year'] = pd.DatetimeIndex(self.data['earliest_cr_line']).year
		sns.displot(data=df, x='fico', y="year")










