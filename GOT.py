#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 09:07:25 2019

@author: pc
"""
"""

##############################################################################
##############################################################################

                        # WINTER IS COMING! #

##############################################################################
##############################################################################


##### Data Dictionary #####

S.No	    --> Character number (by order of appearance)
name	    --> Character name
title	    --> Honorary title(s) given to each character
male	    --> 1 = male, 0 = female
culture	    --> Indicates the cultural group of a character
dateOfBirth	--> Known dates of birth for each character (measurement unknown)

mother	    --> Character's biological mother
father	    --> Character's biological father
heir	    --> Character's biological heir
house	    --> Indicates a character's allegiance to a house 
                (i.e. a powerful family)
spouse	    --> Character's spouse(s)

book1_A_Game_Of_Thrones    --> 1 appeared in book, 0 = did not appear in book
book2_A_Clash_Of_Kings	   --> 1 appeared in book, 0 = did not appear in book
book3_A_Storm_Of_Swords	   --> 1 appeared in book, 0 = did not appear in book
book4_A_Feast_For_Crows	   --> 1 appeared in book, 0 = did not appear in book
book5_A_Dance_with_Dragons --> 1 appeared in book, 0 = did not appear in book

isAliveMother --> 1 = alive, 0  not alive
isAliveFather --> 1 = alive, 0  not alive
isAliveHeir	  --> 1 = alive, 0 = not alive
isAliveSpouse --> 1 = alive, 0 = not alive
isMarried     --> 1 = married, 0 = not married
isNoble       --> 1 = noble, 0 = not noble

age           --> Character's age in years
numDeadRelations	 --> Total number of deceased relatives throughout all of the 
                     books
popularity    --> Indicates the popularity of a character 
                 (1 = extremely popular (max), 0 = extremely unpopular (min))
isAlive	      --> 1 = alive, 0 = not alive


"""



###############################################################################
# Importing libraries & Files
###############################################################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm   
import statsmodels.formula.api as smf

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler 

from sklearn.model_selection import GridSearchCV  
from sklearn.model_selection import  RandomizedSearchCV

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier



#########

file = 'GOT_character_predictions.xlsx'
GOT= pd.read_excel(file)


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


###############################################################################
# Fundamental Dataset Exploration
###############################################################################

# Column names
GOT.columns

# Dimensions of the DataFrame
GOT.shape

# Information about each variable
GOT.info()


# Descriptive statistics
desc = GOT.describe().round(2)

print(desc)


###############################################################################
# Imputing Missing Values
###############################################################################


print(
      GOT
      .isnull()
      .sum()
      )


"""
13 columns with Missing Values

Non-numeric: title, culture, mother, father, heir, house, spouse, 
            isAliveMother, isAliveFather, isAliveHeir, isAliveSpouse
Numeric: age, DOB

"""

# Flagging missing values

for col in GOT:
    
    #print(col)
    
    """Create columns that are 0s if a value was not missing and 1 if
    a value is missing."""
    
    if GOT[col].isnull().astype(int).sum() > 0:
        
        GOT['m_'+col] = GOT[col].isnull().astype(int)

GOT.shape

###########################

"""
Clean GOT (GOT_c) : Remove columns which have >75% missing values
"""

GOT_c = GOT
GOT_c = GOT_c.drop(columns = ['mother','father','isAliveMother', 
                              'isAliveFather','m_mother','m_father',
                              'm_isAliveMother', 'm_isAliveFather' ])

print(
      GOT_c
      .isnull()
      .sum()
      )


GOT_c1 = GOT_c
GOT_c1 = GOT_c1.drop(columns = ['dateOfBirth'])
GOT_c1.shape


"""
Fill in Missing Values
 - for text columns, replace blanks with 'unknown'. 
 - For numeric columns, replace with mean

"""


#######  
# title
####### 
        
GOT_c1['title'] = GOT_c1['title'].fillna('Unknown')

#######  
# culture
####### 

GOT_c1['culture'] = GOT_c1['culture'].fillna('Unknown')

#######  
# heir
####### 

GOT_c1['heir'] = GOT_c1['heir'].fillna('Unknown')


#######  
# house
#######

GOT_c1['house'] = GOT_c1['house'].fillna('Unknown')

#######  
# spouse
#######

GOT_c1['spouse'] = GOT_c1['spouse'].fillna('Unknown')

#######  
# isAliveSpouse 
#######
"""
For missing values in this column, fill with zero as assuming they dont have
any spouse.
"""

GOT_c1['isAliveSpouse'] = GOT_c1['isAliveSpouse'].fillna('0')

#######  
# isAliveHeir
#######

"""
For missing values in this column, fill with zero as assuming they dont have
any heir.
"""

GOT_c1['isAliveHeir'] = GOT_c1['isAliveHeir'].fillna('0') 

#######  
# age
#######  

GOT_c1['age'].describe()

pd.np.mean(GOT_c1['age'][GOT_c1['age']>=0])


def fn(row):
    if row['age'] < 0:
        row['age'] = 36.8
    
    return row
        
GOT_c1 = GOT_c1.apply(fn, axis = 1)   
    

if GOT_c1['age'].isnull().astype(int).sum() > 0:

        age_median = 36.8

        GOT_c1['age'] = GOT_c1['age'].fillna(age_median).round(2)

# Check if age has any missing values
print(
      GOT_c1
      .isnull()
      .sum()
      )

GOT_df = pd.DataFrame(GOT_c1)
GOT_df.columns


###############################################################################
# Feature Engineering
###############################################################################

"""
Changing Categorical text variables into binary / numerical 
New variables with interlinkages

"""
##########
#  Yes/No for Title
##########

def func(title):
    if title == 'Unknown':
        return 0
    else:
        return 1

GOT_df['title_group'] = GOT_df['title'].map(func)

GOT_df['title'].value_counts()


##########
#  Yes/No for Culture
##########

def func(culture):
    if culture == 'Unknown':
        return 0
    else:
        return 1

GOT_df['culture_group'] = GOT_df['culture'].map(func)

##########
#  Has Heir
##########

def func(heir):
    if heir == 'Unknown':
        return 0
    elif heir == 0:
        return 0
    else:
        return 1

GOT_df['heir_group'] = GOT_df['heir'].map(func)


##########
#  Group by ages
##########

"""
Rationale for age group is if the person is a child (<20 years) or old (>=50 
years). This will probably impact survival chances as the young and old may
not be able to fight in the wars and be dependent on others to take care of
them. 
"""

def func(age):
    
    if age < 20:
        return 1
    elif age < 50:
        return 2
    else:
        return 3

GOT_df['age_group'] = GOT_df['age'].map(func)

GOT_df['age_group'].value_counts()


##########
#  Group by number of books character appears in
##########

GOT_df['total_books'] = (GOT_df['book1_A_Game_Of_Thrones'] 
                        + GOT_df['book2_A_Clash_Of_Kings']
                        + GOT_df['book3_A_Storm_Of_Swords'] 
                        + GOT_df['book4_A_Feast_For_Crows'] 
                        + GOT_df['book5_A_Dance_with_Dragons'])



##########
#  Group by popularity
##########

"""
Rationale is that the more popular the character, the greater the chances of
surviving. 
However, after analysis subsequently, this theory is supported. But from
cross research from the author George RR Martin, infact the higher the 
popularity, the more the author would like to kill of the character to shock
reader. 
"""


GOT_df['popularity'].value_counts()
GOT_df['popularity'].hist()

def func(pop):
    
    if pop == 0:
        return 0
    elif pop < 0.05:
        return 1
    elif pop < 0.20:
        return 2
    else:
        return 3

GOT_df['pop_group'] = GOT_df['popularity'].map(func)

GOT_df['pop_group'].value_counts()


def func(pop):
    
    if pop < 0.1:
        return 0
    else:
        return 1

GOT_df['pop_group_2'] = GOT_df['popularity'].map(func)

GOT_df['pop_group_2'].value_counts()

def func(pop):
    
    if pop < 0.1:
        return 1
    elif pop < 0.30:
        return 2
    elif pop < 0.50:
        return 3
    else:
        return 4

GOT_df['pop_group_3'] = GOT_df['popularity'].map(func)

GOT_df['pop_group_3'].value_counts()

##########
#  Number of dead relations
##########
"""
Rationale is that if you have a family , you would be more likely to protect
them or be protected. So, it should influence your chances of survival. 

"""

def func(num):
    if num == 0:
        return 0
    elif num < 5:
        return 1
    elif num < 10:
        return 2
    else:
        return 3

GOT_df['numDeadRelations_group'] = GOT_df['numDeadRelations'].map(func)

GOT_df['numDeadRelations_group'].value_counts()


##########
#  Yes/No for Spouse
##########

def func(spouse):
    if spouse == 'Unknown':
        return 0
    elif spouse == 0:
        return 0
    else:
        return 1

GOT_df['spouse_group'] = GOT_df['spouse'].map(func)


##########
#  Yes/No for House
##########

def func(house):
    if house == 'Unknown':
        return 0
    else:
        return 1

GOT_df['house_group'] = GOT_df['house'].map(func)


##########
#  Group by number of people in each house
##########

"""
Top Houses:
- Unknown: Many small random people
- Night's watch: Responsible for protecting the wall. More exposed to 
  extremeties.
- House Frey, Stark, Targaryen, Lannister: Traditional Powerhouses 
"""

# Grouping by number of people in each house

GOT_df['house_num'] = GOT_df['house'].map(GOT_df['house'].value_counts())

def func(house):
    if house < 10:
        return 1
    elif house < 30:
        return 2
    elif house < 55:
        return 3
    else: 
        return 4

GOT_df['house_num_2'] = GOT_df['house_num'].map(func) 

GOT_df['house_num_2'].value_counts()


# Grouping of houses by average survival %

Alive_house_mean = GOT_df.groupby('house').mean()

def func(x):
    if x < 0.3:
        return 1
    elif x < 0.5:
        return 2
    elif x < 0.7:
        return 3
    else:
        return 4

Alive_house_mean['h_group'] = Alive_house_mean['isAlive'].map(func)

GOT_df['h_group'] = GOT_df['house'].map(Alive_house_mean['h_group'])


List_of_houses = Alive_house_mean['h_group'].sort_values(ascending=False)


# # Grouping of number of Dead Relations by average survival %

Alive_numdead_mean = GOT_df.groupby('numDeadRelations_group').mean()

def func(x):
    if x < 0.5:
      return 0
    elif x < 0.75:
      return 1
    else:
        return 2
   

Alive_numdead_mean['numdead_group'] = Alive_numdead_mean['isAlive'].map(func)

GOT_df['numdead_group'] = (GOT_df['numDeadRelations_group']
                        .map(Alive_numdead_mean['numdead_group']))

GOT_df['numdead_group'].value_counts()


# Grouping of characters with titles by average survival %

Alive_title_mean = GOT_df.groupby('title').mean()

def func(x):
    if x < 0.7:
        return 1
    elif x < 0.75:
        return 2
    elif x < 0.8:
        return 3
    else:
        return 4

Alive_title_mean['title_group_2'] = Alive_title_mean['isAlive'].map(func)

GOT_df['title_group_2'] =GOT_df['title'].map(Alive_title_mean['title_group_2'])

GOT_df['title_group_2'].value_counts()

List_of_title = Alive_title_mean['title_group_2'].sort_values(ascending=False)


# Grouping of characters by popularity by average survival %

Alive_pop_mean = GOT_df.groupby('popularity').mean()

def func(x):
    if x < 0.7:
        return 1
    elif x < 0.8:
        return 2
    else:
        return 3

Alive_pop_mean['pop_m1'] = Alive_pop_mean['isAlive'].map(func)

GOT_df['pop_m1'] = GOT_df['popularity'].map(Alive_pop_mean['pop_m1'])

GOT_df['pop_m1'].value_counts()


# Grouping of characters' culture by average survival %
""" 
Possible clean up of the text in culture column for instance there is "Bravos"
and "Bravosi" which possibly refers to the same culture, just a refernce for
male and female. However, cleaning that up did not influence the grouping
methodology. 
"""

Alive_cul_mean = GOT_df.groupby('culture').mean()

def func(x):
    if x < 0.75:
        return 1
    elif x < 0.77:
        return 2
    else:
        return 3

Alive_cul_mean['cul_m1'] = Alive_cul_mean['isAlive'].map(func)

GOT_df['cul_m1'] = GOT_df['culture'].map(Alive_cul_mean['cul_m1'])

GOT_df['cul_m1'].value_counts()


# Grouping of characters' presence in books by average survival %
"""
Rationale that characters present in the latter books of Books 4 and 5 are more
important characters so they might have higher chance of survival. 

"""

GOT_df.columns

GOT_df['book45'] = (GOT_df['book4_A_Feast_For_Crows'] 
                    + GOT_df['book5_A_Dance_with_Dragons'])

Alive_book_mean = GOT_df.groupby('book45').mean()

def func(x):
    if x < 0.5:
        return 1
    elif x < 0.8:
        return 2
    else:
        return 3

Alive_book_mean ['book45_m1'] = Alive_book_mean['isAlive'].map(func)

GOT_df['book45_m1'] = GOT_df['book45'].map(Alive_book_mean ['book45_m1'])

GOT_df['book45_m1'].value_counts()


"""
Rationale that being male or female should influence the survival rate since
most men are probably out in the battlefield. However, from further analysis
logistic regression was not identifying the male variable as signficant 
(p-value > 0.05), hence exploring various combinations below to see if there 
is any hidden relationship.

"""

# Group by  male + popularity by survival percentage

GOT_df.columns

GOT_df['male_pop'] = GOT_df['male'] + GOT_df['pop_m1'] 
Alive_male_pop_mean = GOT_df.groupby('male_pop').mean()

def func(x):
    if x < 0.6:
        return 1
    elif x < 0.85:
        return 2
    else:
        return 3

Alive_male_pop_mean['male_pop'] = Alive_male_pop_mean['isAlive'].map(func)

GOT_df['male_pop_m1'] = GOT_df['male_pop'].map(Alive_male_pop_mean['male_pop'])

GOT_df['male_pop_m1'].value_counts()


# Group by  male + noble by survival percentage

GOT_df['malenob'] = GOT_df['male'] + GOT_df['isNoble']
Alive_malnob_mean = GOT_df.groupby('malenob').mean()

def func(x):
    if x < 0.7:
        return 1
    elif x < 0.8:
        return 2
    else:
        return 3

Alive_malnob_mean['malnob_m1'] = Alive_malnob_mean['isAlive'].map(func)

GOT_df['malnob_m1'] = GOT_df['malenob'].map(Alive_malnob_mean['malnob_m1'])

GOT_df['malnob_m1'].value_counts()

# Group by male and number of relations and survival percentage

GOT_df['male_rel'] = GOT_df['male'] + GOT_df['numDeadRelations_group']
Alive_malerel_mean = GOT_df.groupby('male_rel').mean()


def func(x):
    if x < 0.8:
        return 0
    else:
        return 1

Alive_malerel_mean['male_rel'] = Alive_malerel_mean['isAlive'].map(func)

GOT_df['male_rel'] = GOT_df['male_rel'].map(Alive_malerel_mean['male_rel'])

GOT_df['male_rel'].value_counts()


# Survival % by male

Alive_male_mean = GOT_df.groupby('male').mean()
def func(x):
    if x < 0.7:
        return 0
    else:
        return 1
Alive_male_mean['male_m1'] = Alive_male_mean['isAlive'].map(func)

GOT_df['male_m1'] = GOT_df['male'].map(Alive_male_mean['male_m1'])

GOT_df['male_m1'].value_counts()  


"""
Rationale: Character is Noble would have more protection and hence more chances
of survival. 
"""

# Group by is noble by survival percentage

Alive_noble_mean = GOT_df.groupby('isNoble').mean()

def func(x):
    if x < 0.85:
        return 1
    elif x < 0.8:
        return 2
    else:
        return 3

Alive_noble_mean['nob_m1'] = Alive_noble_mean['isAlive'].map(func)

GOT_df['nob_m1'] = GOT_df['isNoble'].map(Alive_noble_mean['nob_m1'])

GOT_df['nob_m1'].value_counts()


# Group by is noble + married by survival percentage

"""
Rationale that noble and married should be signfiincat but they are not based 
on logistic regression and random forest feature selection. 
"""

GOT_df['nob_mar'] = GOT_df['isNoble'] + GOT['isMarried']
Alive_nobmar_mean = GOT_df.groupby('nob_mar').mean()


def func(x):
    if x < 0.6:
        return 0
    else:
        return 1


Alive_nobmar_mean['nobmar_m1'] = Alive_nobmar_mean['isAlive'].map(func)

GOT_df['nobmar_m1'] = GOT_df['nob_mar'].map(Alive_nobmar_mean['nobmar_m1'])

GOT_df['nobmar_m1'].value_counts()


# Group by age and survival percentage

Alive_age_mean = GOT_df.groupby('age').mean()

def func(x):
    if x < 0.5:
        return 1
    elif x < 0.75:
        return 2
    else:
        return 3

Alive_age_mean['age_m1'] = Alive_age_mean['isAlive'].map(func)

GOT_df['age_m1'] = GOT_df['age'].map(Alive_age_mean['age_m1'])

GOT_df['age_m1'].value_counts()

# Group by heir and survival percentage

Alive_heir_mean = GOT_df.groupby('heir_group').mean()

def func(x):
    if x < 0.5:
        return 0
    else:
        return 1

Alive_heir_mean['heir_m1'] = Alive_heir_mean['isAlive'].map(func)

GOT_df['heir_m1'] = GOT_df['heir_group'].map(Alive_heir_mean['heir_m1'])

GOT_df['heir_m1'].value_counts()






###############################################################################
# Outlier analysis
###############################################################################

# this df only has the numerical columns

GOT_df_num = pd.concat([GOT_df['age'], GOT_df['popularity']], axis = 1)

# number of numerical variables

total_numvarumi = GOT_df_num.shape[1]

# calculate quantiles for numerical variables

iqr = 1.5*(GOT_df_num.quantile(0.75) - GOT_df_num.quantile(0.25))
GOT_df_num_h = GOT_df_num.quantile(0.75) + iqr
GOT_df_num_l = GOT_df_num.quantile(0.25) - iqr

#create outlier df

out_cols=pd.DataFrame()

for i in range(total_numvarumi):

	out_cols['outlier_'+ GOT_df_num.keys()[i]] =(
                                            ((GOT_df_num.iloc[:,i] 
                                            > GOT_df_num_h[i]) 
                                            | (GOT_df_num.iloc[:,i] 
                                            < GOT_df_num_l[i]))
                                            )

    
# merge original df and outlier columns

GOT_df = pd.concat([GOT_df, out_cols], axis = 1)



###############################################################################
# Histograms
###############################################################################

GOT_df.hist(bins=50, figsize=(20,20))
plt.show()


###############################################################################
# Box Plots
###############################################################################

GOT_df.columns


# Popularity
GOT_df.boxplot(column = ['popularity'],
                by = ['isAlive'],
                vert = True,
                patch_artist = False,
                meanline = True,
                showmeans = True)

# Age
GOT_df.boxplot(column = ['age'],
                by = ['isAlive'],
                vert = True,
                patch_artist = False,
                meanline = True,
                showmeans = True)

# Number of Dead Relations
GOT_df.boxplot(column = ['numDeadRelations'],
                by = ['isAlive'],
                vert = True,
                patch_artist = False,
                meanline = True,
                showmeans = True)

# Number of total books
GOT_df.boxplot(column = ['total_books'],
                by = ['isAlive'],
                vert = True,
                patch_artist = False,
                meanline = True,
                showmeans = True)



###############################################################################
# Scatter Plots
###############################################################################

GOT_df.columns

# Age vs Popularity
plt.scatter(
            x = 'age',
            y = 'popularity',
            alpha = 0.2,
            color = 'blue',
            data = GOT_df
            )
plt.title('Age & Popularity')
plt.xlabel ('Age')
plt.ylabel('Popularity')

# Number of Dead Relations vs age
plt.scatter(
            x = 'numDeadRelations',
            y = 'age',
            alpha = 0.2,
            color = 'blue',
            data = GOT_df
            )
plt.title('Age & Number of dead relations')
plt.xlabel ('Number of Dead Relations')
plt.ylabel('Age')

#  Number of Dead Relations vs popularity
plt.scatter(
            x = 'numDeadRelations',
            y = 'popularity',
            alpha = 0.2,
            color = 'blue',
            data = GOT_df
            )
plt.title('Number of Dead Relations & Popularity')
plt.xlabel ('Number of Dead Relations')
plt.ylabel('Popularity')




###############################################################################
# Correlation Analysis
###############################################################################

df_corr = GOT_df.corr().round(2)

print(df_corr)

df_corr.loc['isAlive'].sort_values(ascending = False)


########################
# Correlation Heatmap
########################

# Using palplot to view a color scheme

sns.palplot(sns.color_palette('coolwarm', 12))

fig, ax = plt.subplots(figsize = (15, 15))


sns.heatmap(
            df_corr,
            cmap = 'coolwarm',
            square = True,
            annot = True,
            linecolor = 'black',
            linewidths = 0.5
            )

plt.savefig('GOT Correlation Heatmap.png')
plt.show()



###############################################################################
###############################################################################
###############################################################################
###############################################################################
                
                      # PREPARING FOR THE MODEL #

###############################################################################
###############################################################################
###############################################################################
###############################################################################

GOT_df.columns


GOT_data_o = GOT_df.loc [ : , ['book1_A_Game_Of_Thrones',
                             'book2_A_Clash_Of_Kings',
                             'book3_A_Storm_Of_Swords',
                             'book4_A_Feast_For_Crows',
                             'book5_A_Dance_with_Dragons',
                             'male',
                             'isAliveHeir',
                             'isAliveSpouse',
                             'isMarried',
                             'isNoble',
                             'age',
                             'numDeadRelations',
                             'popularity',
                             'age_group',
                             'total_books',
                             'pop_group',
                             'pop_group_2',
                             'pop_group_3',
                             'numDeadRelations_group',
                             'spouse_group',
                             'title_group',
                             'culture_group',
                             'heir_group',
                             'house_group',
                             'house_num_2',
                             'h_group',
                             'numdead_group',
                             'title_group_2',
                             'pop_m1',
                             'cul_m1',
                             'nob_m1',
                             'age_m1',
                             'heir_m1',
                             'nobmar_m1',
                             'male_m1',
                            'malnob_m1',
                            'male_rel',
                            'book45_m1',
                            'male_pop_m1',
                        
                             
                             ]]
GOT_target =  GOT_df.loc[ : , 'isAlive']

########################
# Scaling
########################

scaler = StandardScaler()
scaler.fit(GOT_data_o)
X_scaled = scaler.transform(GOT_data_o)
X_scaled_df = pd.DataFrame(X_scaled)
X_scaled_df.columns = GOT_data_o.columns

print(pd.np.var(GOT_data_o))
print(pd.np.var(X_scaled_df))

GOT_df_s = X_scaled_df


###############################################################################
# Train Test Split
###############################################################################

# Whole list of variables -- compiled here for easy reference. 

GOT_data = GOT_df_s.loc [ : , ['book1_A_Game_Of_Thrones',
                             'book2_A_Clash_Of_Kings',
                             'book3_A_Storm_Of_Swords',
                             'book4_A_Feast_For_Crows',
                             'book5_A_Dance_with_Dragons',
                             'male',
                             'isAliveHeir',
                             'isAliveSpouse',
                             'isMarried',
                             'isNoble',
                              'age',
                             'numDeadRelations',
                            'popularity',
                             'age_group',
                             'total_books',
                           'pop_group',
                            'pop_group_2',
                           'pop_group_3',
                             'numDeadRelations_group',
                             'spouse_group',
                             'title_group',
                             'culture_group',
                             'heir_group',
                            'house_group',
                            'house_num_2',
                            'h_group',
                            'numdead_group',
                            'title_group_2',
                             'pop_m1',
                             'cul_m1',
                             'nob_m1',
                             'age_m1',
                              'heir_m1',
                             'nobmar_m1',
                             'male_m1',
                             'malnob_m1',
                               'male_rel',
                                'book45_m1',
                                'male_pop_m1',
                             
                             ]]


GOT_target =  GOT_df.loc[ : , 'isAlive']

# Selected variables:

GOT_data = GOT_df_s.loc [ : , [
                            #'book1_A_Game_Of_Thrones',
                             #'book2_A_Clash_Of_Kings',
                            #'book3_A_Storm_Of_Swords',
                           'book4_A_Feast_For_Crows',
                             #'book5_A_Dance_with_Dragons',
                            # 'male',
                            # 'isAliveHeir',
                            #'isAliveSpouse',
                           # 'isMarried',
                           # 'isNoble',
                            #'age',
                           #  'numDeadRelations',
                            # 'popularity',
                            #'age_group',
                             'total_books',
                            # 'pop_group',
                             #'pop_group_2',
                              #'pop_group_3',
                           # 'numDeadRelations_group',
                            'spouse_group',
                           # 'title_group',
                            #'culture_group',
                            # 'heir_group',
                            # 'house_group',
                            # 'house_num_2',
                             'h_group',
                           # 'numdead_group',
                             'title_group_2',
                             'pop_m1',
                             'cul_m1',
                            # 'nob_m1'
                             'age_m1',
                            # 'heir_m1',
                            # 'nobmar_m1',
                             'male_m1',
                            # 'malnob_m1',
                             #  'male_rel',
                            #  'book45_m1',
                            #'male_pop_m1',
                             ]]

X_train, X_test, y_train, y_test = train_test_split(
            GOT_data,
            GOT_target,
            test_size = 0.1,
            random_state = 508,
            stratify = GOT_target)


# Training set 
print(X_train.shape)
print(y_train.shape)

# Testing set
print(X_test.shape)
print(y_test.shape)

GOT_df_train = pd.concat([X_train, y_train], axis = 1)
GOT_df_train = pd.DataFrame(GOT_df_train)



###############################################################################
###############################################################################
###############################################################################
###############################################################################
                
                                # BEST MODEL #

###############################################################################
###############################################################################
###############################################################################
###############################################################################
"""
Various models were run across 
(1) Logistic Regression
(2) KNN Classification
(3) Random Forest
(4) Gradient Boost Machines


The best model is found using KNN Classification. 

- Training score: 0.863
- Testing score: 0.887 
--> Difference is small at 0.02. Model is marginally underfiting 

- Train AUC: 0.790
- Test AUC: 0.793
--> Difference is small at 0.003. Model is marginally underfitting.

- CV score using roc_auc : 0.88
--> Is a high score using cv=3, meaning the model is doing well.

- F1 score: 88
--> The F1 score is a balance between precision and recall. 88% is a relatively
    high value and reflects well on the model accuracy. Out of the 195 test 
    samples, 173 were correctly predicted.


Rationale for choosing best model:
- Highest testing score, with minimal difference between train and test score
- Highest AUC score after cross validation
- Highest F1 score
- Features selected are impactful and signficant in helping in analysis of
how characters in Game Of Thrones can survive based on 9 traits


Details below on the model, hyperparamter tuning, accuracy and AUC scores, 
cross-validations, confusion matrix and classifcation reports.

"""



###############################################################################
# KNN - Classifiers
###############################################################################

########################
# Finding Optimal Neighbours
########################

# Running the neighbor optimization code with a small adjustment for classification
training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train.values.ravel())
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


fig, ax = plt.subplots(figsize=(12,9))
plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()


# Looking for the highest test accuracy
print(test_accuracy)

# Printing highest test accuracy
print(test_accuracy.index(max(test_accuracy)) + 1)

"""
Number of neighbors with highest accuracy is at 12.
"""

########################
# Model
########################

knn_classifier = KNeighborsClassifier(n_neighbors =12)
knn_clf_fit = knn_classifier.fit(X_train,y_train)
knn_pred_test = knn_classifier.predict(X_test)

knn_pred_train = knn_classifier.predict(X_train)
knn_clf_pred_probabilities = knn_clf_fit.predict_proba(X_test)


########################
# Hyerparameter Tuning
########################

"""
Hyperparamter tuning was done but it did not improve the score. 
"""


params = {'n_neighbors': [7,8,9,10,11,12,13,14,15,16,17,18,19,20],
          'leaf_size': [1,2,3,4,5],
          'weights' : ['uniform' , 'distance'],
          'algorithm' : ['auto'],
          'metric': ['minkowski', 'euclidean', 'manhattan']}

knn_p_1 = GridSearchCV(knn_clf_fit , param_grid = params)
knn_p_1_fit = knn_p_1.fit(X_train,y_train)
knn_p_1_pred = knn_p_1_fit.predict(X_test)


knn_p_1_train_score = knn_p_1_fit.score(X_train, y_train).round(4)
knn_p_1_test_score = knn_p_1_fit.score(X_test, y_test).round(4)

print(classification_report(y_true = y_test,
                            y_pred = knn_p_1_pred))

print("Best Hyper Parameters: \n" , knn_p_1.best_params_)
print(knn_p_1_train_score)
print(knn_p_1_test_score)


########################
# Model post hyperparameter tuning
########################

knn_classifier = KNeighborsClassifier(n_neighbors = 12, 
                                      metric = 'minkowski', 
                                      weights = 'uniform', 
                                      leaf_size = 2, 
                                      algorithm = 'auto')
knn_clf_fit_hp = knn_classifier.fit(X_train,y_train)
knn_pred_test_hp = knn_classifier.predict(X_test)

knn_pred_train_hp = knn_classifier.predict(X_train)
knn_clf_pred_probabilities_hp = knn_clf_fit.predict_proba(X_test) 



########################
# Scores
########################

knn_clf_pred_probabilities = knn_clf_fit.predict_proba(X_test)

knn_train_score = knn_clf_fit.score(X_train, y_train).round(4)
knn_test_score = knn_clf_fit.score(X_test, y_test).round(4)
knn_diff_score = knn_test_score  - knn_train_score

knn_train_auc_score = roc_auc_score(y_train,knn_pred_train).round(4)
knn_test_auc_score = roc_auc_score(y_test, knn_pred_test).round(4)
knn_diff_auc_score = knn_test_auc_score - knn_train_auc_score

print('Training Score', knn_train_score)
print('Testing Score:', knn_test_score)
print('Training AUC Score',knn_train_auc_score)
print('Testing AUC Score:', knn_test_auc_score)



########################
# Creating a confusion matrix
########################

print(confusion_matrix(y_true = y_test,
                       y_pred = knn_pred_test))


# Visualizing a confusion matrix

labels = ['Died', 'Alive']


cm = confusion_matrix(y_true = y_test,
                      y_pred = knn_pred_test)


sns.heatmap(cm,
            annot = True,
            xticklabels = labels,
            yticklabels = labels,
            cmap = 'coolwarm')


plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix of the classifier')
plt.show()


########################
# Creating a classification report
########################

print(classification_report(y_true = y_test,
                            y_pred = knn_pred_test))



# Changing the labels on the classification report
print(classification_report(y_true = y_test,
                            y_pred = knn_pred_test,
                            target_names = labels))



#####################################
# Cross Validation with k-folds
#####################################

# Rule of thumb: observations >= 50*features * folds
# In this case, 1751 >= 50 * 9 * 3


# Cross Validating the knn model with 3 folds

cv_knn_scores = cross_val_score(knn_clf_fit,
                                    GOT_data,
                                    GOT_target,
                                    cv=3, 
                                    scoring = 'roc_auc')
cv_knn_scores_mean = pd.np.mean(cv_knn_scores)






###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
                
                            ## FYI ONLY ##
                      # OTHER MODELS AND INFORMATION #

###############################################################################
###############################################################################
###############################################################################
###############################################################################
 ###############################################################################
###############################################################################
                 
"""
Details on all the other models below so that the analyst reading this will
have a complete overview. 

"""


##########################################################################
# Logistic Regression 
##########################################################################

""" Used to determine if the 9 features selected are signficant with p-values
less than 0.05. 
Answer: all are significant except male and total books, however if we remove
them the predictive ability decreases hence have included in the model for 
further analysis. 

"""

log_selected = smf.logit(formula = """isAlive ~ 
                                        + book4_A_Feast_For_Crows
                                        + total_books
                                        + spouse_group
                                        + h_group
                                        + title_group_2
                                        + pop_m1
                                        + cul_m1
                                        + age_m1
                                        + male_m1
                                        """,
                                       data = GOT_df_train)

results = log_selected.fit()

print(results.summary())

# for popularity : 
np.exp(0.9234)
# for title
np.exp(0.5643)

########################
# Model
########################

# predict model
logreg = LogisticRegression(C = 1)
logreg_fit = logreg.fit(X_train, y_train)
logreg_pred_test = logreg_fit.predict(X_test)
logreg_pred_train = logreg_fit.predict(X_train) 

########################
# Scores
########################

log_train_score = logreg_fit.score(X_train, y_train)
log_test_score = logreg_fit.score(X_test, y_test)
log_diff_score = log_test_score  - log_train_score

print('Training Score', log_train_score)
print('Testing Score:', log_test_score)


log_train_auc_score = roc_auc_score(y_train,logreg_pred_train).round(4)
log_test_auc_score = roc_auc_score(y_test, logreg_pred_test).round(4)
log_diff_auc_score = log_test_auc_score - log_train_auc_score

print('Training AUC Score',log_train_auc_score)
print('Testing AUC Score:', log_test_auc_score)



cv_logistic_scores = cross_val_score(logreg_fit,
                                    GOT_data,
                                    GOT_target,
                                    cv=3, 
                                    scoring = 'roc_auc')

cv_log_scores_mean = pd.np.mean(cv_logistic_scores)


########################
# Creating a confusion matrix
########################

print(confusion_matrix(y_true = y_test,
                       y_pred = logreg_pred_test))


# Visualizing a confusion matrix

labels = ['Died', 'Alive']


cm = confusion_matrix(y_true = y_test,
                      y_pred = logreg_pred_test)


sns.heatmap(cm,
            annot = True,
            xticklabels = labels,
            yticklabels = labels,
            cmap = 'coolwarm')


plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix of the classifier')
plt.show()


########################
# Creating a classification report
########################

print(classification_report(y_true = y_test,
                            y_pred = logreg_pred_test))



# Changing the labels on the classification report
print(classification_report(y_true = y_test,
                            y_pred = logreg_pred_test,
                            target_names = labels))





########################
# Adjusting the hyperparameter C to 100
########################

logreg_100 = LogisticRegression(C = 100
                                )


logreg_100_fit = logreg_100.fit(X_train, y_train)

logreg_pred = logreg_100_fit.predict(X_test)


# Let's compare the testing score to the training score.
print('Training Score', logreg_100_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_100_fit.score(X_test, y_test).round(4))


########################
# Adjusting the hyperparameter C to 0.000001
########################

logreg_000001 = LogisticRegression(C = 0.000001
                                )

logreg_000001_fit = logreg_000001.fit(X_train, y_train)

logreg_pred = logreg_000001_fit.predict(X_test)


# Let's compare the testing score to the training score.
print('Training Score', logreg_000001_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_000001_fit.score(X_test, y_test).round(4))


########################
# Plotting each model's coefficient magnitudes
########################


fig, ax = plt.subplots(figsize=(8, 6))

plt.plot(logreg.coef_.T,
         'o',
         label = "C = 1",
         markersize = 12)

plt.plot(logreg_100.coef_.T,
         '^',
         label = "C = 100",
         markersize = 12)

plt.plot(logreg_000001.coef_.T,
         'v',
         label = "C = 0.000001",
         markersize = 12)


plt.xticks(range(X_train.shape[1]), X_train.columns, rotation=90)
plt.hlines(0, 0, X_train.shape[1])
plt.ylim(-.11, .11)
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.legend()

plt.show()



##############################################################################
# RandomForest
##############################################################################

# Full forest using gini
full_forest_gini = RandomForestClassifier(n_estimators = 500,
                                     criterion = 'gini',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)

# Full forest using entropy
full_forest_entropy = RandomForestClassifier(n_estimators = 500,
                                     criterion = 'entropy',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)


# Fitting the models
full_gini_fit = full_forest_gini.fit(X_train, y_train)
full_entropy_fit = full_forest_entropy.fit(X_train, y_train)


# Check if predictions are the same for both models
pd.DataFrame(full_gini_fit.predict(X_test), full_entropy_fit.predict(X_test))
full_gini_fit.predict(X_test).sum() == full_entropy_fit.predict(X_test).sum()


# Scoring the gini model 
print('Training Score', full_gini_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_gini_fit.score(X_test, y_test).round(4))

# Scoring the entropy model
print('Training Score', full_entropy_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_entropy_fit.score(X_test, y_test).round(4))


# Saving score objects
gini_full_train = full_gini_fit.score(X_train, y_train)
gini_full_test  = full_gini_fit.score(X_test, y_test)

entropy_full_train = full_entropy_fit.score(X_train, y_train)
entropy_full_test  = full_entropy_fit.score(X_test, y_test)


# Feature importance function
########################

def plot_feature_importances(model, train = X_train, export = False):
    fig, ax = plt.subplots(figsize=(12,9))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    
    if export == True:
        plt.savefig('Tree_Leaf_Feature_Importance.png')

########################

plot_feature_importances(full_gini_fit,
                         train = X_train,
                         export = False)



plot_feature_importances(full_entropy_fit,
                         train = X_train,
                         export = False)


################
# Hyperparameter tuning
################

# Creating a hyperparameter grid
depth_space = pd.np.arange(1, 20)
leaf_space = pd.np.arange(1, 10)
split_space = pd.np.arange(0.10, 1.00)

param_grid = {'max_depth' : depth_space,
              'min_samples_leaf' : leaf_space,
              'min_samples_split' : split_space}


# Building the model object one more time
c_tree_4_hp = RandomForestClassifier(random_state = 508)

# Creating a GridSearchCV object
c_tree_4_hp_cv = GridSearchCV(c_tree_4_hp, param_grid, cv = 3)

# Fit it to the training data
c_tree_4_hp_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", c_tree_4_hp_cv.best_params_)
print("Tuned Logistic Regression Accuracy:", 
      c_tree_4_hp_cv.best_score_.round(4))



# Building a tree model object with optimal hyperparameters
c_tree_optimal = DecisionTreeClassifier(criterion = 'gini',
                                        random_state = 508,
                                        max_depth = 6,
                                        min_samples_leaf = 1,
                                        min_samples_split = 0.1)


c_tree_optimal_fit = c_tree_optimal.fit(X_train, y_train)

plot_feature_importances(c_tree_optimal,
                         train = X_train,
                         export = True)


#### 
# Plotting the optimal tree
#### 

dot_data = StringIO()

export_graphviz(decision_tree = c_tree_optimal_fit,
                out_file = dot_data,
                filled = True,
                rounded = True,
                special_characters = True,
                feature_names = X_train.columns)


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png(),
      height = 500,
      width = 800)

# Saving the visualization in the working directory
graph.write_png("GOT_Optimal_Classification_Tree_2.png")

########################
# Scores
########################

c_pred_test = c_tree_optimal_fit.predict(X_test)
c_pred_train = c_tree_optimal_fit.predict(X_train)


c_train_score = c_tree_optimal_fit.score(X_train, y_train).round(4)
c_test_score = c_tree_optimal_fit.score(X_test, y_test).round(4)
c_diff_score = c_test_score  - c_train_score


c_train_auc_score = roc_auc_score(y_train,c_pred_train).round(4)  
c_test_auc_score = roc_auc_score(y_test, c_pred_test).round(4)  
c_diff_auc_score = c_test_auc_score - c_train_auc_score

print('Training Score', c_train_score)
print('Testing Score:', c_test_score)
print('Training AUC Score',c_train_auc_score)
print('Testing AUC Score:', c_test_auc_score)


cv_rf_scores = cross_val_score(c_tree_optimal_fit,
                                    GOT_data,
                                    GOT_target,
                                    cv=3, 
                                    scoring = 'roc_auc')

cv_rf_scores_mean = pd.np.mean(cv_rf_scores)


# Changing the labels on the classification report
print(classification_report(y_true = y_test,
                            y_pred = c_pred_test,
                            target_names = labels))



##############################################################################
# Gradient Boosted Machines
##############################################################################

gbm = GradientBoostingClassifier(loss='deviance',
                                 learning_rate = 1.5,
                                 n_estimators = 100,
                                 max_depth = 3,
                                 criterion = 'friedman_mse',
                                 warm_start = False,
                                 random_state = 508,
                                 )

gbm_fit = gbm.fit(X_train,y_train)
gbm_pred_test = gbm_fit.predict(X_test)
gbm_pred_train = gbm_fit.predict(X_train)


gbm_train_score = gbm_fit.score(X_train, y_train).round(4)
gbm_test_score = gbm_fit.score(X_test, y_test).round(4)
gbm_diff_score = gbm_test_score  - gbm_train_score


gbm_train_auc_score = roc_auc_score(y_train,gbm_pred_train).round(4)  
gbm_test_auc_score = roc_auc_score(y_test, gbm_pred_test).round(4)  
gbm_diff_auc_score = gbm_test_auc_score - gbm_train_auc_score

print('Training Score', gbm_train_score)
print('Testing Score:', gbm_test_score)
print('Training AUC Score',gbm_train_auc_score)
print('Testing AUC Score:', gbm_test_auc_score)


# Changing the labels on the classification report
print(classification_report(y_true = y_test,
                            y_pred = gbm_pred_test,
                            target_names = labels))


########################
# Hyperparameter
########################

# Creating a hyperparameter grid
learn_space = pd.np.arange(0.1, 1.6, 0.1)
estimator_space = pd.np.arange(50, 250, 50)
depth_space = pd.np.arange(1, 10)
criterion_space = ['friedman_mse', 'mse', 'mae']


param_grid = {'learning_rate' : learn_space,
              'max_depth' : depth_space,
              'criterion' : criterion_space,
              'n_estimators' : estimator_space}


# Building the model object one more time
gbm_grid = GradientBoostingClassifier(random_state = 508)

gbm_grid_cv = RandomizedSearchCV(gbm_grid,
                                 param_grid,
                                 cv = 3,
                                 n_iter = 50,
                                 scoring = 'roc_auc') 

gbm_grid_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned GBM Parameter:", gbm_grid_cv.best_params_)
print("Tuned GBM Accuracy:", gbm_grid_cv.best_score_.round(4))


########################
# Building GBM Model Based on Best Parameters
########################

gbm_optimal = GradientBoostingClassifier(criterion = 'friedman_mse',
                                      learning_rate = 0.2,
                                      max_depth = 3,
                                      n_estimators = 150,
                                      random_state = 508,
                                      )


gbm_optimal_fit = gbm_optimal.fit(X_train, y_train)

gbm_optimal_score = gbm_optimal.score(X_test, y_test)

gbm_optimal_pred_test = gbm_optimal.predict(X_test)

gmb_optimal_pred_train = gbm_optimal.predict(X_train)



########################
# Scores
########################
gbm_op_train_score = gbm_optimal.score(X_train, y_train).round(4)
gbm_op_test_score = gbm_optimal.score(X_test, y_test).round(4)
gbm_diff_score = gbm_op_test_score  - gbm_op_train_score


gbm_op_train_auc_score = roc_auc_score(y_train,gmb_optimal_pred_train).round(4)  
gbm_op_test_auc_score = roc_auc_score(y_test, gbm_optimal_pred_test).round(4)  
gbm_diff_auc_score = gbm_test_auc_score - gbm_train_auc_score

print('Training Score', gbm_op_train_score)
print('Testing Score:', gbm_op_test_score)
print('Training AUC Score',gbm_op_train_auc_score)
print('Testing AUC Score:', gbm_op_test_auc_score)


cv_gbm_scores = cross_val_score(gbm_optimal_fit,
                                    GOT_data,
                                    GOT_target,
                                    cv=3, 
                                    scoring = 'roc_auc')

cv_gbm_scores_mean = pd.np.mean(cv_gbm_scores)



# Changing the labels on the classification report
print(classification_report(y_true = y_test,
                            y_pred = gbm_optimal_pred_test,
                            target_names = labels))


##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

                            # Saving Results

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################


# Saving best model scores after cross-validation, scoring method = AUC
                            
model_scores_df = pd.DataFrame({'Log_Score': [cv_log_scores_mean],
                                'KNN_Score': [cv_knn_scores_mean],
                                'RF_Score': [cv_rf_scores_mean],
                                'GBM_Score': [cv_gbm_scores_mean]})


model_scores_df.to_excel("GOT_Ensemble_Model_Results.xlsx")



########################
# FOR KNN BEST MODEL 
########################                            
                            
model_predictions_df = pd.DataFrame({'Actual' : y_test,
                                     'KNN_Predicted': knn_pred_test})


model_predictions_df.to_excel("BestModel_Ensemble_Predictions.xlsx")
                        


