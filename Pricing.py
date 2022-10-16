
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import shapiro
import scipy.stats as stats
import statsmodels.stats.api as sms
import itertools
import warnings
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
warnings.filterwarnings("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


pd.set_option("display.max_columns", None)
pd.options.display.float_format = '{:.4f}'.format

pricing= pd.read_csv("../DataSets/pricing.csv", sep=";")
df = pricing.copy()
sms.DescrStatsW(df["price"]).tconfint_mean()
df["price"].mean()


# Descriptive Statistics

def general(dataframe):
    print('Dataset :', '\n')
    print(dataframe.head() ,'\n')
    print("Dataset Shape:" ,dataframe.shape ,'\n')
    print('Feature Data Types:', '\n')
    print(dataframe.info() ,'\n')
    print('Value Counts:', '\n')
    print(dataframe.category_id.value_counts() ,'\n')
    print("Number of Na values in Dataset:", dataframe.isna().any().sum() ,'\n')
    print('Unique Ids:', '\n')
    print(dataframe.category_id.unique())

general(df)




# Creation of Analysis Report

def analysis(dataframe ,category ,target ,alpha):
    AB =pd.DataFrame()
    combin =list(itertools.combinations(df.category_id.unique() ,2))
    print("-" *20 ,"Group Comparisons | Alpha Confidence Coefficient :", alpha ,"-" *20,)
    for i in range(0 ,len(combin)):
        grA = dataframe[dataframe[category]==combin[i][0]][target]
        grB = dataframe[dataframe[category] == combin[i][1]][target]

        # Tests
        # Normality Assumption
        normA =shapiro(grA)[1] < alpha
        normB = shapiro(grB)[1] < alpha

        # H0 : Series are normally distributed. H0 > 0.05
        # H0: Series are not normally distributed. H0 > 0.05

        if (normA==False) & (normB==False):
            # Both series are normally distributed. We can use Levene Test.
            # Levene Test : Are variances homogeneous ?

            levene = stats.levene(grA,grB)[1] < alpha

            # H0: Variances are homogeneous. H0 > 0.05
            # H1: Variances are not homogeneous. H0 < 0.05

            if levene == False:
                # Variances are homogeneous.

                ttest= stats.ttest_ind(grA ,grB ,equal_var=True)[1]
                # Ho: M1=M2 There is no difference between. Ho > 0.05
                # H1: M1!=M2 There is difference between. Ho < 0.05

            else:
                # Variances are not homogeneous. (Welch Test)

                ttest = stats.ttest_ind(grA, grB, equal_var=False)[1]
                # Ho: M1=M2 There is no difference between. Ho > 0.05
                # H1: M1!=M2 There is difference between. Ho < 0.05

        else: # At least one of the distributions is not normal. (Non-parametric test)

            ttest= stats.mannwhitneyu(grA,grB)[1]
            # Ho: M1=M2 There is no difference between. Ho > 0.05
            # H1: M1!=M2 There is difference between. Ho < 0.05

        # Result:

        temp =pd.DataFrame({"Group Comparison" : [ttest <alpha],
                           "p-value" : ttest,
                           "GroupA Mean": [grA.mean()] ,"GroupB Mean": [grB.mean()],
                           "GroupA Median": [grA.median()] ,"GroupB Median": [grB.median()],
                           "GroupA Count": [grA.count()] ,"GroupB Count": [grB.count()]} ,index=[combin[i]])

        temp["Group Comparison" ] =np.where(temp["Group Comparison" ]==True ,"Has Difference" ,"No Difference")
        temp["Test Type" ] =np.where((normA==False ) &(normB==False) ,"Parametric" ,"Non-Parametric")

        AB =pd.concat \
            ([AB ,temp[["Test Type" ,"Group Comparison" ,"p-value" ,"GroupA Mean" ,"GroupB Mean" ,"GroupA Median",
                               "GroupB Median" ,"GroupA Count" ,"GroupB Count"]]])

    return AB


# Report Results

# 1. Does the Item's Price Different According to the Categories?

AB = analysis(df ,"category_id" ,"price" ,0.05)
AB


# A) Groups With a  Difference Between

AB[AB["Group Comparison"]=="Has Difference"]


# B) Groups With a No Difference Between

AB[AB["Group Comparison"]=="No Difference"]

# As the groups do not meet the assumption of normality,
# we should intervene in outliers observations to normalize.

# OUTLIERS

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.05)
    quartile3 = dataframe[variable].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def has_outliers(dataframe, num_col_names, plot=False):
    variable_names = []
    for col in num_col_names:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]
            print("Total Data Size:",dataframe.shape[0])
            print(col, ":", number_of_outliers, "outlier value")
            variable_names.append(col)
            if plot:
                sns.boxplot(x=dataframe[col])
                plt.show()
    return variable_names

has_outliers(df, ["price"])

# Total Data Size: 3448
# price : 77 outlier value

def remove_outliers(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    df_without_outliers = dataframe[~((dataframe[variable] < low_limit) | (dataframe[variable] > up_limit))]
    return df_without_outliers

df1 = remove_outliers(df,"price")
df1.shape # New Data Size: 3371


# Report Results After Outliers Deletion

AB = analysis(df1 ,"category_id" ,"price" ,0.05)
AB

# The categories are still not distributed normally, but after removing the outlier observations,
# the averages between the categories were composed of values close to each other.

# Groups with a statistically significant difference between Ho <0.05

AB[AB["Group Comparison"]=="Has Difference"]

# Groups with no statistically significant difference between Ho <0.05

AB[AB["Group Comparison"]=="No Difference"]

# Comparison

Has_Difference=AB[AB["Group Comparison"]=="Has Difference"].index
Has_Difference =pd.DataFrame({"Different": Has_Difference})
Has_Difference

# Categories that differ from others : 489756, 326584

No_Difference=AB[AB["Group Comparison"]=="No Difference"].index
No_Difference=pd.DataFrame({"No Difference": No_Difference})
No_Difference

# Similar categories among themselves : 361254, 874521, 675201, 201436

# Q : 1. Does the Item's Price Different According to the Categories?
# A : As can be seen, the price of the item varies according to the categories.

# 2. What Should Be The Item Price ?

df1[['price']].describe().T
df1.groupby(["category_id"]).agg({"price":["mean","median","min","max","count"]})
df1["category_id"].value_counts()

### A) Price Catalog for the Same Price for All Categories

def pricing(dataframe, category="category_id", target="price", method="median"):
    total = 0
    if method == "mean":
        for i in dataframe[category].unique():
            total = total + dataframe[dataframe[category] == i][target].mean()
    if method == "median":
        for i in dataframe[category].unique():
            total = total + dataframe[dataframe[category] == i][target].median()
    return (total / dataframe[category].nunique())



def price_list(dataframe,target,list): # (Price list for all categories)

    lower, upper = sms.DescrStatsW(dataframe[target]).tconfint_mean()
    mid = (lower + upper) / 2
    mean = pricing(dataframe, method="mean")
    median = pricing(dataframe, method="median")
    list.append(lower)
    list.append(mid)
    list.append(upper)
    list.append(mean)
    list.append(median)
    return list

sameprice_allcategory = []
price_list(df1, "price",sameprice_allcategory)

# Lower  : 39.7838594960083
# Mid    : 40.398651625339035
# Upper  : 41.01344375466977
# Mean   : 37.94444677595247
# Median : 33.837631079516676

def revenue_upd(dataframe,target,th):
    frequency = len(dataframe[dataframe[target]>=th])
    revenue = frequency * th #gelir hesabı
    return revenue


# SAME PRICE SIMULATION FOR ALL CATEGORIES

def sale_simulation(price):
    count=1
    for th in price:
        print(count,"Situation | %.4f" %th ,"revenue forecast for sale price")
        x = revenue_upd(df1,"price",th)
        print(x)
        print("\n")
        count += 1

sale_simulation(sameprice_allcategory)

# Result :
#
# The best income was obtained from the sales made by taking the median averages(33.8376) of the categories.
# 5 Situation; 33.8376 revenue forecast for sale price 68013.63846982853.

# The best sales according to the confidence interval; The lowest limit of the confidence interval of the prices of
# the categories that do not differ was the sales made with the sales price.
# 1 Situation; 39.7839 revenue forecast for sale price 34532.390042535204



### B) Same Price For Cateories That "No Difference", And Different For "Has Difference"

# Groups that no statistically difference between
No_Difference

df_36= df1[df1["category_id"]==361254]
df_87= df1[df1["category_id"]==874521]
df_67= df1[df1["category_id"]==675201]
df_20= df1[df1["category_id"]==201436]

group_no_difference = pd.concat([df_36, df_87, df_67, df_20], axis=0, sort=False)
group_no_difference["category_id"].value_counts()



list_no_difference=[]   ## price catalog for "No Difference"
price_list(group_no_difference,"price",list_no_difference)

# Lower  : 36.7109597897918
# Mid    : 37.443361392032315
# Upper  : 38.17576299427283
# Mean   : 37.09238177238653
# Median : 33.9801643426125

df_32 = df1[df1["category_id"] == 326584]
df_32_list = []  # price catolog for Category 326584
price_list(df_32,"price",df_32_list)

# Lower  : 33.709933231454606
# Mid    : 35.693170414655555
# Upper  : 37.6764075978565
# Mean   : 35.69317041465555
# Median : 31.706022729350003

df_48 = df1[df1["category_id"] == 489756]
df_48_list = []  # price catolog for Category 489756
price_list(df_48,"price",df_48_list)

# Lower  : 42.6003695637611
# Mid    : 43.60398315151312
# Upper  : 44.60759673926514
# Mean   : 43.60398315151315
# Median : 35.3991063773


revenue_no_difference=[]
for th in list_no_difference:
    revenue1 = revenue_upd(group_no_difference,"price",th)
    revenue_no_difference.append(revenue1)
revenue_no_difference = np.array(revenue_no_difference)

revenue_32 = []
for th in df_32_list:
    revenue2 = revenue_upd(df_32,"price",th)
    revenue_32.append(revenue2)
revenue_32 = np.array(revenue_32)

revenue_48 = []
for th in df_48_list:
    revenue3 = revenue_upd(df_48,"price",th)
    revenue_48.append(revenue3)
revenue_48 = np.array(revenue_48)

all_revenues = revenue_no_difference + revenue_32 + revenue_48
all_revenues

def sale_show(price):
    count=1
    for i in range(0,5):
        print(count,"Situation | Revenue forecast")
        print(all_revenues[i])
        print("\n")
        count += 1


sale_show(all_revenues)

# 1 Situation | Revenue forecast 34417.4571408421
# 2 Situation | Revenue forecast 31887.914347797694
# 3 Situation | Revenue forecast 30389.26833955753
# 4 Situation | Revenue forecast 32345.67480126304
# 5 Situation | Revenue forecast 59968.14988503372

# Result :
#
# The best income was obtained from the sales made by taking the median averages of the categories.
# 5 Situation (59968.1498)

# The best sales according to the confidence interval; The lowest limit of the confidence interval of the prices of
# the categories. 1 Situation (34417.4571408421)


### C) Price determination based on "No Difference"

# In this case, a price catalog was selected from the values of the categories that were not statistically different
# from each other, and these price configurations were applied to the different categories.

last_step = []
price_list(group_no_difference,"price",last_step)

# Lower  : 36.7109597897918
# Mid    : 37.443361392032315
# Upper  : 38.17576299427283
# Mean   : 37.09238177238653
# Median : 33.9801643426125


sale_simulation(last_step)

## Result

# The confidence interval was determined according to the prices of the groups that did not differ statistically between
# and the entire price policy was made according to the values of these categories.

# As a result best revenue : sales made by averaging the medians of the categories that did not differ. (67076.84441232)

# The best sales according to the confidence interval; The lowest limit of the confidence interval of the prices
# of the non-different categories was sales made with the sales price. (38436.374899912014)

## SUMMARY

# Within the Confidence Interval,
# the highest revenues in all scenarios were obtained from the lower bound point of the confidence level.
# Highest revenue: Income for the lower limit price of the 95% Confidence Interval: 38436.375 - Selling Price: 36.71096