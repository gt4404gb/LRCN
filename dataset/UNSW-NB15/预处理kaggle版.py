from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np

#————————————————————————————数据集加载————————————————————————————————
x_train = pd.read_csv("UNSW_NB15_training.csv",  low_memory=False)
x_test = pd.read_csv("UNSW_NB15_testing.csv",  low_memory=False)

# 删除第一列和最后一列
x_train = x_train.drop(x_train.columns[0], axis=1)  # 删除第一列
x_train = x_train.iloc[:, :-1]  # 删除最后一列

x_test = x_test.drop(x_test.columns[0], axis=1)  # 删除第一列
x_test = x_test.iloc[:, :-1]  # 删除最后一列

# 提取最后一列为y
y_train = x_train.iloc[:, -1]
y_test = x_test.iloc[:, -1]
# 定义文本类别和对应的数字编码
class_mapping = {
    'Normal': 0,
    'Analysis': 1,
    'Backdoor': 2,
    'DoS': 3,
    'Exploits': 4,
    'Fuzzers': 5,
    'Generic': 6,
    'Reconnaissance': 7,
    'Shellcode': 8,
    'Worms': 9
}

# 将文本类别替换为数字编码
y_train_encoded = y_train.replace(class_mapping)
y_test_encoded = y_test.replace(class_mapping)

# 从x中删除最后一列
x_train = x_train.iloc[:, :-1]
x_test = x_test.iloc[:, :-1]

#--------------kaggle预处理--------------

df_numeric = x_train.select_dtypes(include=[np.number])
df_numeric.describe(include='all')
#Apply Clamping
#The extreme values should be pruned to reduce the skewness of some distributions. The logic applied here is that the features with a maximum value more than ten times the median value is pruned to the 95th percentile. If the 95th percentile is close to the maximum, then the tail has more interesting information than what we want to discard.
#The clamping is also only applied to features with a maximum of more than 10 times the median. This prevents the bimodals and small value distributions from being excessively pruned.
DEBUG =0
for feature in df_numeric.columns:
    if DEBUG == 1:
        print(feature)
        print('max = '+str(df_numeric[feature].max()))
        print('75th = '+str(df_numeric[feature].quantile(0.95)))
        print('median = '+str(df_numeric[feature].median()))
        print(df_numeric[feature].max()>10*df_numeric[feature].median())
        print('----------------------------------------------------')
    if df_numeric[feature].max()>10*df_numeric[feature].median() and df_numeric[feature].max()>10 :
        x_train[feature] = np.where(x_train[feature]<x_train[feature].quantile(0.95), x_train[feature], x_train[feature].quantile(0.95))
df_numeric = x_train.select_dtypes(include=[np.number])
df_numeric.describe(include='all')

#Apply log function to nearly all numeric, since they are all mostly skewed to the right
#It would have been too much of a slog to apply the log function individually, therefore a simple rule has been set up: if the number of unique values in the continuous feature is more than 50 then apply the log function. The reason more than 50 unique values are sought is to filter out the integer based features that act more categorically.
df_numeric = x_train.select_dtypes(include=[np.number])
df_before = df_numeric.copy()
DEBUG = 0
for feature in df_numeric.columns:
    if DEBUG == 1:
        print(feature)
        print('nunique = '+str(df_numeric[feature].nunique()))
        print(df_numeric[feature].nunique()>50)
        print('----------------------------------------------------')
    if df_numeric[feature].nunique()>50:
        if df_numeric[feature].min()==0:
            x_train[feature] = np.log(x_train[feature]+1)
        else:
            x_train[feature] = np.log(x_train[feature])

df_numeric = x_train.select_dtypes(include=[np.number])
#Reduce the labels in catagorical features
#Some features have very high cardinalities, and this section reduces the cardinality to 5 or 6 per feature. The logic is to take the top 5 occuring labels in the feature as the labels and set the remainder to '-' (seldom used) labels. When the encoding is done later on, the dimensionality will not explode and cause the curse of dimensionality.
df_cat = x_train.select_dtypes(exclude=[np.number])
df_cat.describe(include='all')
DEBUG = 0
for feature in df_cat.columns:
    if DEBUG == 1:
        print(feature)
        print('nunique = ' + str(df_cat[feature].nunique()))
        print(df_cat[feature].nunique() > 6)
        print(sum(x_train[feature].isin(x_train[feature].value_counts().head().index)))
        print('----------------------------------------------------')

    if df_cat[feature].nunique() > 6:
        x_train[feature] = np.where(x_train[feature].isin(x_train[feature].value_counts().head().index), x_train[feature], '-')
df_cat = x_train.select_dtypes(exclude=[np.number])
df_cat.describe(include='all')
x_train['proto'].value_counts().head().index
x_train['proto'].value_counts().index

#-----test处理-----
df_numeric = x_test.select_dtypes(include=[np.number])
df_numeric.describe(include='all')
#Apply Clamping
#The extreme values should be pruned to reduce the skewness of some distributions. The logic applied here is that the features with a maximum value more than ten times the median value is pruned to the 95th percentile. If the 95th percentile is close to the maximum, then the tail has more interesting information than what we want to discard.
#The clamping is also only applied to features with a maximum of more than 10 times the median. This prevents the bimodals and small value distributions from being excessively pruned.
DEBUG =0
for feature in df_numeric.columns:
    if DEBUG == 1:
        print(feature)
        print('max = '+str(df_numeric[feature].max()))
        print('75th = '+str(df_numeric[feature].quantile(0.95)))
        print('median = '+str(df_numeric[feature].median()))
        print(df_numeric[feature].max()>10*df_numeric[feature].median())
        print('----------------------------------------------------')
    if df_numeric[feature].max()>10*df_numeric[feature].median() and df_numeric[feature].max()>10 :
        x_test[feature] = np.where(x_test[feature]<x_test[feature].quantile(0.95), x_test[feature], x_test[feature].quantile(0.95))
df_numeric = x_test.select_dtypes(include=[np.number])
df_numeric.describe(include='all')

#Apply log function to nearly all numeric, since they are all mostly skewed to the right
#It would have been too much of a slog to apply the log function individually, therefore a simple rule has been set up: if the number of unique values in the continuous feature is more than 50 then apply the log function. The reason more than 50 unique values are sought is to filter out the integer based features that act more categorically.
df_numeric = x_test.select_dtypes(include=[np.number])
df_before = df_numeric.copy()
DEBUG = 0
for feature in df_numeric.columns:
    if DEBUG == 1:
        print(feature)
        print('nunique = '+str(df_numeric[feature].nunique()))
        print(df_numeric[feature].nunique()>50)
        print('----------------------------------------------------')
    if df_numeric[feature].nunique()>50:
        if df_numeric[feature].min()==0:
            x_test[feature] = np.log(x_test[feature]+1)
        else:
            x_test[feature] = np.log(x_test[feature])

df_numeric = x_test.select_dtypes(include=[np.number])
#Reduce the labels in catagorical features
#Some features have very high cardinalities, and this section reduces the cardinality to 5 or 6 per feature. The logic is to take the top 5 occuring labels in the feature as the labels and set the remainder to '-' (seldom used) labels. When the encoding is done later on, the dimensionality will not explode and cause the curse of dimensionality.
df_cat = x_test.select_dtypes(exclude=[np.number])
df_cat.describe(include='all')
DEBUG = 0
for feature in df_cat.columns:
    if DEBUG == 1:
        print(feature)
        print('nunique = ' + str(df_cat[feature].nunique()))
        print(df_cat[feature].nunique() > 6)
        print(sum(x_test[feature].isin(x_test[feature].value_counts().head().index)))
        print('----------------------------------------------------')

    if df_cat[feature].nunique() > 6:
        x_test[feature] = np.where(x_test[feature].isin(x_test[feature].value_counts().head().index), x_test[feature], '-')
df_cat = x_test.select_dtypes(exclude=[np.number])
df_cat.describe(include='all')
x_test['proto'].value_counts().head().index
x_test['proto'].value_counts().index


#--------------kaggle预处理--------------

# 将第无用的2，3，4列提取出来
train_labels = x_train.iloc[:, 1:4]
test_labels = x_test.iloc[:, 1:4]

all_labels = pd.concat([train_labels, test_labels])
all_labels_encoded = pd.get_dummies(all_labels)

train_labels_encoded = all_labels_encoded[:len(x_train)]
test_labels_encoded = all_labels_encoded[len(x_train):]



# 从x_train和x_test中删除原有的第2、3和4列，并在最后加入onehot编码后的值
columns_to_drop = [1, 2, 3]
x_train = x_train.drop(x_train.columns[columns_to_drop], axis=1)
x_test = x_test.drop(x_test.columns[columns_to_drop], axis=1)

x_train = pd.concat([x_train, train_labels_encoded], axis=1)
x_test = pd.concat([x_test, test_labels_encoded], axis=1)


scaler = MinMaxScaler()
transfer = StandardScaler()
# 对训练数据进行归一化处理
x_train_encoded = x_train
x_test_encoded = x_test
#对前39列标准化处理，因为提取出3列并放在了最后，所以最后应该是42-3=39
x_train_encoded.iloc[:, :39] = transfer.fit_transform(x_train.iloc[:, :39])
x_train_encoded = scaler.fit_transform(x_train_encoded)

# 对测试数据进行归一化处理
x_test_encoded.iloc[:, :39] = transfer.transform(x_test.iloc[:, :39])
x_test_encoded = scaler.transform(x_test_encoded)
x_train_encoded = pd.DataFrame(x_train_encoded, columns=x_train.columns)
x_test_encoded = pd.DataFrame(x_test_encoded, columns=x_test.columns)

#加回标签到最后一列
x_train_encoded = pd.concat([x_train_encoded, y_train_encoded], axis=1)
x_test_encoded = pd.concat([x_test_encoded, y_test_encoded], axis=1)

# 保存归一化和标准化结果为CSV文件
x_train_encoded.to_csv("kaggle_UNSW_NB15_training.csv", index=False)
x_test_encoded.to_csv("kaggle_UNSW_NB15_testing.csv", index=False)