NB15数据集下载地址：https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15

使用UNSW-NB15全部数据，将所有文件：
UNSW-NB15_1.csv
UNSW-NB15_2.csv
UNSW-NB15_3.csv
UNSW-NB15_4.csv
放置于此文件夹下

并执行：预处理全文件版.py
将会将所有数据合并，预处理并以50%/50%的比例生成：
kaggle_UNSW_NB15_full_training.csv
kaggle_UNSW_NB15_full_testing.csv
最后生成的数据共有204维度

然后执行：数据剪裁.py
修改测试集比例分布，降低正常分类数据
最后生成
kaggle_UNSW_NB15_full_edit_testing.csv



预处理kaggle版.py用于处理已经划分好的预设训练集和测试集
UNSW_NB15_training.csv
UNSW_NB15_testing.csv
将其处理为具有58维的数据。在本实验中暂不使用