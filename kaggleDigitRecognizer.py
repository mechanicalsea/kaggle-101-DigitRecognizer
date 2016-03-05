'''
================
Digit Recognizer
----------------
    数字识别
================
[Introducation]

The data files train.csv and test.csv contain gray-scale
images of hand-drawn digits, from zero through nine.

Each image is 28 pixels in height and 28 pixels in width,
for a total of 784 pixels in total. Each pixel has a single
pixel-value associated with it, indicating the lightness
or darkness of that pixel, with higher numbers meaning darker.
This pixel-value is an integer between 0 and 255, inclusive.

The training data set, (train.csv), has 785 columns. The first
column, called "label", is the digit that was drawn by the user.
The rest of the columns contain the pixel-values of the associated
image.

Each pixel column in the training set has a name like pixelx,
where x is an integer between 0 and 783, inclusive. To locate
this pixel on the image, suppose that we have decomposed x as
x = i * 28 + j, where i and j are integers between 0 and 27,
inclusive. Then pixelx is located on row i and column j of a
28 x 28 matrix, (indexing by zero).

For example, pixel31 indicates the pixel that is in the fourth
column from the left, and the second row from the top, as in
the ascii-diagram below.

Visually, if we omit the "pixel" prefix, the pixels make up
the image like this:

< Input Data:
    000 001 002 003 ... 026 027
    028 029 030 031 ... 054 055
    056 057 058 059 ... 082 083
     |   |   |   |  ...  |   |
    728 729 730 731 ... 754 755
    756 757 758 759 ... 782 783
>

The test data set, (test.csv), is the same as the training set,
except that it does not contain the "label" column.

Your submission file should be in the following format: For each
of the 28000 images in the test set, output a single line with
the digit you predict. For example, if you predict that the first
image is of a 3, the second image is of a 7, and the third image
is of a 8, then your submission file would look like:

< Output Data
    3
    7
    8
    (27997 more lines)
>

The evaluation metric for this contest is the categorization accuracy,
or the proportion of test images that are correctly classified. For
example, a categorization accuracy of 0.97 indicates that you have
correctly classified all but 3% of the images.
'''

#print(__doc__)

import csv
import numpy as np
from numpy import ravel
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import datasets,  metrics
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, IncrementalPCA

################################################################################
# 数据读取模块

# 读取 csv 数据 train
def csvTrainDataRead():
    print("Load train data...")
    features, labels = [], []
    with open('train.csv') as myCSV:
        reader = csv.reader(myCSV)
        for index, row in enumerate(reader):
            if index > 0:
                labels.append(row[:1])
                features.append(row[1:])
            
    # list -> array
    features = np.float_(features)
    labels = ravel(np.int_(labels))
    '''
    plt.figure(figsize=(6, 5))
    for i, comp in enumerate(features):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape((28, 28)), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('100 components extracted by RBM', fontsize=16)
    plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

    plt.show()
    '''
    return features, labels

# 读取 csv 数据 test
def csvTestDataRead():
    print("Load test data...")
    features = []
    with open('test.csv') as myCSV:
        reader = csv.reader(myCSV)
        for index, row in enumerate(reader):
            if index > 0:
                features.append(row)
    # list -> array.float
    features = np.float_(features)

    return features

################################################################################
# csv 预测结果存储
def csvResultDataSave(result, csvName):
    print('数据存储中...')
    ids = np.arange(1,28001)
    with open(csvName, 'w') as myCSV:
        myWriter = csv.writer(myCSV)
        myWriter.writerow(["ImageId", "Label"])
        myWriter.writerows(zip(ids, result))

################################################################################
# 分类模块

# 分类
def classificationDigit(X, Y, testX):

    print("维度归约...")

    ''' 
    # 降维 IPCA
    # IPCA + KNN[Accuracy=0.84614]
    # IPCA + RandomForest[Accuracy=0.83843]
    ipca = IncrementalPCA(n_components=90, batch_size=100)
    ipca.fit(X)
    
    X = ipca.fit_transform(X)
    testX = ipca.fit_transform(testX)
    
    print('explained variance ratio : %s, %f'
      % (str(ipca.explained_variance_ratio_), \
         min(ipca.explained_variance_ratio_)))

    '''
    
    # 分类器 RandomForest[Accuracy=0.96443]
    classifier = RandomForestClassifier(n_estimators=100)
    print('开始训练...')
    # 训练
    classifier.fit(X, Y)
    print('开始预测...')
    predicted = classifier.predict(testX)

    return predicted


################################################################################
# 测试对比模块
def compareResult(result):
    print("测试中...[1:1000]")
    labels = []
    with open('rf_benchmark.csv') as myCSV:
        reader = csv.reader(myCSV)
        for index, row in enumerate(reader):
            if index > 0:
                labels.append(row[1:])
            if index > 1000:
                break
            
    # list -> array
    labels = ravel(np.int_(labels))

    ans = np.sum((labels[:1000]-result[:1000])==0)/float(len(labels[:1000]))
    
    print("Accuracy is %f\n" % ans)
    

################################################################################
# 主模块

trainX, trainY = csvTrainDataRead()
testX = csvTestDataRead()
result = classificationDigit(trainX, trainY, testX)
csvResultDataSave(result, 'rfc.csv')
compareResult(result)

