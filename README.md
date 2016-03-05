# kaggle-101-DigitRecognizer
Kaggle 平台上练习——手写数字识别。基于随机森林的分类系统，同时测试了 IPCA 对其的影响。
================
Digit Recognizer
----------------
    数字识别
================


尝试了 K阶近邻、决策树、随机森林等方法，最终最好的结果是

    RandomForestClassifier(n_estimators=100), Accuracy = 0.96443

当然，主成分分析也被用来降维提高效率，不过结果并不理想。

测试结果如下：

1）KNN：Accuracy = 0.83886
    KNN 算法在预测过程中花费的时间很长

2）IPCA + KNN：Accuracy = 0.84614
    IPCA 降维可能会遇到超出内存的问题
    注：测试程序的电脑内存是 8GB

3）IPCA + RandomForest：Accuracy = 0.84614
    RandomForest 随机森林的效率显然比 KNN 高很多，训练与预测效果都非常好

4）RandomForest：Accuracy = 0.96443
    随机森林是 4 种算法中效果最好的，但同时从论坛(Forum)中得到的信息，
    深度学习中的卷积神经网络能达到 Accuracy = 0.99+ 的效果
