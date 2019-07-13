# DenseNet
@[toc]
## 简介
- 从2012年AlexNet大展身手以来，卷积神经网络经历了（LeNet、）AlexNet、ZFNet、VGGNet、GoogLeNet（借鉴Network in Network）、ResNet、DenseNet的大致发展路线。其实，自从ResNet提出之后，ResNet的变种网络层出不穷，各有特点，性能都略有提高。
- 在这种情况下，DenseNet可以说是“继往开来”也不为过，作为2017年CVPR最佳论文，DenseNet思想上部分借鉴了“前辈”ResNet，但是采用的确实完全不同的结构，结构上并不复杂却十分有效，在CIFAR数据集上全面超越了ResNet。
- 本项目着重实现使用Keras搭建DenseNet的网络结构，同时，利用其在数据集上进行效果评测。
## 网络说明
- 参考论文
  - 官网地址
    - [Arxiv](https://arxiv.org/abs/1608.06993)
  - 项目另附
    - [PDF文件](/asset/1608.06993.pdf)
- 设计背景
  - 以往的卷积神经网络提高效果的方法，要么更深（如ResNet，解决了深层网络出现的梯度消失问题），要么更宽（如GoogLeNet的Inception结构），而DenseNet的作者**从feature着手，通过对feature的极致利用来达到更好的效果同时减少参数**。
  - 为了解决梯度消失（vanishing-gradient）问题，很多论文及结构被提出如ResNet、Highway Networks、Stochastic depth等，尽管网络结构有所差异，但是都不难改变一个核心思路**从靠前的层到靠后的层之间创建直通路线**如之前提到的[ResNet中的shortcut](https://blog.csdn.net/zhouchen1998/article/details/94651438)。DenseNet的作者延续这个思路，提出在保证网络中层与层之间最大程度的信息传输的前提下，直接将所有层连接起来。
- 网络优点
  - 减轻了梯度消失（vanishing-gradient）
  - 大大加强了feature的传递
  - 更加深入利用了feature
  - 一定程度上减少了参数数量