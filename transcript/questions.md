#### Q1

What kind of tasks are you planning to support next?

We have image classifiaction in mind.


#### Q2

Can you train a network when you use your implementation as a building block?

Yes, you can train them as long as you keep the computational graph.
In this case, you need to use `__call__` method instead of `predict` method.


#### Q3

What is Faster R-CNN?

This is one of the state of the art network for object detection tasks. It is supported by ChainerCV.


#### Q4

What is the difference between Chainer and ChainerCV?

Chainer is a software library that is targeted to machine learning tasks in general.
On the other hand, ChainerCV focuses on Computer Vision tasks.




#### Notes

##### Chainer's goal
Chainer:  Speed up research and development of deep learning and its applications.
In order to do that, Chainer has focused on debugability and simplicity in implementation as its main goals.
