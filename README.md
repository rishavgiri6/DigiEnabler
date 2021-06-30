# DigiEnabler

A Handwritten Digit Recognition Model Using scikit-learn

#Author-Rishav Giri

Recognizing handwritten text is a problem that can be traced back to the first automatic machines that needed to recognize individual characters in handwritten documents. Think about, for example, the ZIP codes on letters at the post office and the automation needed to recognize these five digits. Perfect recognition of these codes is necessary in order to sort mail automatically and efficiently. Included among the other applications that may come to mind is OCR (Optical Character Recognition) software. OCR software must read the handwritten text, or pages of printed books, for general electronic documents in which each character is well defined.

Hypothesis :
The Digits data set of the Scikit-learn library provides numerous data-sets that are useful for testing many problems of data analysis and prediction of the results. Some Scientist claims that it predicts the digit accurately 95% of the times. Perform data Analysis to accept or reject this Hypothesis.

Dataset :

In this project, we are using the Handwritten Digits dataset which is already ready in the sklearn library. we can import the dataset using the below code.
from sklearn import datasets
digits = datasets.load_digits()
Digits dataset is a dictionary that contains data, targets, images, features names, description of the dataset, target names, etc.
We focus mainly on data and targets. We extract both on different variables.
main_data = digits['data']
targets = digits['target']
Now we can see our data look.

def view_digit(index):
    plt.imshow(digits.images[index] , cmap = plt.cm.gray_r , interpolation = 'nearest')
    plt.title('Orignal it is: '+ str(digits.target[index]))
    plt.show()
view_digit(17)

#Model Planning:

To see how different models work on different data sizes we are using 3 models Support vector Classifier, Decision Tree Classifier, Random Forest Classifier.

![image](https://user-images.githubusercontent.com/26641017/123914599-3196ee80-d99d-11eb-93fe-bbf38d74cd7a.png)

#1.Support Vector Classifier :
The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points.


![image](https://user-images.githubusercontent.com/26641017/123914568-293eb380-d99d-11eb-875a-f0acd5d78a6d.png)

![image](https://user-images.githubusercontent.com/26641017/123914638-3a87c000-d99d-11eb-8c57-c22df9337bdc.png)

#2. Decision Tree Classifier :
Decision Tree Classifier is a simple and widely used classification technique. It applies a straightforward idea to solve the classification problem. Decision Tree Classifier poses a series of carefully crafted questions about the attributes of the test record. Each time it receives an answer, a follow-up question is asked until a conclusion about the class label of the record is reached.

![image](https://user-images.githubusercontent.com/26641017/123914462-0c09e500-d99d-11eb-96f6-7c6e1d62c0e3.png)


![image](https://user-images.githubusercontent.com/26641017/123914676-45daeb80-d99d-11eb-84b3-f9d5fe090d66.png)

#3. Random Forest Classifier :

Random forests is a supervised learning algorithm. It can be used both for classification and regression. It is also the most flexible and easy-to-use algorithm. A forest is comprised of trees. It is said that the more trees it has, the more robust a forest is. Random forests create decision trees on randomly selected data samples, get a prediction from each tree, and selects the best solution by means of voting. It also provides a pretty good indicator of the feature's importance.

![image](https://user-images.githubusercontent.com/26641017/123914492-13c98980-d99d-11eb-8515-e9ff2658ba51.png)

#4. Conclusion :
As per our hypothesis, we can say with hyperparameter tunning with different machine learning models or using more data we can achieve near 95% accuracy on the handwritten dataset. But make sure we also have a good amount of test data otherwise the model will get overfit.
