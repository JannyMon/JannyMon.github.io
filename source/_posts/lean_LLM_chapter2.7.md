---
title: 从头开始构建大语言模型第二章2.7&2.8
date: 2024-12-18 22:44:00
categories: [LLM, 文本处理]
tags: [LLM, Text,Token,Embedding,BPE]
comments: false
---
### 2.7创建标记嵌入
&nbsp;&nbsp;&nbsp;&nbsp;为LLM（大型语言模型）训练准备输入文本的最后一步是将令牌ID转换为嵌入向量，如图2.15所示。作为初步步骤，我们必须使用随机值初始化这些嵌入权重。这种初始化是LLM学习过程的起点。在第五章中，我们将作为LLM训练的一部分来优化这些嵌入权重。

&nbsp;&nbsp;&nbsp;&nbsp;由于GPT等LLM是使用反向传播算法训练的深度神经网络，因此需要使用连续的向量表示，即嵌入。

![alt text](../images/image2_15.png)
&nbsp;&nbsp;&nbsp;&nbsp;***图2.15展示了文本准备的三个步骤：文本分词、将文本令牌转换为令牌ID，以及将令牌ID转换为嵌入向量。在这里，我们利用之前生成的令牌ID来创建令牌嵌入向量。***

&nbsp;&nbsp;&nbsp;&nbsp;***注意：如果您对如何使用反向传播算法训练神经网络不熟悉，请阅读附录A中的B.4部分。***

&nbsp;&nbsp;&nbsp;&nbsp;让我们通过一个实际操作示例来看看如何将令牌ID转换为嵌入向量。假设我们有以下四个输入令牌，其ID分别为2、3、5和1：

```python
input_ids = torch.tensor([2, 3, 5, 1])
```

&nbsp;&nbsp;&nbsp;&nbsp;为了简化说明，假设我们有一个仅包含6个单词的小型词汇表（而不是BPE分词器词汇表中的50,257个单词），并且我们希望创建大小为3的嵌入（在GPT-3中，嵌入大小为12,288维）：

```python
vocab_size = 6
output_dim = 3
```
&nbsp;&nbsp;&nbsp;&nbsp;使用vocab_size和output_dim，我们可以在PyTorch中实例化一个嵌入层，并为了可重复性设置随机种子为123：

```python
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
```
&nbsp;&nbsp;&nbsp;&nbsp;print语句会打印出嵌入层的底层权重矩阵：

```
Parameter containing:
tensor([[ 0.3374, -0.1778, -0.1690],
        [ 0.9178, 1.5810, 1.3010],
        [ 1.2753, -0.2010, -0.1606],
        [-0.4015, 0.9666, -1.1481],
        [-1.1589, 0.3255, -0.6315],
        [-2.8400, -0.7849, -1.4096]], requires_grad=True)
```
&nbsp;&nbsp;&nbsp;&nbsp;嵌入层的权重矩阵包含小的随机值。这些值在LLM训练过程中作为LLM优化本身的一部分进行优化。此外，我们可以看到权重矩阵有6行和3列。词汇表中的每个可能的令牌都有一行，每个嵌入维度都有一列。

&nbsp;&nbsp;&nbsp;&nbsp;现在，让我们将其应用于一个令牌ID以获得嵌入向量：

```python
print(embedding_layer(torch.tensor([3])))
```

&nbsp;&nbsp;&nbsp;&nbsp;此代码将输出与令牌ID 3相对应的嵌入向量:

```
tensor([[-0.4015, 0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>)
```

&nbsp;&nbsp;&nbsp;&nbsp;如果我们比较令牌ID 3的嵌入向量与之前的嵌入矩阵，我们会发现它与第四行（由于Python的索引从0开始，因此它对应于索引3的行）完全相同。换句话说，嵌入层本质上是一个查找操作，它通过令牌ID从嵌入层的权重矩阵中检索行。

&nbsp;&nbsp;&nbsp;&nbsp;***注意：对于那些熟悉one-hot编码的人来说，这里描述的嵌入层方法本质上只是实现one-hot编码后在全连接层中进行矩阵乘法的一种更高效的方式。这在GitHub上的补充代码中得到了说明，网址为：https://mng.bz/ZEB5。由于嵌入层只是one-hot编码和矩阵乘法方法的一种更高效实现，因此它可以被视为一个可以通过反向传播优化的神经网络层。***

&nbsp;&nbsp;&nbsp;&nbsp;我们已经了解了如何将单个令牌ID转换为三维嵌入向量。现在，让我们将所有四个输入ID（torch.tensor([2, 3, 5, 1])）都应用这个转换：

```python
print(embedding_layer(input_ids))
```
&nbsp;&nbsp;&nbsp;&nbsp;打印输出显示，这会产生一个4×3的矩阵：

```
tensor([[ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-2.8400, -0.7849, -1.4096],
        [ 0.9178,  1.5810,  1.3010]], grad_fn=<EmbeddingBackward0>)
```
在这个输出矩阵中，每一行都是通过从嵌入权重矩阵中查找得到的，如图2.16所示（注意：图2.16在此文本环境中无法直接展示）。
&nbsp;&nbsp;&nbsp;&nbsp;现在我们已经从令牌ID创建了嵌入向量，接下来我们将对这些嵌入向量进行一个小修改，以编码文本中令牌的位置信息。


### 2.8 文本位置信息编码
&nbsp;&nbsp;&nbsp;&nbsp;原则上，令牌嵌入（Token Embeddings）是大型语言模型（LLM）的合适输入。然而，大型语言模型的一个微小缺陷是它们的自注意力机制（见第3章）没有序列中令牌的位置或顺序的概念。之前介绍的嵌入层的工作方式是，无论令牌ID在输入序列中的位置如何，相同的令牌ID总是被映射到相同的向量表示，如图2.17所示。


![alt text](../images/image2_16.png)
&nbsp;&nbsp;&nbsp;&nbsp;***图2.16解释：嵌入层执行查找操作，从嵌入层的权重矩阵中检索与令牌ID对应的嵌入向量。例如，令牌ID为5的嵌入向量是嵌入层权重矩阵的第六行（之所以是第六行而不是第五行，是因为Python的索引是从0开始的）。我们假设这些令牌ID是由第2.3节中的小词汇表生成的。***

![alt text](../images/image2_17.png)