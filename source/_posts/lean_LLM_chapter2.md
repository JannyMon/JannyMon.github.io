---
title: 从头开始构建大语言模型第二章
date: 2024-12-13 22:44:00
categories: [LLM, 文本处理]
tags: [LLM, text]
comments: false
---

# 本章包含
- 为大语言模型训练准备文本数据
- 将文本拆分为词和字的标记
- 字节对编码作为一种更加高级的文本分词方式
- 采用滑动窗口进行训练样本采样
- 将分词转换为向量用来输入大语言模型
&nbsp;&nbsp;&nbsp;&nbsp;到目前位置我们已经介绍了大语言模型的一般架构并且了解了它是在大量的文本上进行预训练的。具体来说，我们将重点研究基于Transformer并仅具有解码器功能的大语言模型，它们是GPT和其他流行的GPT类大模型的基础。
&nbsp;&nbsp;&nbsp;&nbsp;在预训练的过程中，大语言模型逐字处理文本。通过使用下一个词的预测方式来训练数百万到数十亿参数的大语言模型将会是模型产生出色的能力。这一些模型可以进一步微调以遵循一般指令或执行特定的任务。但是在我们实施和训练大语言模型之前，我们需要准备训练数据集，如图2.1中所示。
![2.1](../images/image.png)
你将会续写到如何为训练大语言模型准备输入文本。这包含将文本拆分为单词和字词标记(token),这些标记将会被编码为大语言模型所需要的向量表示。你也会学到像字节对编码这样高级的文本分词方法，他在GPT等流行的大语言模型中被广泛使用。最后我们将实现一种采样和数据加载策略来产生大语言模型训练所需的输入输出对。

## 2.1理解词嵌入
深度神经网络包含大语言模型无法直接处理原始的文本。因为文本时离散的，它与实现和训练神经网络的数学操作不兼容。因此我们需要一种方式将单词表示成连续的向量。
&nbsp;&nbsp;&nbsp;&nbsp;***注意***：对于在计算环境中不熟悉向量和张量的读者，可以在附录A的第A.2.2节了解更多相关信息。
将数据转换为向量格式的概念通常被成为嵌入。通过使用特定的神经网络层或者其他预训练的神经网络模型，我们可以嵌入不同的数据类型比如视频，音频和文本，如图2.2所示。但是值得注意的是不同的数据类型需要特定的嵌入模型。例如为文本设计的嵌入模型就不适用于音频和视频的嵌入。
![2.2](../images/image-1.png)
其核心在于，嵌入是一种将离散对象(如单词，图像甚至整个文档)映射到连续的向量空间中的点的技术，嵌入的主要目的是为了将非数字数据转换为神经网络可以处理的格式。
&nbsp;&nbsp;&nbsp;&nbsp;虽然次嵌入是文本嵌入最常见的形式，但是还有其他的如句子，段落甚至整个文档的嵌入。句子或者段落的嵌入在检索增强型生成中备受欢迎。检索增强型生成结合了生成(如文本生成)和检索(如搜索外部知识库)两个过程，在生成文本时提取相关信息，这是一种超出本书范围的技术。由于我们的目前是训练类似GPT的大语言模型，这些模型学习一次产生一个单词，我们将聚焦在单词嵌入。
&nbsp;&nbsp;&nbsp;&nbsp;为了生存词嵌入已经开发除了一些算法和框架。其中一个比较早且最受欢迎的例子就是Word2Vec方法。Word2Vec通过训练神经网络架构来生成词嵌入，具体做法是通过预测目标词汇的上下文或者反过来根据上下文预测目标词汇。Word2Vec的主要思想是出现在相同上下文中的词汇往往具有相同的含义。因此，当为了可视化目的将词嵌入投影到二维空间时，相似的术语会聚集在一起，如图2.3所示。
&nbsp;&nbsp;&nbsp;&nbsp;词嵌入的维度可以是从一维到数千维度。更高维度可以捕获更多细致入微的关系，但是代价是牺牲计算效率。
![alt text](../images/image2_3.png)
虽然我们可以使用预训练模型如Word2Vec来为机器学习模型产生嵌入，但是大语言模型通常都会产生它自己的嵌入层，这些嵌入层是输入层的一部分，并且在训练中会更新。将嵌入层优化成大语言模型的一部分而不是使用Word2Vec的好处是这些嵌入是针对当前特定的任务和数据进行了优化的。我们会在后续的章节中实现这样的嵌入层。(大语言模型也可以创建上下文化的输出嵌入，如我们在第三章中所述)
&nbsp;&nbsp;&nbsp;&nbsp;不幸的是高维嵌入会带来可视化的挑战，因为我们的感官感知和通常的图形表述本质上局限于三维或者更低的维度，这也是为什么图2.3中只显示二维嵌入的原因。然而，在使用大型语言模型（LLMs）时，我们通常使用维度远高于此的嵌入。对于GPT-2和GPT-3来说，嵌入的大小（通常称为模型隐藏状态的维度）会根据具体的模型变体和大小而有所不同。这是性能和效率之间的权衡。以GPT-2的最小模型（1.17亿和1.25亿参数）为例，它们使用768维的嵌入来提供具体实例。而GPT-3中最大的模型（1750亿参数）则使用12,288维的嵌入。
&nbsp;&nbsp;&nbsp;&nbsp;接下来，我们将逐步介绍为大型语言模型（LLM）准备嵌入所需的步骤，这些步骤包括将文本拆分为单词、将单词转换为标记（token），以及将标记转换为嵌入向量。
## 2.2文本标记化
让我讨论一下如何将文本拆分为单个的标记，这是为大语言模型生成嵌入的一个必须的预处理步骤。这些标记可以是单个的词，或者是包含标点符号在内的特殊字符，如图2.4所示
![alt text](../images/image2_4.png)

您提到要将Edith Wharton的短篇小说《The Verdict》进行分词处理，以用于大型语言模型（LLM）的训练。这篇小说已经处于公有领域，因此可以被用于LLM训练任务。这个文本可以在Wiki下载，https://en.wikisource.org/wiki/The_Verdict，你可以拷贝并且粘贴到一个文本文件，我将它粘贴到了"the-verdict.txt"文本文件。
您提到可以在本书的GitHub仓库，位于https://mng.bz/Adng找到"the-verdict.txt"文本文件。你可以使用如下的python代码下载该文件。
```python
import urllib.request
url = ("https://raw.githubusercontent.com/rasbt/"
 "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
 "the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)
```
接下来，我们可以使用Python的标准文件读取工具来加载"the-verdict.txt"文件。
&nbsp;&nbsp;&nbsp;&nbsp;***代码块 2.1 将一部短篇小说作为文本读入的python示例代码***
```python
with open("the-verdict.txt", "r", encoding="utf-8") as f:
 raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])
```
print命令打印出文件的总字符数以及为了说明目的而展示的前100个字符
```
Total number of character: 20479
I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow
enough--so it was no 
```
我们的目标是将这个20,479个字符的短篇小说进行分词处理，将其拆分成单个的单词和特殊字符，然后可以将这些分词结果转换成用于大型语言模型（LLM）训练的嵌入表示。
&nbsp;&nbsp;&nbsp;&nbsp;***注意***：在使用大型语言模型（LLMs）时，处理数百万篇文章和数十万本书——即数Gb字节的文本——是很常见的。然而，出于教育目的，使用较小的文本样本（如一本书）就足够了，这样可以阐明文本处理步骤背后的主要思想，并使其能够在消费级硬件上以合理的时间运行。
为了将文本拆分成一个标记（token）列表，我们可以采取多种策略。在这里，我们将简要介绍如何使用Python的正则表达式库re来实现这一目标，但请注意，后续我们会转向使用更高级、预构建的分词器。因此，您无需深入学习或记忆正则表达式的具体语法。
&nbsp;&nbsp;&nbsp;&nbsp;借助一段简单的示例文本，我们可以通过re.split命令，并遵循下面的语法规则，根据空白字符（如空格、制表符等）来拆分这段文本。
```python
import re
text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
print(result)
```
结果是一个包含单个单词，空格和特殊字符的列表。
```
['Hello,', ' ', 'world.', ' ', 'This,', ' ', 'is', ' ', 'a', ' ', 'test.']
```
这种简单的分词方案基本上可以将示例文本拆分成单个的单词，但是有些单词仍然与标点符号相连，而我们希望这些标点符号能够作为单独的列表项。此外，我们并不将所有文本都转换为小写，因为大写有助于大型语言模型（LLMs）区分专有名词和普通名词，理解句子结构，并学习生成具有正确大写格式的文本。
&nbsp;&nbsp;&nbsp;&nbsp;让我们修改正则表达式，以便根据空白字符（\s）、逗号（,）和句号（.）来进行拆分：
```python
result = re.split(r'([,.]|\s)', text)
print(result)
```
我们可以看到，单词和标点符号现在已经作为单独的列表项分开了，这正是我们想要的：
```
['Hello', ',', '', ' ', 'world', '.', '', ' ', 'This', ',', '', ' ', 'is',' ', 'a', ' ', 'test', '.', '']
```
剩余的一个小问题是，列表中仍然包含了空白字符。我们可以选择安全地移除这些多余的字符，具体方法如下：
```python
result = [item for item in result if item.strip()]
print(result)
```
去除空格字符的结果输出如下：
```
['Hello', ',', 'world', '.', 'This', ',', 'is', 'a', 'test', '.']
```
&nbsp;&nbsp;&nbsp;&nbsp;***注意***：在开发一个简单的分词器时，是否将空白字符编码为单独的字符还是直接移除它们，这取决于我们的应用程序及其需求。移除空白字符可以减少内存和计算需求。然而，如果我们训练的是对文本精确结构敏感的模型（例如，对缩进和空格敏感的Python代码），那么保留空白字符可能是有用的。在这里，为了简洁和分词输出的简洁性，我们选择移除空白字符。稍后，我们将切换到一种包含空白字符的分词方案。
我们在这里设计的分词方案在简单的示例文本上效果很好。让我们对其进行一些进一步的修改，以便它也能够处理其他类型的标点符号，比如问号、引号，以及我们在埃迪丝·华顿短篇小说前100个字符中看到的双破折号，还有一些其他的特殊字符。
```python
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)
```
输出结果如下：
```
['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']
```
根据图2.5中总结的结果，我们可以看到，我们的分词方案现在已经能够成功地处理文本中的各种特殊字符了。
![alt text](../images/image2_5.png)
现在我们已经有了一个基本的分词器，让我们将其应用于埃迪丝·华顿的整个短篇小说：
```python
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
```
这条打印语句输出了4690，这是该文本中token的数量（不包括空白字符）。为了快速进行视觉检查，我们打印出前30个token：
```
print(preprocessed[:30])
```
输出结果显示，我们的分词器似乎很好地处理了文本，因为所有的单词和特殊字符都被整齐地分开了。
```
['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', 'rather', 'a','cheap', 'genius', '--', 'though', 'a', 'good', 'fellow', 'enough','--', 'so', 'it', 'was', 'no', 'great', 'surprise', 'to', 'me', 'to','hear', 'that', ',', 'in']
```
## 2.3将标记转换为ID
接下来，让我们将这些来自Python字符串的标记转换为整数表示，以生成标记ID。这一转换是将标记ID转换为嵌入向量之前的中间步骤。
&nbsp;&nbsp;&nbsp;&nbsp;为了将之前生成的标记映射到标记ID，我们首先需要构建一个词汇表。这个词汇表定义了如何将每个唯一的单词和特殊字符映射到一个唯一的整数，如图2.6所示。
![alt text](../images/image2_6.png)
既然我们已经将Edith Wharton的短篇小说进行了标记化处理，并将其分配给了一个名为preprocessed的Python变量，接下来我们将创建一个包含所有唯一标记的列表，并按字母顺序对其进行排序，以确定词汇表的大小:
```python
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)
```
在通过代码确定词汇表大小为1,130之后，我们将创建这个词汇表，并为了说明目的，打印出其前51个条目。
&nbsp;&nbsp;&nbsp;&nbsp;***代码块 2.2 创建单词表***
```python
vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
 print(item)
 if i >= 50:
 break
```
代码输出如下:
```
('!', 0)
('"', 1)
("'", 2)
...
('Her', 49)
('Hermia', 50)
```
正如我们所见，该字典包含了与唯一整数标签相关联的单个标记。我们的下一个目标是将此词汇表应用于将新文本转换为标记ID（如图2.7所示）。
![alt text](../images/image2_7.png)
当我们想要将大型语言模型（LLM）的输出从数字转换回文本时，我们需要一种方法将标记ID转换回对应的文本标记。为此，我们可以创建一个词汇表的逆版本，它将标记ID映射回对应的文本标记。
&nbsp;&nbsp;&nbsp;&nbsp;下面是一个完整的Python分词器类的实现，它包含了一个encode方法用于将文本拆分成标记并通过词汇表进行字符串到整数的映射以生成标记ID，以及一个decode方法用于执行反向的整数到字符串的映射以将标记ID转换回文本。以下是该分词器实现的代码：
&nbsp;&nbsp;&nbsp;&nbsp;***代码块 2.3 实现一个简单的分词器***
![alt text](../images/imagelist2_3.png)
使用SimpleTokenizerV1这个Python类，我们现在可以通过一个已存在的词汇表来实例化新的分词器对象，然后利用这个对象对文本进行编码和解码，正如图2.8所展示的那样。
&nbsp;&nbsp;&nbsp;&nbsp;接下来，我们将从SimpleTokenizerV1类中实例化一个新的分词器对象，并对Edith Wharton短篇小说中的一段文字进行分词实践，以检验其效果。
```python
tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
 Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
```
前面的代码会打印出以下标记ID（token IDs）：
```
[1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108,
754, 793, 7]
```
接下来，让我们看看是否可以使用decode方法将这些标记ID转换回文本。
```python
print(tokenizer.decode(ids))
```
![alt text](../images/image2_8.png)
图2.8&nbsp;&nbsp;&nbsp;&nbsp;所示的分词器实现通常包含两个通用方法：encode方法和decode方法。encode方法负责接收输入的样本文本，然后将其拆分成单独的标记（tokens），并通过词汇表将这些标记转换为相应的标记ID。这一过程是文本数据预处理中的重要步骤，特别是在需要将文本输入到机器学习模型时。
上述代码输出如下：
```
'" It\' s the last he painted, you know," Mrs. Gisburn said with
pardonable pride.'
```
基于这个输出结果，我们可以确认decode方法成功地将标记ID转换回了原始文本。
&nbsp;&nbsp;&nbsp;&nbsp;到目前为止，一切顺利。我们已经实现了一个能够根据训练集片段对文本进行分词和解分词的分词器。现在，让我们将其应用于一个未包含在训练集中的新文本样本:
```python
text = "Hello, do you like tea?"
print(tokenizer.encode(text))
```
执行改代码会导致如下错误：
```
KeyError: 'Hello'
```
&nbsp;&nbsp;&nbsp;&nbsp;问题在于，“Hello”这个词并未在“The Verdict”这篇短篇小说中出现，因此它并未被纳入词汇表中。这再次强调了在使用大型语言模型（LLMs）时，为了扩展词汇表，需要采用规模庞大且内容多样的训练数据集。
&nbsp;&nbsp;&nbsp;&nbsp;接下来，我们将进一步测试分词器在处理包含未知词汇的文本时的表现，并探讨在训练大型语言模型（LLM）时，可以使用哪些额外的特殊标记来提供更多的上下文信息。

## 2.4增加特殊的上下文标记