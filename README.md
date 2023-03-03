# 

本文件夹实现的功能是DNN网络的cpp代码，主要包含keras的模型权重读取以及DNN中部分网络的前向推理阶段实现

目前实现功能包括：
- 读取h5模型文件的函数，利用二叉树的层序遍历方法读取权重，并用unordered_map进行权重存储
- 批归一化层
- Conv1D层
- 激活函数（ELU、tanh）
- Bi-LSTM
