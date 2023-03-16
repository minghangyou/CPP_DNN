# 

本文件夹实现的功能是DNN网络的cpp代码，主要包含keras的模型权重读取以及DNN中部分网络的前向推理阶段实现

网络实现的代码在_public.h与_public.cpp中

目前实现功能包括：
- 读取h5模型文件的存储并储存在unordered_map中方便后续调用
- 批归一化层
- Conv1D层
- 激活函数（Sigmoid、tanh、relu）
- Bi-LSTM
