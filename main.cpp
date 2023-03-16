#include "_public.h"
/*
	代码逻辑主要分为两块，第一块为加载各层权重数据并实例化各层，第二块为前向推理过程
	为了方便调试，暂时未对代码做进一步封装
*/

const size_t OneSymbol_size = 64;
const size_t NumOfCarriers = 64;

int main() {
	
	// 从h5文件中读取模型的参数
	unordered_map<string, vector<param_t>> model_params;
	vector<string> layers_name = { "batch_normalization","batch_normalization_1","batch_normalization_2","batch_normalization_3","batch_normalization_4","batch_normalization_5","batch_normalization_6","conv1d","conv1d_1" };
	GetLayerWeights("./model/BPSK_iciv1_snr12_model1.h5", layers_name, model_params);
	
	// 读取信号
	string input_data_path = "./input_data/input_data.txt";
	vector<data_t> input_data;
	read_data_from_file<data_t>(input_data_path, input_data);
	
	// 读入信道
	string input_channelA_path = "./input_data/input_channelA.txt";
	vector<data_t> input_channelA;
	read_data_from_file<data_t>(input_channelA_path, input_channelA);

	string input_channelB_path = "./input_data/input_channelB.txt";
	vector<data_t> input_channelB;
	read_data_from_file<data_t>(input_channelB_path, input_channelB);



	/*****************************************************************************************************************************************************************************************************/
	/*****************************************************************  实例化模型的各层，预载各层的参数  ************************************************************************************************/

	/**********************************************  实例化BN0  *****************************************************************/
	//获取bn层的权重参数
	vector<param_t> bn0_weights(std::move(model_params["batch_normalization"]));
	assert(bn0_weights.size() % 4 == 0);
	auto n0_params = bn0_weights.size() / 4;

	// 使用emplace构造对象，减少一次拷贝
	vector<param_t> beta; beta.reserve(n0_params);
	vector<param_t> gamma; gamma.reserve(n0_params);
	vector<param_t> moving_mean; moving_mean.reserve(n0_params);
	vector<param_t> moving_variance; moving_variance.reserve(n0_params);

	for (size_t i = 0; i < n0_params; i++) {
		beta.emplace_back(std::forward<param_t>(bn0_weights[i]));
		gamma.emplace_back(std::forward<param_t>(bn0_weights[i + n0_params]));
		moving_mean.emplace_back(std::forward<param_t>(bn0_weights[i + n0_params * 2]));
		moving_variance.emplace_back(std::forward<param_t>(bn0_weights[i + n0_params * 3]));
	}

	// 实例化对象
	unique_ptr<BN> bn0(new BN());
	bn0->init(std::move(beta), std::move(gamma), std::move(moving_mean), std::move(moving_variance));
	


	/**********************************************  实例化BN1  *****************************************************************/
	//获取bn层的权重参数
	vector<param_t> bn1_weights(std::move(model_params["batch_normalization_1"]));
	assert(bn1_weights.size() % 4 == 0);
	auto n1_params = bn1_weights.size() / 4;


	beta.clear();
	gamma.clear();
	moving_mean.clear();
	moving_variance.clear();

	// 前文使用了右值引用进行了传递，因此这里可以接着赋值
	for (size_t i = 0; i < n1_params; i++) {
		beta.emplace_back(std::forward<param_t>(bn1_weights[i]));
		gamma.emplace_back(std::forward<param_t>(bn1_weights[i + n1_params]));
		moving_mean.emplace_back(std::forward<param_t>(bn1_weights[i + n1_params * 2]));
		moving_variance.emplace_back(std::forward<param_t>(bn1_weights[i + n1_params * 3]));
	}

	// 实例化对象
	unique_ptr<BN> bn1(new BN());
	bn1->init(std::move(beta), std::move(gamma), std::move(moving_mean), std::move(moving_variance));



	/**********************************************  实例化BN2  *****************************************************************/
	//获取bn层的权重参数
	vector<param_t> bn2_weights(std::move(model_params["batch_normalization_2"]));
	assert(bn2_weights.size() % 4 == 0);
	auto n2_params = bn2_weights.size() / 4;

	beta.clear();
	gamma.clear();
	moving_mean.clear();
	moving_variance.clear();

	// 前文使用了右值引用进行了传递，因此这里可以接着赋值
	for (size_t i = 0; i < n2_params; i++) {
		beta.emplace_back(std::forward<param_t>(bn2_weights[i]));
		gamma.emplace_back(std::forward<param_t>(bn2_weights[i + n2_params]));
		moving_mean.emplace_back(std::forward<param_t>(bn2_weights[i + n2_params * 2]));
		moving_variance.emplace_back(std::forward<param_t>(bn2_weights[i + n2_params * 3]));
	}

	// 实例化对象
	unique_ptr<BN> bn2(new BN());
	bn2->init(std::move(beta), std::move(gamma), std::move(moving_mean), std::move(moving_variance));

	/**********************************************  实例化BN3  *****************************************************************/
	//获取bn层的权重参数
	vector<param_t> bn3_weights(std::move(model_params["batch_normalization_3"]));
	assert(bn3_weights.size() % 4 == 0);
	auto n3_params = bn3_weights.size() / 4;

	beta.clear();
	gamma.clear();
	moving_mean.clear();
	moving_variance.clear();

	// 前文使用了右值引用进行了传递，因此这里可以接着赋值
	for (size_t i = 0; i < n3_params; i++) {
		beta.emplace_back(std::forward<param_t>(bn3_weights[i]));
		gamma.emplace_back(std::forward<param_t>(bn3_weights[i + n3_params]));
		moving_mean.emplace_back(std::forward<param_t>(bn3_weights[i + n3_params * 2]));
		moving_variance.emplace_back(std::forward<param_t>(bn3_weights[i + n3_params * 3]));
	}

	// 实例化对象
	unique_ptr<BN> bn3(new BN());
	bn3->init(std::move(beta), std::move(gamma), std::move(moving_mean), std::move(moving_variance));


	/**********************************************  实例化BN4  *****************************************************************/
	//获取bn层的权重参数
	vector<param_t> bn4_weights(std::move(model_params["batch_normalization_4"]));
	assert(bn4_weights.size() % 4 == 0);
	auto n4_params = bn4_weights.size() / 4;

	beta.clear();
	gamma.clear();
	moving_mean.clear();
	moving_variance.clear();

	// 前文使用了右值引用进行了传递，因此这里可以接着赋值
	for (size_t i = 0; i < n4_params; i++) {
		beta.emplace_back(std::forward<param_t>(bn4_weights[i]));
		gamma.emplace_back(std::forward<param_t>(bn4_weights[i + n4_params]));
		moving_mean.emplace_back(std::forward<param_t>(bn4_weights[i + n4_params * 2]));
		moving_variance.emplace_back(std::forward<param_t>(bn4_weights[i + n4_params * 3]));
	}

	// 实例化对象
	unique_ptr<BN> bn4(new BN());
	bn4->init(std::move(beta), std::move(gamma), std::move(moving_mean), std::move(moving_variance));


	/**********************************************  实例化BN5  *****************************************************************/
	//获取bn层的权重参数
	vector<param_t> bn5_weights(std::move(model_params["batch_normalization_5"]));
	assert(bn5_weights.size() % 4 == 0);
	auto n5_params = bn5_weights.size() / 4;

	beta.clear();
	gamma.clear();
	moving_mean.clear();
	moving_variance.clear();

	// 前文使用了右值引用进行了传递，因此这里可以接着赋值
	for (size_t i = 0; i < n5_params; i++) {
		beta.emplace_back(std::forward<param_t>(bn5_weights[i]));
		gamma.emplace_back(std::forward<param_t>(bn5_weights[i + n5_params]));
		moving_mean.emplace_back(std::forward<param_t>(bn5_weights[i + n5_params * 2]));
		moving_variance.emplace_back(std::forward<param_t>(bn5_weights[i + n5_params * 3]));
	}

	// 实例化对象
	unique_ptr<BN> bn5(new BN());
	bn5->init(std::move(beta), std::move(gamma), std::move(moving_mean), std::move(moving_variance));


	/**********************************************  实例化BN6  *****************************************************************/
	vector<param_t> bn6_weights(std::move(model_params["batch_normalization_6"]));
	assert(bn6_weights.size() % 4 == 0);
	auto n6_params = bn6_weights.size() / 4;

	beta.clear();
	gamma.clear();
	moving_mean.clear();
	moving_variance.clear();

	// 前文使用了右值引用进行了传递，因此这里可以接着赋值
	for (size_t i = 0; i < n6_params; i++) {
		beta.emplace_back(std::forward<param_t>(bn6_weights[i]));
		gamma.emplace_back(std::forward<param_t>(bn6_weights[i + n6_params]));
		moving_mean.emplace_back(std::forward<param_t>(bn6_weights[i + n6_params * 2]));
		moving_variance.emplace_back(std::forward<param_t>(bn6_weights[i + n6_params * 3]));
	}

	// 实例化对象
	unique_ptr<BN> bn6(new BN());
	bn6->init(std::move(beta), std::move(gamma), std::move(moving_mean), std::move(moving_variance));



	/**********************************************  实例化Conv1D_0  *****************************************************************/
	
	// 提前定义好conv1d的参数，conv1d_0的参数为：layers.Conv1D(32,1, padding='same')
	const size_t conv1d0_filters = 32;
	const size_t conv1d0_kernel_size = 1;
	const size_t conv1d0_step = 1;
	bool conv1d0_padding = true;

	// 通过上述参数设置可以得到权重中bias和kenel矩阵的大小
	const size_t input_data_width = 6;	// 输入矩阵的宽度

	vector<param_t> conv_bias; conv_bias.reserve(conv1d0_filters);
	vector<param_t> conv_kernel; conv_kernel.reserve(conv1d0_kernel_size * conv1d0_filters * input_data_width);
	
	//获取Conv1d层的权重参数
	vector<param_t> conv1d0_weights(std::move(model_params["conv1d"]));
	
	assert(conv1d0_weights.size() == conv1d0_filters + conv1d0_kernel_size * conv1d0_filters * input_data_width);
	for (size_t i = 0; i < conv1d0_filters; i++) {
		conv_bias.emplace_back(std::forward<data_t>(conv1d0_weights[i]));
	}
	for (size_t i = 0; i < conv1d0_filters; i++) {
		for (size_t k = 0; k < conv1d0_kernel_size; k++) {
			for (size_t j = 0; j < input_data_width; j++) {
				conv_kernel.emplace_back(std::forward<data_t>(conv1d0_weights[k* conv1d0_filters* input_data_width + conv1d0_filters + j * conv1d0_filters + i]));
			}
		}
	}

	
	// 实例化对象
	unique_ptr<Conv1d> conv1d0(new Conv1d(conv1d0_filters, conv1d0_kernel_size, conv1d0_step, conv1d0_padding));
	conv1d0->init(std::move(conv_bias), std::move(conv_kernel));
	
	/*****************************************************************************************************************************************************************************************************/
	/*****************************************************************************************************************************************************************************************************/

	
	
	/*****************************************************************************************************************************************************************************************************/
	/*************************************************************************************  前向推理  ****************************************************************************************************/
	
	size_t SymbolNum = input_data.size() / (OneSymbol_size * 2);
	vector<data_t> OneSymbol_input_data; OneSymbol_input_data.resize(OneSymbol_size * 2);					// 存储单个符号的叠加信号数据
	vector<data_t> OneSymbol_input_channelA; OneSymbol_input_channelA.resize(OneSymbol_size * 2);			// 存储单个符号对应的信道A
	vector<data_t> OneSymbol_input_channelB; OneSymbol_input_channelB.resize(OneSymbol_size * 2);			// 处理单个符号对应的信道B

	

	for (size_t i = 0; i < SymbolNum; i++) {
		// 每次取出一个符号（包含64个子载波，由于数据是复数，将实部和虚部进行拆分后就变为128个数据）
		
		vec_copy(input_data, input_data.begin() + i * (OneSymbol_size * 2), (OneSymbol_size * 2), OneSymbol_input_data);
		vec_copy(input_channelA, input_channelA.begin() + i * (OneSymbol_size * 2), (OneSymbol_size * 2), OneSymbol_input_channelA);
		vec_copy(input_channelB, input_channelB.begin() + i * (OneSymbol_size * 2), (OneSymbol_size * 2), OneSymbol_input_channelB);
		ShowVec(OneSymbol_input_data);

		// BN0层的前向推理
		vector<data_t> bn0_output;
		bn0->bn_forward(OneSymbol_input_data, bn0_output);

		// BN1层的前向推理
		vector<data_t> bn1_output;
		bn1->bn_forward(OneSymbol_input_channelA, bn1_output);

		// BN2层的前向推理
		vector<data_t> bn2_output;
		bn2->bn_forward(OneSymbol_input_channelB, bn2_output);

		// reshape0层和Permute0层一起
		vector<vector<data_t>> permute0_output(NumOfCarriers, vector<data_t>(2, 0));
		// reshape1层和Permute1层一起
		vector<vector<data_t>> permute1_output(NumOfCarriers, vector<data_t>(2, 0));
		// reshape2层和Permute2层一起
		vector<vector<data_t>> permute2_output(NumOfCarriers, vector<data_t>(2, 0));


		for (size_t i = 0; i < NumOfCarriers; i++) {
			for (size_t j = 0; j < 2; j++) {
				permute0_output[i][j] = bn0_output[j * NumOfCarriers + i];
				permute1_output[i][j] = bn1_output[j * NumOfCarriers + i];
				permute2_output[i][j] = bn2_output[j * NumOfCarriers + i];
			}
		}

		// concatenate层
		vector<vector<data_t>> concatenate_output = Concatence(permute0_output, permute1_output, permute2_output);
		


		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/*  这一部分是残差块，对应于下面这段函数，后续可以对其进行封装，目前由于debug需要暂时不封装成函数  
			def residual_block1(input,filters,kernel_size):
				F1,F2,F3 = filters
				x = input
				x = layers.Conv1D(F1, 1, padding='same')(x)
				#x = layers.Dropout(0.25)(x)
				x = layers.BatchNormalization()(x)
				x = layers.Activation('ELU')(x)
				x = layers.Conv1D(F2, kernel_size, padding='same')(x)
				#x = layers.Dropout(0.25)(x)
				x = layers.BatchNormalization()(x)
				x = layers.Activation('ELU')(x)
				x = layers.Conv1D(F3, 1, padding='same')(x)
				#x = layers.Dropout(0.25)(x)
				x = layers.BatchNormalization()(x)
				# 捷径部分
				residual = input
				residual_output = layers.Add()([residual, x])
				residual_output = layers.Activation('ELU')(residual_output)
				return residual_output
		*/
		/**************************************** 连接部分 *************************************/
		// conv1d_0层
		vector<vector<data_t>> conv1d0_output = conv1d0->conv1d_forward(concatenate_output);
		// bn3层
		vector<vector<data_t>> bn3_output;
		for (size_t row = 0; row < conv1d0_output.size(); row++) {
			vector<data_t> bn3_output_tmp;
			bn3->bn_forward(conv1d0_output[row], bn3_output_tmp);
			bn3_output.push_back(bn3_output_tmp);
		}

		/**************************************** 捷径部分 *************************************/


		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


	}
	/***********************************************************************************************************************************************/
	/***********************************************************************************************************************************************/
	return 0;

	

}