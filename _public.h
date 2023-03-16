#ifndef _PUBLIC_H
#define _PUBLIC_H
#include <stdint.h>
#include <hdf5.h>
#include <H5Cpp.h>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <queue>
#include <fstream>
#include <sstream>
#include <assert.h>
using namespace H5;
using namespace std;

#ifdef _DEBUG
#pragma comment(lib, "hdf5_D.lib")
#pragma comment(lib, "hdf5_cpp_D.lib")
#else
#pragma comment(lib, "hdf5.lib")
#pragma comment(lib, "hdf5_cpp.lib")
#endif


typedef float d_type;
typedef int   d_size;

typedef float data_t; // 数据类型
typedef float param_t; // 参数类型


/*
	功能：将字符串转变为数组
*/
template <typename T>
T String2Num(const string& str)
{
	istringstream iss(str);
	T num;
	iss >> num;
	return num;
}


/*
	功能：从txt文件中读取数据
*/
template <typename T>
void read_data_from_file(string path, vector<T>& data) {
	ifstream infile;
	infile.open(path, ifstream::in);
	string line;
	if (!infile.is_open()) {
		cout << "Can not find" << path << endl;
		system("pause");
	}
	while (getline(infile, line)) {
		stringstream ss(line);
		string tmp;
		while (getline(ss, tmp, ' ')) {
			if (tmp != " ") {
				T data_tmp = String2Num<T>(tmp);
				data.push_back(data_tmp);
				
			}

		}
	}
}

/*
	功能：展示vector中的数据
*/
void ShowVec(const vector<data_t>& vec);


/*
	功能：检查两组数据结果是否一致
*/
int check_error_num(vector<int> a, vector<int> b);

/*
	功能：获取指定层的权重参数
	参数：
		filename			读取的hdf5文件
		layers_name			指定的layer名称
		model_data			读取的layer权重参数
*/
bool GetLayerWeights(const string& filename, vector<string>& layers_name, unordered_map<string, vector<float>>& model_data);



/*
	功能：从一个容器中拷贝部分数据到另外一个容器
*/
void vec_copy(const vector<data_t>& data1, const vector<data_t>::iterator begin, size_t n, vector<data_t>& data2);


/*
	功能：reshape层的功能，将一维vector变为二维vector
*/
template<typename T>
void reshape(const vector<T>& input, vector<T>& output, int row, int col ) {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			output[i][j] = input[i * col + j];
		}
	}
}


/*
	定义BN层的类
*/
struct bn_params {
	param_t epsilon = 0.001;
	vector<param_t>	beta;
	vector<param_t>	gamma;
	vector<param_t>	moving_mean;
	vector<param_t> moving_variance;
};
class BN {
public:
	explicit BN(): params(make_shared<bn_params>()) {};

	template<class F, std::enable_if_t<std::is_same_v<std::decay_t<F>, vector<param_t>>, int> = 0>
	void init(F&& beta, F&& gamma, F&& moving_mean, F&& moving_variance) {
		params->beta = std::forward<F>(beta);
		params->gamma = std::forward<F>(gamma);
		params->moving_mean = std::forward<F>(moving_mean);
		params->moving_variance = std::forward<F>(moving_variance);
	}
	void bn_forward(const vector<data_t>& input_data, vector<data_t>& output_data); // 运行
	void ShowWeights();			// 展示权重参数，一般用于debug检查
private:
	std::shared_ptr<bn_params> params;

};


/*
	定义concatence层，利用递归的方式合并输入对象
*/
template<class F>
vector<vector<F>> Concatence(const vector<vector<F>>& vec) {
	return vec;
}
template<class F, class ...Args>
vector<vector<F>> Concatence(const vector<vector<F>>& vec1, Args ...args) {
	vector<vector<F>> result(vec1);
	auto next = Concatence(args...);
	size_t sizeOfresult = result.size();
	size_t sizeOfnext = next.size();
	assert(sizeOfresult == sizeOfnext);	// 确保所有的对象长度一致
	for (size_t i = 0; i < sizeOfresult; i++) {
		result[i].insert(result[i].end(), next[i].begin(), next[i].end());
	}
	return result;
}


/*
	定义Conv1D类
		filters: filters的个数
		kernel_size: kernel的大小
		step：步长
		默认padding为same
*/
struct conv1d_params {
	size_t filters;
	size_t kernel_size;
	size_t steps;
	bool padding;
	vector<param_t>	bias;
	vector<param_t>	kernel;
};
class Conv1d {
public:
	explicit Conv1d(size_t filters, size_t kernel_size, size_t step, bool padding=true) :params(make_shared<conv1d_params>()) {
		params->filters = filters;
		params->kernel_size = kernel_size;
		params->steps = step;
		params->padding = padding;
	}
	template<class F, std::enable_if_t<std::is_same_v<std::decay_t<F>, vector<param_t>>, int> = 0>
	void init(F&& bias, F&& kernel) {
		params->bias = std::forward<F>(bias);
		params->kernel  = std::forward<F>(kernel);
	}

	vector<vector<data_t>> conv1d_forward(const vector<vector<data_t>>& input_data); // 运行



private:
	std::shared_ptr<conv1d_params> params;

};




/*
	定义BiLSTM（待整理为类）
*/
template<int INPUT_SIZE, int LSTM_SIZE, typename T>
void lstm(T input[INPUT_SIZE], T h_pre[LSTM_SIZE], T c_pre[LSTM_SIZE], T BIAS[LSTM_SIZE * 4], T W_X[INPUT_SIZE * LSTM_SIZE * 4], T W_H[LSTM_SIZE * LSTM_SIZE * 4]) {
	T gates_i[LSTM_SIZE] = { 0 };
	T gates_f[LSTM_SIZE] = { 0 };
	T gates_g[LSTM_SIZE] = { 0 };
	T gates_o[LSTM_SIZE] = { 0 };

	T acc_x[LSTM_SIZE * 4] = { 0 };
	T acc_h[LSTM_SIZE * 4] = { 0 };

	T gates_i_activ[LSTM_SIZE] = { 0 };
	T gates_f_activ[LSTM_SIZE] = { 0 };
	T gates_g_activ[LSTM_SIZE] = { 0 };
	T gates_o_activ[LSTM_SIZE] = { 0 };

	T c_pre_activ[LSTM_SIZE] = { 0 };
	for (int i = 0; i < INPUT_SIZE; i++) {
		//#pragma HLS unroll
		for (int j = 0; j < LSTM_SIZE * 4; j++) {

			acc_x[j] += input[i] * W_X[i * LSTM_SIZE * 4 + j];

		}
	}


	for (int i = 0; i < LSTM_SIZE; i++) {

		for (int j = 0; j < LSTM_SIZE * 4; j++) {
			acc_h[j] += h_pre[i] * W_H[i * LSTM_SIZE * 4 + j];

		}
	}


	for (int igate = 0; igate < LSTM_SIZE; igate++) {
		//#pragma HLS PIPELINE
		gates_i[igate] = acc_x[igate] + acc_h[igate] + BIAS[igate];
		gates_f[igate] = acc_x[LSTM_SIZE + igate] + acc_h[LSTM_SIZE + igate] + BIAS[LSTM_SIZE + igate];
		gates_g[igate] = acc_x[2 * LSTM_SIZE + igate] + acc_h[2 * LSTM_SIZE + igate] + BIAS[2 * LSTM_SIZE + igate];
		gates_o[igate] = acc_x[3 * LSTM_SIZE + igate] + acc_h[3 * LSTM_SIZE + igate] + BIAS[3 * LSTM_SIZE + igate];
	}


	sigmoid_LUT<T, LSTM_SIZE, 1024>(gates_i, gates_i_activ);
	sigmoid_LUT<T, LSTM_SIZE, 1024>(gates_f, gates_f_activ);
	tanh_LUT<T, LSTM_SIZE, 1024>(gates_g, gates_g_activ);
	sigmoid_LUT<T, LSTM_SIZE, 1024>(gates_o, gates_o_activ);

	/*
	for (int i = 0; i < LSTM_SIZE; i++) {
		gates_i_activ[i] = sigmoid<T>(gates_i[i]);
	}
	for (int i = 0; i < LSTM_SIZE; i++) {
		gates_f_activ[i] = sigmoid<T>(gates_f[i]);
	}
	for (int i = 0; i < LSTM_SIZE; i++) {
		gates_g_activ[i] = tanh_T<T>(gates_g[i]);
	}
	for (int i = 0; i < LSTM_SIZE; i++) {
		gates_o_activ[i] = sigmoid<T>(gates_o[i]);
	}
	*/
	for (int i = 0; i < LSTM_SIZE; i++) {
		c_pre[i] = gates_f_activ[i] * c_pre[i] + gates_i_activ[i] * gates_g_activ[i];
	}
	/*
	for (int i = 0; i < LSTM_SIZE; i++) {
		c_pre_activ[i] = tanh_T<T>(c_pre[i]);
	}*/
	tanh_LUT<T, LSTM_SIZE, 1024>(c_pre, c_pre_activ);

	for (int i = 0; i < LSTM_SIZE; i++) {
		h_pre[i] = gates_o_activ[i] * c_pre_activ[i];
	}
}
template <typename T, int TIME_STEP, int INPUT_SIZE, int LSTM_SIZE>
void bilstm(
	T input[TIME_STEP * INPUT_SIZE],
	T output[TIME_STEP * LSTM_SIZE],
	T weights_x[4 * INPUT_SIZE * LSTM_SIZE],
	T weights_h[LSTM_SIZE * LSTM_SIZE * 4],
	T bias[LSTM_SIZE * 4]
) {

	T  c_pre[LSTM_SIZE] = { 0 };
	T  h_pre[LSTM_SIZE] = { 0 };
	T* in = input;
	for (int its = 0; its < TIME_STEP; its++) {
		in = input + its * INPUT_SIZE;
		lstm<INPUT_SIZE, LSTM_SIZE, T>(in, h_pre, c_pre, bias, weights_x, weights_h);
		for (int i = 0; i < LSTM_SIZE; i++) {
			*(output + its * LSTM_SIZE + i) = *(h_pre + i);
		}
	}
}





/*
	定义激活函数（待整理为类）
*/
template<typename T>
T sigmoid(T x) {
	float x_float = x;
	float output = 1.0 / (1.0 + exp(-x_float));
	return (T)output;
}
template<typename T>
T relu(T x) {
	return (x > 0 ? (T)x : (T)0);
}
template<typename T>
T tanh_T(T x) {
	return (T)tanh((float)x);
}

template<typename T, int N_TABLE>
void init_sigmoid_table(T table_out[N_TABLE]) {
	for (int ii = 0; ii < N_TABLE; ii++) {
		float in_val = 2 * 8 * (ii - float(N_TABLE) / 2) / float(N_TABLE);
		T real_val = sigmoid<T>(in_val);
		table_out[ii] = real_val;
	}
}


template<typename T, int SIZE_IN, int TABLE_SIZE>
void  sigmoid_LUT(T data[SIZE_IN], T res[SIZE_IN])
{
	static bool initialized_sigmoid = false;
	static T sigmoid_table[TABLE_SIZE];
	if (!initialized_sigmoid) {
		init_sigmoid_table<T, TABLE_SIZE>(sigmoid_table);
		initialized_sigmoid = true;
	}
	// Index into the lookup table based on data
	int data_round;
	int index;
	for (int ii = 0; ii < SIZE_IN; ii++) {
		data_round = data[ii] * TABLE_SIZE / 16;
		index = data_round + 8 * TABLE_SIZE / 16;
		if (index < 0)   index = 0;
		if (index > TABLE_SIZE - 1) index = TABLE_SIZE - 1;
		res[ii] = (T)sigmoid_table[index];
	}
}

template<typename T, int N_TABLE>
void init_tanh_table(T table_out[N_TABLE])
{
	// Implement tanh lookup
	for (int ii = 0; ii < N_TABLE; ii++) {
		// First, convert from table index to X-value (signed 8-bit, range -4 to +4)
		float in_val = 2 * 8.0 * (ii - float(N_TABLE) / 2.0) / float(N_TABLE);
		// Next, compute lookup table function
		T real_val = tanh(in_val);
		//std::cout << "Tanh:  Lookup table Index: " <<  ii<< " In Value: " << in_val << " Result: " << real_val << std::endl;
		table_out[ii] = real_val;
	}
}

template<typename T, int SIZE_IN, int TABLE_SIZE>
void  tanh_LUT(T data[SIZE_IN], T res[SIZE_IN])
{
	static bool initialized_tanh = false;
	static T tanh_table[TABLE_SIZE];
	if (!initialized_tanh) {
		init_tanh_table<T, TABLE_SIZE>(tanh_table);
		initialized_tanh = true;
	}

	// Index into the lookup table based on data
	int data_round;
	int index;
	for (int ii = 0; ii < SIZE_IN; ii++) {
		data_round = data[ii] * TABLE_SIZE / 16;
		index = data_round + 8 * TABLE_SIZE / 16;
		//std::cout << "Input: "  << data[ii] << " Round: " << data_round << " Index: " << index << std::endl;
		if (index < 0)   index = -1;
		if (index > TABLE_SIZE - 1) index = TABLE_SIZE - 1;
		res[ii] = tanh_table[index];
	}
}




#endif