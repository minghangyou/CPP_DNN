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
	定义激活函数类
		filters: filters的个数
		kernel_size: kernel的大小
		step：步长
		默认padding为same
*/
class Activation{
public:


private:



};



#endif