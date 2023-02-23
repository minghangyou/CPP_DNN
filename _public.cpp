#include "_public.h"


int check_error_num(vector<int> a, vector<int> b) {
	float threshold = 0.0000001;
	if (a.size() != b.size()) {
		cout << "两组数据长度不一致" << endl;
	}
	int error_num = 0;
	for (auto i = 0; i < a.size(); i++) {
		if (abs(a[i] - b[i]) > threshold) {
			++error_num;
		}
	}
	return error_num;

}


void vec_copy(const vector<data_t>& data1, const vector<data_t>::iterator begin, size_t n, vector<data_t>& data2) {
	assert(begin + (n-1) < data1.end());
	for (size_t i = 0; i < n; i++) {
		data2[i] = *(begin + i);
	}

}


//层序遍历得到叶子节点（dataset）参数
void GetFloorWeights_QUEUE(const string& layer_name, const string& root_str, const H5File& file, const hid_t& root_id, unordered_map<string, vector<float>>& model_data) {
	queue<pair<hid_t,string>> que;	//为了方便后续操作，这里将objectID以及其路径一起存储在队列中
	que.push({root_id,root_str});
	while (!que.empty()) {
		int que_size = que.size();
		for (int i = 0; i < que_size; i++) {
			pair<hid_t, string> cur_node = que.front();
			hid_t cur_id = cur_node.first;
			string cur_str = cur_node.second;
			que.pop();
			H5I_type_t obj_type = H5Iget_type(cur_id);
			//如果是group，就把其下所有的内容都入队列
			if (obj_type == H5I_GROUP) {
				//获取该group下object的数量
				Group rg(cur_id);
				const hsize_t objs = rg.getNumObjs();
				//将group下的所有object全部入队列
				for (hsize_t j = 0; j < objs; j++) {
					const H5std_string name = rg.getObjnameByIdx(j);
					string tmp_str = cur_str + "/" + name;
					hid_t tmp_id = file.getObjId(tmp_str);
					que.push({ tmp_id, tmp_str});
				}
				rg.close();

			}
			//如果是dataset，说明到达了叶子节点，把datasett输入写入model_data即可
			if (obj_type == H5I_DATASET) {
				//打开对应的dataset
				string dataset_name = cur_str;
				DataSet dset = file.openDataSet(dataset_name);
				// 获取Dataset中数据的数据类型
				DataType dt = dset.getDataType();
				const H5T_class_t t = dt.getClass();
				//判定读取的数据类型为float（目前只写了float的逻辑，需要支持多数据类型的话，后续可以直接用函数模板解决）
				if (t == H5T_FLOAT) {
					// 数据在内存中的字节数除以数据类型得到 buf 的大小
					const hsize_t data_size = dset.getInMemDataSize() / sizeof(float);
					float* buf = new float[data_size];
					// 读出数据到 buf 中
					dset.read(buf, dt);
					//cout<< dataset_name <<":" << endl;
					for (int k = 0; k < data_size; k++)
					{
						//cout << buf[k] << endl;
						model_data[layer_name].push_back(buf[k]);
					}
					delete[]buf;
					buf = nullptr;
				}
				dt.close();
				dset.close();
			}
		}
	}
	return;
}


bool GetLayerWeights(const string& filename, vector<string>& layers_name, unordered_map<string, vector<float>>& model_data) {
	// 用只读方式打开文件, 用完后记得要调用 file.close() 关闭释放资源
	H5File file(filename, H5F_ACC_RDONLY);
	//遍历所有的layer
	for (int i = 0; i < layers_name.size(); i++) {
		//拼接指定的object路径
		string group_string = "";
		group_string = "/" + layers_name[i];
		hid_t root_id = file.getObjId(group_string);//相当于每个layer分支的根节点
		GetFloorWeights_QUEUE(layers_name[i], group_string, file, root_id, model_data);
		
	}
	file.close();
	return true;
}



/*
	BN层的类相关成员函数
*/
void BN::bn_forward(const vector<data_t>& input_data, vector<data_t>& output_data) {
	size_t param_size = params->gamma.size();	
	assert(input_data.size() == param_size);	// 确保BN层的输入size和参数的size一致
	for (size_t i = 0; i < param_size; i++) {
		data_t output = 0;
		//由于sqrt函数这些没有对于定点数的重载，因此这里需要用float强制类型转换
		float moving_variance_float = params->moving_variance[i];
		float epsilon_float = params->epsilon;
		output = params->gamma[i] / ((data_t)sqrt(moving_variance_float + epsilon_float)) * input_data[i] + (params->beta[i] - params->gamma[i] * params->moving_mean[i] / ((data_t)sqrt(moving_variance_float + epsilon_float)));
		output_data.push_back(output);
	}
}


void BN::ShowWeights() {
	cout << "参数为：" << endl;
	cout << "beta：" << endl;
	for (auto& v : params->beta) {
		cout << v << ", ";
	}
	cout << endl;
	cout << "gamma：" << endl;
	for (auto& v : params->gamma) {
		cout << v << ", ";
	}
	cout << endl;
	cout << "moving_mean：" << endl;
	for (auto& v : params->moving_mean) {
		cout << v << ", ";
	}
	cout << endl;
	cout << "moving_variance：" << endl;
	for (auto& v : params->moving_variance) {
		cout << v << ", ";
	}
	cout << endl;
}


/*
	Conv1D层相关的类
*/
vector<vector<data_t>> Conv1d::conv1d_forward(const vector<vector<data_t>>& input_data) {
	assert(!input_data.empty());
	size_t in_height = input_data.size();
	size_t in_width = input_data[0].size();
	size_t kernel_nums = params->filters;
	size_t kernel_height = params->kernel_size;
	size_t kernel_width = params->kernel.size() / (kernel_nums * kernel_height);
	size_t out_width = params->filters;
	size_t out_height = 0;
	size_t pad_h = 0;

	// 根据padding方式计算padding数量
	if (params->padding) {
		pad_h = (kernel_height - 1) / 2;
		out_height = in_height;
	}
	else{
		out_height = (in_height - kernel_height + params->steps)/ params->steps;
	}
	// 构建输出矩阵
	vector<vector<data_t>> output(out_height, vector<data_t>(out_width, 0));

	// 执行卷积操作
	for (int i = 0; i < out_height; ++i) {
		for (int j = 0; j < out_width; ++j) {
			int in_y = i * params->steps - pad_h;
			data_t sum = 0;
			for (int ky = 0; ky < kernel_height; ++ky) {
				int y = in_y + ky;
				if (y < 0 || y >= in_height) {
					continue;
				}
				for (int kx = 0; kx < kernel_width; ++kx) {
					int x = kx;
					if (x < 0 || x >= in_width) {
						continue;
					}
					sum += input_data[y][x] * params->kernel[j * kernel_height * kernel_width + ky * kernel_width + kx];
				}
			}
			output[i][j] = sum;
		}
	}
	return output;

}