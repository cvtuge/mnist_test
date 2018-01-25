/************************************************* 
 *作者:图戈 
 *功能:实现手写数字识别功能 
**************************************************/  
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;  

//网络配置文件
const std::string model_file{ "./build/lenet.prototxt" };  
//训练好的模型文件
const std::string trained_file{ "./build/lenet_iter_10000.caffemodel" };  
//测试图片集路径
const std::string image_path{ "./images/num_4.png" };  
//像素值缩放比例
const float scale_value = 0.00390625;

//所有类别值
const std::vector<int> labels{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };  

//定义一个std::pair<string, float>型的变量，用来存放类别的标签及类别对应的概率值 
typedef std::pair<int, float> Prediction;

/* MnistNet类的声明 */
class MnistNet {
 public:
  //MnistNet构造函数声明
  MnistNet(const string& model_file,
             const string& trained_file);
  //分类函数声明
  std::vector<Prediction> Classify(const cv::Mat& img, int N = 1);

 private:
  //预测函数声明
  std::vector<float> Predict(const cv::Mat& img);
  //将输入图片分通道存储
  void WrapInputLayer(std::vector<cv::Mat>* input_channels);
  //预处理函数声明
  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  //模型变量
  shared_ptr<Net<float> > net_;
  //输入层图片大小
  cv::Size input_geometry_;
  //输入层通道数
  int num_channels_;
};

/**
 * MnistNet - 在MnistNet类外定义的构造函数
 * @model_file: 配置文件
 * @trained_file：训练好的模型文件
 */
MnistNet::MnistNet(const string& model_file,
                       const string& trained_file) {

#ifdef CPU_ONLY
  //设置caffe在CPU上运行
  Caffe::set_mode(Caffe::CPU);
#else
  //设置caffe在GPU上运行
  Caffe::set_mode(Caffe::GPU);
#endif

  //加载网络配置文件，设定模式为测试模式 
  net_.reset(new Net<float>(model_file, TEST));
  //加载训练好的模型文件
  net_->CopyTrainedLayersFrom(trained_file);

  //定义输入层变量
  Blob<float>* input_layer = net_->input_blobs()[0];
  //获取输入层通道数
  num_channels_ = input_layer->channels();
  //检查输入图像通道数，3为RGB图像，1为灰度图
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  //获取输入图像大小
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
}

static bool PairCompare(const std::pair<float, int>& lhs,  
    const std::pair<float, int>& rhs) {  
    return lhs.first > rhs.first;  
}  

/**
 * Argmax - 返回概率最大的N个类别的标签
 * @v: 存放所有概率值的vector
 * @N: 表示概率值最大的N个类别
 */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/**
 * Classify - 预测函数，返回最大的类别概率值和标签
 * @img: 输入图片
 * @N: 表示概率值最大的N个类别
 */
std::vector<Prediction> MnistNet::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

  //获取概率值最大的N个类别的标签，放入vector容器maxN
  std::vector<int> maxN = Argmax(output, N);  
  //定义一个容器，存放最大N个类型的概率值和标签
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {  
      int idx = maxN[i];  
      predictions.push_back(std::make_pair(labels[idx], output[idx]));  
  }  

  return predictions;
}

/**
 * Predict - MnistNet类预测函数
 * @img: 输入单张图片
 */
std::vector<float> MnistNet::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);

  //输入带预测的图片数据，然后进行预处理，包括归一化、缩放等操作   
  net_->Reshape();
  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);
  Preprocess(img, &input_channels);
  //前向传播
  net_->Forward();

  //将输出层的输出值，保存到vector中
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();

  //返回每个类的概率值
  return std::vector<float>(begin, end);
}

/**
 * WrapInputLayer - 为了获得MnistNet网络的输入层数据的指针
 * @input_channels: 把输入图片数据拷贝到这个指针里面
 */
void MnistNet::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

/**
 * Preprocess - 输入图片预处理，对图片进行缩放，归一化，多通道分开存储
 * @img: 输入图片
 * @input_channels：输入层通道
 */
void MnistNet::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  //输入图片通道转换，满足网络输入图片格式要求
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  //修改输入图片大小
  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  //像素值归一化处理
  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3,scale_value);
  else
    sample_resized.convertTo(sample_float, CV_32FC1,scale_value);

  //通道分开存储
  cv::split(sample_float, *input_channels);
}


int main(int argc, char** argv) {
  MnistNet MnistNet(model_file, trained_file);

  string file = argv[3];

  std::cout << "---------- Prediction for "
            << file << " ----------" << std::endl;

  cv::Mat img = cv::imread(file, -1);
  CHECK(!img.empty()) << "Unable to decode image " << file;
  std::vector<Prediction> predictions = MnistNet.Classify(img);

  /* Print the top N predictions. */
  for (size_t i = 0; i < predictions.size(); ++i) {
    Prediction p = predictions[i];
    std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
              << p.first << "\"" << std::endl;
  }
}

