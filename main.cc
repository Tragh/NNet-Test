#include <iostream>
#include <vector>
#include <inttypes.h>
#include <armadillo>
#include <cassert>
#include <math.h>
#include <fstream>
#include <algorithm>
#include <arpa/inet.h>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>



//Compile with: g++ -march=native -O3 -std=gnu++11 -ffast-math -pipe -larmadillo -lboost_iostreams main.cc -o main


template <typename type>
class NeuralNet{
	public:
	using Matrix=arma::Mat<type>;
	using Vector=arma::Col<type>;

	
	std::vector<int> LayerSizes;
	int NumLayers;
	std::vector<Matrix> LinMaps;
	std::vector<Vector> Translations;
	std::vector<Vector> a; //outputs - should be done in a function but class scope is faster
	std::vector<Vector> z; //tmp should be inside a function
	std::vector<Vector> delta; //derivs - sould be done in a function but class scope is faster
	
	NeuralNet(const std::vector<int> &LayerSizes_):
		LayerSizes(LayerSizes_),
		NumLayers(LayerSizes_.size()),
		LinMaps(NumLayers-1),
		Translations(NumLayers-1),
		a(NumLayers),
		z(NumLayers),
		delta(NumLayers-1)
	{
		
		assert(NumLayers>=3);
		for(auto i : LayerSizes)
			assert(i>=1);
		
		for(int i=0;i<NumLayers-1;++i){
			LinMaps[i]= arma::randn<Matrix>(LayerSizes[i+1],LayerSizes[i])/sqrt(LayerSizes[i+1]);
			Translations[i]= arma::randn<Vector>(LayerSizes[i+1]);
		}
				
	}
	
	inline type Sigma(type z)
	{
		return tanh(z);
	}
		
	void Sigma(const Matrix& v, Matrix& out){
		if(v.n_elem!=out.n_elem) out=v;
		for (int i = 0; i < v.n_elem; ++i)
			out[i] = Sigma(v[i]);
	}
	
	inline type SigmaPrime(type z)
	{
		return 1.0-tanh(z)*tanh(z);
	}
	
		void VSigmaPrime(Matrix &v) {
    for (int i = 0; i < v.n_elem; ++i)
        v[i] = SigmaPrime(v[i]);
	}

	Vector RunThrough(const Vector &v){
		assert(v.n_elem == LayerSizes[0]);
		Vector ret=v;
		for(int i=0;i<NumLayers-1;++i)
			Sigma(LinMaps[i]*ret+Translations[i], ret);
		return ret;
	}
		

	
	void BackPropM(const Matrix& input, const Matrix& expected_output, std::vector<Vector>& a, std::vector<Vector>& delta)
	{
		std::vector<Matrix> aM(NumLayers);
		std::vector<Matrix> zM(NumLayers);
		std::vector<Matrix> deltaM(NumLayers-1);
		
		aM[0]=input; //copy
			
		for(int i=0;i<NumLayers-1;++i){
			zM[i+1]=LinMaps[i]*aM[i];
			zM[i+1].each_col()+=Translations[i];
			Sigma(zM[i+1],aM[i+1]);
		}
		
		deltaM[NumLayers-2]=(aM[NumLayers-1]-expected_output);
		
		for(int i=NumLayers-3;i>=0;--i){
			VSigmaPrime(zM[i+1]); //z trashed
			deltaM[i]=(LinMaps[i+1].t()*deltaM[i+1])%zM[i+1];
		}
		
		
		for(int i=0;i<NumLayers;i++){
			
			a[i]=aM[i].col(0);
			for(int c=1;c<aM[i].n_cols;c++)
				a[i]+=aM[i].col(c);
		}

		for(int i=0;i<NumLayers-1;i++){
			delta[i]=deltaM[i].col(0);
			for(int c=1;c<deltaM[i].n_cols;c++)
				delta[i]+=deltaM[i].col(c);
		}
		
	}

	type ECost(Vector input, const Vector& expected_output)
	{
		input=RunThrough(input);
		type ret=0;
		for (int i = 0; i < input.n_elem; ++i){
			const double &a = input[i];
			const double &y = expected_output[i];
			ret+=(-(y+1)*log((a+1)/2)/2-(1-y)*log((1-a)/2)/2);
		}
		return ret;
	}
	

	void LearnEpochM(const Matrix &input, const Matrix &expected_output,int epoch_size, int batches, type rate, type lambda)
	{
		assert(input.n_cols==expected_output.n_cols);
		assert(batches!=0);

		type adjusted_rate=rate/batches;
		std::vector<int> unique_randoms;
		for(int i=0;i<epoch_size;++i)
			unique_randoms.push_back(i);
		std::random_shuffle(unique_randoms.begin(), unique_randoms.end());
		int random_index=0;
		
		
		Matrix inputM(input.n_rows,batches);
		Matrix expected_outputM(expected_output.n_rows,batches);
		
		for(int rep=0;rep<epoch_size;rep+=batches){
			
			
			for(int i=0;i<batches;++i){
				int num=unique_randoms[random_index++];
				inputM.col(i)=input.col(num);
				expected_outputM.col(i)=expected_output.col(num);
			}	
							
				
			BackPropM(inputM, expected_outputM, a, delta);
		
			//Now update the weights and biases
			//Derivative function for weights
			for(int i=0;i<NumLayers-1;++i){
				for(int r=0;r<LinMaps[i].n_rows;++r)
					for(int c=0;c<LinMaps[i].n_cols;++c){
						type DerivativeW=a[i][c]*delta[i][r];
						LinMaps[i](r,c)=-(rate*lambda-1)*LinMaps[i](r,c)-DerivativeW*adjusted_rate;
					}
						//The above regularise and fall down the gradient
						//First shrinking the weights according to the regularizer
						//Then subtracting the derivative
							
			Translations[i]-=delta[i]*adjusted_rate; //Subtracting the derivative
			}
		}
	}
};


class DataReader{
	std::ifstream ImageFile;
	boost::iostreams::filtering_istream ImageFileS;
	
	std::ifstream LabelFile;
	boost::iostreams::filtering_istream LabelFileS;
	
	struct netint_t{  //like an int, but with network (big endian) ordering.
		private: int netint;
		public:	netint_t(){}
		netint_t(const int i){netint = htonl(i);}
		operator int(){return ntohl(netint);}
		char* data(){return reinterpret_cast<char*>(&netint);}
	};
	
	struct {
		netint_t magic_number;
		netint_t num_items;
		netint_t image_height;
		netint_t image_width;
		char* data(){return reinterpret_cast<char*>(this);}
	} HeaderI;
	
	struct {
		netint_t magic_number;
		netint_t num_items;
		char* data(){return reinterpret_cast<char*>(this);}
	} HeaderL;
	
	public:
	DataReader()
	{

		ImageFile.open("data/train-images-idx3-ubyte.gz", std::ios::in | std::ios::binary);
		ImageFileS.push(boost::iostreams::gzip_decompressor());
		ImageFileS.push(ImageFile);
		
		ImageFileS.read (HeaderI.data(), sizeof HeaderI);
		assert(ImageFileS);

		std::cout << "Loaded Image File, data:" << std::endl
				<< "	Magic Number: " << HeaderI.magic_number << std::endl
				<< "	Number Images: " << HeaderI.num_items << std::endl
				<< "	Image Height: " << HeaderI.image_height << std::endl
				<< "	Image Width: " << HeaderI.image_width << std::endl << std::endl;
				
		LabelFile.open("data/train-labels-idx1-ubyte.gz", std::ios::in | std::ios::binary);	
		LabelFileS.push(boost::iostreams::gzip_decompressor());
		LabelFileS.push(LabelFile);	
		
		LabelFileS.read (HeaderL.data(), sizeof HeaderL);
		assert(LabelFileS);
		
		std::cout << "Loaded Label File, data:" << std::endl
				<< "	Magic Number: " << HeaderL.magic_number << std::endl
				<< "	Number Labels: " << HeaderL.num_items << std::endl << std::endl;
		

	}

	void GetImages(arma::Mat<double> &ImageM)
	{
		ImageFileS.reset();  //reset the stream and seek to beginning of data
		ImageFile.seekg(0);
		ImageFileS.push(boost::iostreams::gzip_decompressor());
		ImageFileS.push(ImageFile);
		ImageFileS.ignore(sizeof HeaderI);
		
		int size=HeaderI.image_width*HeaderI.image_height;
		ImageM.resize(size,HeaderI.num_items);
		for(int i=0;i<HeaderI.num_items;++i){
			auto image=GetNextImage();
			for(int j=0;j<size;++j){
				ImageM.col(i)[j]=(static_cast<double>(static_cast<unsigned char>(image[j])))/512;
			}
		}
	}
	
	
	void GetLabels(std::vector<int> &Label)
	{
		LabelFileS.reset();  //reset the stream and seek to beginning of data
		LabelFile.seekg(0);
		LabelFileS.push(boost::iostreams::gzip_decompressor());
		LabelFileS.push(LabelFile);
		LabelFileS.ignore(sizeof HeaderL );
		
		Label.clear();
		for(int i=0;i<HeaderL.num_items;++i){
			Label.push_back(GetNextLabel());
		}
	}
	
	private:
	std::vector<char> GetNextImage()
	{
		std::vector<char> ret(HeaderI.image_height*HeaderI.image_width);
		ImageFileS.read(ret.data(), HeaderI.image_height*HeaderI.image_width);
		assert(ImageFileS);
		return ret;
	}
	
	int GetNextLabel()
	{
		int ret=0;
		LabelFileS.read(reinterpret_cast<char*>(&ret),sizeof(char));
		assert(LabelFileS);
		return ret;
	}
	
	
/*	void PrintImage(int image_number)
	{
		auto image=GetImage(image_number);
		for(int r=0;r<HeaderI.image_height;++r){
			for(int c=0;c<HeaderI.image_width;++c)
				std::cout << static_cast<int>(image[c+r*HeaderI.image_width]+128)/26;
			std::cout << std::endl;
		}
	}*/
};



void PrintV(arma::Col<double> p)
{
	std::cout.precision(2);
	std::cout.setf(std::ios::fixed);
	for(auto v : p)
		std::cout << v << "  ";
}

int Max(arma::Col<double> c)
{
	double max=-1;
	int ret=0;
	for(unsigned int i=0;i<c.n_elem;i++)
		if(c[i]>max){
			max=c[i];
			ret=i;
		}
	return ret;
}


int main()
{	
	NeuralNet<double> NNet({784,30,10}); //784 input layer, 30 hidden, 10 output layer
	DataReader Images; //loads images from files
	
	const float rate=0.03; //learning rate
	const float lambda=2; //normalization rate
	const int TrainingSetSize=1000;
	
	arma::Mat<double> InputM;
	arma::Mat<double> EOutputM;
	std::vector<int> Label;
	

	std::cout << "Caching images and labels for training..." << std::endl;
	Images.GetImages(InputM);
	Images.GetLabels(Label);
	assert(InputM.n_cols == Label.size());
	
	EOutputM.resize(10,Label.size());  //put the outputs in a nice format for learning
	for(int i=0;i<Label.size();++i){
		EOutputM.col(i).fill(-1);
		EOutputM(Label[i],i)=1;
	}
	
	
	std::cout << "Starting learning..." << std::endl;
	int epoch_number=0;
	for(int i=0;i<300;i++){
		NNet.LearnEpochM(InputM, EOutputM, TrainingSetSize, 10, rate, lambda/TrainingSetSize);
		std::cout << " Completed epoch: " << ++epoch_number << std::endl;
	}
	

	std::cout << "Testing learning..." << std::endl;
	
	float accuracy=0;
	for(int i=0;i<1000;i++){
		if(Max(NNet.RunThrough(InputM.col(i))) == Label[i])
			accuracy+=0.1;
	}
	std::cout << "Accuracy on known cases: " << accuracy << "%" << std::endl;
	
	accuracy=0;
	for(int i=50000;i<60000;i++){ //50000-60000 is test data
		if(Max(NNet.RunThrough(InputM.col(i))) == Label[i])
			accuracy+=0.01;
	}
	std::cout << "Accuracy on unknown cases: " << accuracy << "%" << std::endl;
	
	return 0;
}

