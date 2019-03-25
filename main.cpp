#include <iostream>
#include <sstream>   
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <unistd.h>
#include <sys/time.h>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/export.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/list.hpp>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <unistd.h>
#include <signal.h>
#include "json/json.h"

#include "include/mtcnn.hpp"
#include "include/face_align.hpp"
#include "include/feature_extractor.hpp"
#include "include/face_verify.hpp"
#include <glog/logging.h>

#include "include/utils.hpp"
#include "falconn/eigen_wrapper.h"
#include "falconn/lsh_nn_table.h"

#ifndef MODEL_DIR
#define MODEL_DIR "../models"
#endif
using falconn::construct_table;
using falconn::DenseVector;
using falconn::DistanceFunction;
using falconn::LSHConstructionParameters;
using falconn::LSHFamily;
using falconn::LSHNearestNeighborTable;
using falconn::QueryStatistics;
using falconn::StorageHashTable;
face_verifier   * p_verifier;


mtcnn * p_mtcnn;
feature_extractor * p_extractor;
int numKNN = 50;
int numReranking = 2;
std::unique_ptr<falconn::LSHNearestNeighborTable<falconn::DenseVector<float>>> cptable;
std::vector<falconn::DenseVector<float>> data;
std::pair<std::vector<std::string>, std::vector<std::vector<float> >>  namesFeats;//存储图片名、特征

	
	
void saveFeaturesFilePair(std::pair<std::vector<std::string>, std::vector<std::vector<float> >>  &features, std::string &filename){
    std::ofstream out(filename.c_str());
    std::stringstream ss;
    boost::archive::binary_oarchive oa(ss);
    oa << features.first << features.second;
    out << ss.str();
    out.close();
}

int face_init()
{
	const char * type="caffe";
	std::string model_dir=MODEL_DIR; //加载model地址

	p_mtcnn=mtcnn_factory::create_detector(type);//创建mtcnn
	if(p_mtcnn==nullptr)
	{
		std::cerr<<type<<" is not supported"<<std::endl;
		std::cerr<<"supported types: ";
		std::vector<std::string> type_list=mtcnn_factory::list();

		for(int i=0;i<type_list.size();i++)
			std::cerr<<" "<<type_list[i];

		std::cerr<<std::endl;

		return 1;
	}

	p_mtcnn->load_model(model_dir);  //加载model
	p_mtcnn->set_threshold(0.9,0.9,0.9);//设置阈值
	p_mtcnn->set_factor_min_size(0.6,40);//设置factor最小尺寸

	

	const std::string extractor_name("lightened_cnn");//特征提取

	p_extractor=extractor_factory::create_feature_extractor(extractor_name);

	if(p_extractor==nullptr)
	{
		std::cerr<<"create feature extractor: "<<extractor_name<<" failed."<<std::endl;

		return 2;
	}

	p_extractor->load_model(model_dir);

	p_verifier=get_face_verifier("cosine_distance");//特征比对
	p_verifier->set_feature_len(p_extractor->get_feature_length());
	
	
	//提取ｌｏｇ列表中的特征
	FILE *  fp=fopen("../log.txt","r");
	char str[160];
	
	std::vector<std::vector<float> > feats;//存储人脸特征
    std::vector<std::string> imgNames;
	while(1)
	{
		if ( fgets(str,160,fp)==NULL) break;
     		str[strlen(str)-1]='\0';   // 加一个字符串结束符
		cv::Mat img_color = cv::imread(str);
		if(!img_color.data) continue;
		std::vector<face_box> face_info;
        p_mtcnn->detect(img_color,face_info);

		int face_num = face_info.size();

        if (face_num!=0)
          {
			    float feat[256];
		        cv::Mat aligned;
		      
		        face_box& box=face_info[0];
		        get_aligned_face(img_color,(float *)&box.landmark,5,128,aligned);
		        p_extractor->extract_feature(aligned,feat);


		        std::vector<float> featVector(std::begin(feat), std::end(feat));
		        feats.push_back(featVector);
		        std::string dd(str);
		        imgNames.push_back(dd);
		        
		}
	}
	
	
	
	
	namesFeats.first = imgNames;
    namesFeats.second = feats;

        // Save image names and features
	std::string s("../feat");
    saveFeaturesFilePair(namesFeats,s);


	falconn::LSHConstructionParameters params_cp;
	

	int numFeats = (int)namesFeats.first.size();
    int dim = (int)namesFeats.second[0].size();

    // Data set parameters
    uint64_t seed = 119417657;

    // Common LSH parameters
    int num_tables = 8;
    int num_setup_threads = 0;
    falconn::StorageHashTable storage_hash_table = falconn::StorageHashTable::FlatHashTable;
    falconn::DistanceFunction distance_function = falconn::DistanceFunction::NegativeInnerProduct;


    for (int ii = 0; ii < numFeats; ++ii) 
     {
                falconn::DenseVector<float> v = Eigen::VectorXf::Map(&namesFeats.second[ii][0], dim);
                v.normalize(); // L2归一化
                data.push_back(v);
     }

            // Cross polytope hashing
     params_cp.dimension = dim;
     params_cp.lsh_family = LSHFamily::CrossPolytope;
     params_cp.distance_function = distance_function;
     params_cp.storage_hash_table = storage_hash_table;
     params_cp.k = 2; // 每个哈希表的哈希函数数目
     params_cp.l = num_tables; // 哈希表数目
     params_cp.last_cp_dimension = 2;
     params_cp.num_rotations = 2;
     params_cp.num_setup_threads = num_setup_threads;
     params_cp.seed = seed ^ 833840234;
     cptable = std::unique_ptr<falconn::LSHNearestNeighborTable<falconn::DenseVector<float>>>(std::move(construct_table<falconn::DenseVector<float>>(data, params_cp)));
     cptable->set_num_probes(896);
return true;
	}


//face返回结果
int face_recall(cv::Mat& img,std::string& send_datas)
{
	
		cv::Mat img5;
		img.copyTo(img5);
		std::vector<face_box> face_info1;
        p_mtcnn->detect(img5,face_info1);
		float query_feat[256];
		int face_num5 = face_info1.size();
		printf("num is %d \n",face_num5);
		if (face_num5!=0)
    		{
			
			Json::Value json_send;
			Json::Value json_array;
			std::vector<int32_t> idxCandidate;
			cv::Mat aligned;
			/* align face */
			face_box& box=face_info1[0];
			get_aligned_face(img5,(float *)&box.landmark,5,128,aligned);//人脸矫正
			
			
			p_extractor->extract_feature(aligned,query_feat);//人脸特征提取
			
			falconn::DenseVector<float> q = Eigen::VectorXf::Map(&query_feat[0], 256);//生成ｍmap
			q.normalize();//归一化
			cptable->find_k_nearest_neighbors(q, numKNN, &idxCandidate);//KNN

			std::vector<std::pair<float, size_t> > dists_idxs;
			for (int i = 0 ; i < numReranking ; i++) {
			    float tmp_cosine_dist = q.dot(data[idxCandidate[i]]);
			    dists_idxs.push_back(std::make_pair(tmp_cosine_dist, idxCandidate[i]));
			}

			std::sort(dists_idxs.begin(), dists_idxs.end());
			std::reverse(dists_idxs.begin(), dists_idxs.end());

			for(int i = 0 ; i < numReranking ; i++){
			    idxCandidate.at(i) = (int32_t)dists_idxs[i].second;
			}

			for (size_t i = 0 ; i != idxCandidate.size() ; i++) {
			   	float sim = p_verifier->compare(query_feat, namesFeats.second[idxCandidate[i]].data(),256);//比较
			    	if(sim < 0.58) break;
				std::string tmpImgName = namesFeats.first.at(idxCandidate[i]);
				char r[16];
				sprintf(r,"%d",i);
				Json::Value vv;
				vv[r] = Json::Value(tmpImgName.c_str());
				json_array.append(vv);
        		}
			json_send["data"] = json_array;
			send_datas = json_send.toStyledString();
			printf("hah %s \n",send_datas.c_str());
		//	send(connfd,send_data1.c_str(),send_data1.length(),0); 
			
		}
}

int main(int argc, char *argv[])
{
	face_init();
		
	cv::Mat img5 = cv::imread(argv[1]);
		
	std::string send_datas;
		
	face_recall(img5,send_datas);  

}
