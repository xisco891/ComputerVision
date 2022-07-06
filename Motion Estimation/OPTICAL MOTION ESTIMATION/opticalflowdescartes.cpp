



//
//void opticalFlow::covariance(){
//
//
//	VectorXd dif;
//
//	cov.resize(tracks[0].size(),tracks[0].size());
//	cov.fill(0);
//
//	for(int i=0;i<clusters.size();i++)
//	{
//		for(int j=0;j<clusters[i].size();j++)
//		{
//
//			dif=(tracks[clusters[i][j]]-seeds[i]);
//			cov+=dif*dif.transpose();
//
//		}
//		cov/=clusters[i].size();
//		covariances.push_back(cov);
//	}
//
//}
//
//
//void opticalFlow::read_covariances(){
//
//	for(int h=0;h<covariances.size();h++) {
//		cout<<"1"<<endl;
//		for (int i=0;i<covariances[h].cols();i++)
//		{
//			for(int j=0;j<covariances[h].rows();j++){
//				cout<<" "<<endl<<covariances[h](i,j);
//			}
//
//			cout<<"\n"<<endl;
//			waitKey(30);
//		}
//		cout<<"\n\n"<<endl;
//	}
//
//}
//
//
//
//void opticalFlow::writetoFile_covariance(char * filename)
//{
//	float min_sum=0;
//	FILE* pfile = fopen(filename, "w");
//
//	//	cout<<"covariances size: "<<endl<<covariances.size()<<endl;
//
//	for(int h=0;h<covariances.size();h++) {
//
//		for (int i=0;i<covariances[h].cols();i++)
//		{
//
//			for(int j=0;j<covariances[h].rows();j++){
//				fprintf(pfile,"%f ",covariances[h](i,j));
//			}
//
//			fprintf(pfile,"\n");
//
//		}
//
//	}
//
//	fclose(pfile);
//}




//void opticalFlow::readFile(char * filename)
//{
//	covariances2.resize(0);
//
//	MatrixXd cov2(18,18);
//	int h=0;
//	int i=0;
//
//	ifstream theStream(filename);
//	if( ! theStream )
//		cerr << "file.in\n";
//
//	while (true)
//	{
//		string line;
//		getline(theStream, line);
//
//
//		if (line.empty())
//			break;
//
//		istringstream myStream( line );
//		istream_iterator<float> begin(myStream), eof;;
//		vector<float> numbers(begin, eof);
//
//
//		if(h==numbers.size()){
//				covariances2.push_back(cov2);
//				i++;
//				h=0;
//			}
//
//		for(int j=0;j<numbers.size();j++)
//		{
//			cov2(h,j)=numbers[j];
//		}
//
//		h++;
//
//
//	}
//
//	covariances2.push_back(cov2);
//
//}






//
//void opticalFlow::distances_covariances(char * filename, char * filename2)
//{
//
//
//	cout<<"0.0"<<endl;
//	min=1000;
//	vector<double> min_dis(80,0);
//	vector<double> minimum(80,0);
//
//	int k=0;
//	int i=0;
//	int end1=0;
//	int end2=0;
//	double rest=0;
//	min_sum=0;
//
//	VectorXd resta;
//	VectorXd dist;
//
//
//	ifstream theStream1(filename);
//	if( ! theStream1 ){
//		cerr << "file1.in\n";
//	}
//
//	while(true)
//	{
//		ifstream theStream2(filename2);
//
//		if( ! theStream2 ){
//			cerr << "file2.in\n";
//		}
//
//		string line1;
//		getline(theStream1, line1);
//
//		if (line1.empty()){
//			cout<<"hallo!"<<endl;
//			end1++;
//			getchar();
//			break;
//		}
//
//		if(end1>0){
//			break;
//		}
//
//
//		istringstream myStream1( line1 );
//		istream_iterator<float> aux(myStream1), eof;
//		vector<float> numbers1(aux, eof);
//
//
//
//		while(k<80)
//		{
//
//			string line2;
//			getline(theStream2, line2);
//
//			if (line2.empty()){
//				cout<<"hallo2!"<<endl;
//				end2++;
//				break;
//			}
//
//			if(end2>0){
//				break;
//			}
//			cout<<"1.1"<<endl;
//			cout<<"1.2"<<endl;
//
//			istringstream myStream2( line2 );
//			istream_iterator<float> aux2(myStream2), eof;
//			vector<float> numbers2(aux2, eof);
//			cout<<"1.3"<<endl;
//
//
//			resta.resize(numbers1.size());
//			cout<<"1.4"<<endl;
//
//			for(int j=0;j<numbers1.size();j++)
//			{
//				double dValue1(0.0);
//				double dValue2(0.0);
//
//				dValue1 = static_cast<double>(numbers1[j]);
//				dValue2 = static_cast<double>(numbers2[j]);
//
//				rest=dValue1-dValue2;
//				resta[j]=abs(rest);
//				cout<<"j:"<<j;
//				cout<<""<<endl;
//
//			}
//
//
//
//            cout<<"i"<<i;
//            cout<<""<<endl;
//
////            cout<<"distances size:"<<distances.size();
////            cout<<""<<endl;
////            cout<<"\n"<<covariances[i].rows();
////            cout<<""<<endl;
////            waitKey(1500);
//
//
//			distances=(resta.transpose())*(covariances2[i].transpose());
//			cout<<"1.6"<<endl;
//			distances=distances*resta;
//			cout<<"1.7"<<endl;
//			min_dis[k]=sqrt(distances(0,0));
//
//
//			if(min_dis[k]<min){
//
//				min=min_dis[k];
//				}
//			k++;
//		}
//
//
//		minimum[i]=min;
//		min_sum+=minimum[i];
//
//
//		end2=0;
//		k=0;
//		i++;
//	}
//
//	cout<<"min_sum"<<min_sum;
//	cout<<""<<endl;
//	waitKey(3000);
//
//}



//
//void opticalFlow::draw_visualwords()
//{
//
//
//
//	Mat img1(860,860, CV_8UC3, Scalar(0,0,0));
//	int end1=0;
//
//	int h=img1.cols/2;
//	int l=img1.rows/2;
//
//	int m=0;
//	int res=0;
//	int i=0;
//	int k=0;
//
//	imshow("",img1);
//	waitKey(2000);
//
//	vector<Point2f> traj(8);
//	vector<Point2f> traj2(8);
//
//
//	cout<<"---1"<<endl;
//
//	for(int i=0;i<min_pos_vect.size();i++){
//
//		cout<<"min_pos_vect["<<i;
//		cout<<"]="<<min_pos_vect[i];
//		cout<<""<<endl;
//
//		int f=min_pos_vect[i];
//
//		cout<<"\n f="<<f;
//		cout<<""<<endl;
//
//		traj[m]=Point(h,l/4 +res);
//		traj2[m]=Point(h,l/4 + res);
//
//		cout<<"tracks_aux["<<i;
//		cout<<"].size="<<tracks_aux[i].size();
//		cout<<""<<endl;
//
//
//		for(int j=0;j<tracks_aux[i].size();j+=2){
//
//			cout<<"2"<<endl;
//			cout<<"i:"<<i;
//			cout<<""<<endl;
//
//			double dValue1(0.0);
//			dValue1 = static_cast<double>(tracks_aux[f](j));
//			double dValue2(0.0);
//			dValue2 = static_cast<double>(tracks_aux[f](j+1));
//
//			double dValue3(0.0);
//			dValue3 = static_cast<double>(tracks_aux_2[i](j));
//			double dValue4(0.0);
//			dValue4 = static_cast<double>(tracks_aux_2[i](j+1));
//
//
//
//
//			dValue1=dValue1*10;
//			dValue2=dValue2*10;
//			dValue3=dValue3*10;
//			dValue4=dValue4*10;
//
//			traj2[m+1]=traj2[m]+Point2f(dValue1,dValue2);
//			line(img1,traj2[m],traj2[m+1],Scalar(0,0,255),1);
//			imshow("img1",img1);
//			waitKey(50);
//			traj[m+1]=traj[m]+Point2f(dValue3,dValue4);
//			line(img1,traj[m],traj[m+1],Scalar(0,255,255),1);
//			imshow("img1",img1);
//			waitKey(50);
//			cout<<"traj2:"<<traj2[m+1];
//			cout<<"\ntraj:"<<traj[m+1];
//			cout<<""<<endl;
//
//			m++;
//			waitKey(200);
//
//
//
//		}
//
//		m=0;
//		res+=5;
//
//		cout<<"i:"<<i;
//		cout<<""<<endl;
//		imshow("img1",img1);
//		waitKey(1000);
//
//
//	}
//
//		imshow("img1",img1);
//		cout<<"4"<<endl;
//		imwrite( "../comparefolder/probe/Walk_min%(5,5).jpg", img1);
//		cout<<"5"<<endl;
//		waitKey(1000);
//}




//void opticalFlow::compare_covariances(char * filename)
//{
//
//
//	vector<double> cofs;
//	vector<vector<int > resta(clusters.size(),0);
//	vector<vector<int> > clust(clusters.size());
//	float min=1000;
//	ifstream theStream(filename);
//	if( ! theStream )
//		cerr << "file.in\n";
//	int n_clust=0;
//
//	int i=0;
//		while (true)
//		{
//			string line;
//			getline(theStream, line);
//			if (line.empty())
//				break;
//			istringstream myStream( line );
//			istream_iterator<int> begin(myStream), eof;
//			vector<int> numbers(begin, eof);
//
//		int l=0;
//		for(int h=0;h<covariances.size();h++){
//			do{
//			for(int j=0;j<numbers.size();j++)
//			{
//				clust(i,j)=numbers[j];
//				resta+=clusters(i,j)-clust(i,j);
//			}
//			if(resta[h]<min)
//				min=resta[h][l];
//				n_clust=h;
//
//			l++;
//			}while(l<numbers.size());
//			cout<<"Cluster number: "<<h;
//			cout<<"of probe videos have more similarity with "<<endl;
//			cout<<"\nCluster n° "<<n_clust;
//			cout<<" of trained system  "<<endl;
//			cout<<" belongs to video: "<<filename;
//			cout<<" "<<endl;
//			waitKey();
//
//		}
//		i++;
//		}
//
//}







//void opticalFlow::denseflow(int resolution){
//	Mat uv;
//	Mat channels[2];
//	char file_flow[100];
//	//
//	//	cout<<"SIZE IMAGE:"<<I.size();
//
//	for(int i=0;i<I.size()-1;i++){
//
//		calcOpticalFlowFarneback (I[i], I[i+1], uv, 0.5, 3, 15, 3, 5, 1.2, 0);
//		split(uv,channels);
//		u1[i]=channels[0].clone();
//		v1[i]=channels[1].clone();
//	}
//
//	Mat Ic;
//	vector<Point2f> point;
//	int m=0;
//	cvtColor(I[I.size()-1], Ic,CV_GRAY2BGR);
//
//	num_maps=I.size()/10;
//	cout<<"Number of maps: "<<endl;
//	cout<<"\n "<<num_maps;
//	vector<Point2f> traj(I.size());
//
//	for(int k=0;k<I[m].cols;k+=resolution){
//		for(int l=0;l<I[m].rows;l+=resolution)
//		{
//			VectorXd tv(2*u1.size());
//			tracks.push_back(tv);
//			traj[m]=Point2f(k,l);
//
//
//			for(int j=m;j<m+9;j++)
//			{
//
//				//	cout<<"j= "<<j<<", "<<num_points<<endl;
//				traj[j+1]=traj[j]+Point2f(u1[j].at<float>(l,k),v1[j].at<float>(l,k));
//				line(Ic,traj[j],traj[j+1],Scalar(0,0,255),1);
//
//				tracks[num_points](2*j)=u1[j].at<float>(traj[j]);
//				tracks[num_points](2*j+1)=v1[j].at<float>(traj[j]);
//
//			}
//			num_points++;
//		}
//	}
//	cout<<"5"<<endl;
//	imwrite("../FLOW/%03d",Ic);
//	dense_count++;
//
//
//
//}



//
//void opticalFlow::drawFlow(Mat& flow, int step)
//{
//	for(int x1=0;x1<I1.cols;x1+=step)
//	{
//		for(int y1=0;y1<I1.rows;y1+=step)
//		{
//			Point2f p1(x1,y1);
//			Point2f p2;
//			p2.x=p1.x+5*u.at<float>(y1,x1);
//			p2.y=p1.y+5*v.at<float>(y1,x1);
//
//			line(flow,p1,p2,Scalar(0,0,255),1);
//			circle(flow,p2,2,Scalar(0,255,255),1);
//		}
//}
//
//
//
//
//
//					if(tracks[num_points](2*j)>1){
//						cout<<"TRACK: "<<num_points;
//						cout<<"\n"<<2*j;
//
//					}
//
//					if(tracks[num_points](2*j+1)>1){
//						cout<<"TRACK: "<<num_points;
//						cout<<"\n"<<2*j+1;
//					}
//
//}
