# coding = utf-8

# 基于用户的协同过滤推荐算法实现
import random

import math
from operator import itemgetter
import multiprocessing


class UserBasedCF():
    # 初始化相关参数
    def __init__(self):
        # 找到与目标用户兴趣相似的20个用户，为其推荐10部电影
        self.n_sim_user = 20
        self.n_rec_movie = 10

        # 将数据集划分为训练集和测试集
        # 用户相似度矩阵
        self.user_sim_matrix = {}
        self.movie_count = 0

        print('Similar user number = %d' % self.n_sim_user)
        print('Recommneded movie number = %d' % self.n_rec_movie)


    # 读文件，返回文件的每一行
    def load_file(self, filename,cols):
        self.trainSet = pd.read_csv(filename,error_bad_lines = False)
        self.trainSet = self.trainSet.sort_values(by = ['user','movie'], kind = 'mergesort').reset_index(drop = True)
        self.users = self.trainSet.yyuid.sort_values(kind = 'mergesort').drop_duplicates(keep='first')
        self.user_movielen = self.trainSet.groupby('user').movie.nunique().reset_index()
        self.user_movielen = self.user_movielen.sort_values(by = ['user'], kind = 'mergesort').reset_index(drop = True).ayyuid
        batch = 2000
        self.userlist = []
        self.user_movielenlist = []
        for i in range(500):
            if (i+1)*batch<len(self.users):
                self.userlist.append(self.users[i*batch:(i+1)*batch])
                self.user_movielenlist.append(self.user_movielen[i*batch:(i+1)*batch])
            else:
                self.userlist.append(self.users[i*batch:]) 
                self.user_movielenlist.append(self.users[i*batch:]) 
                break
        print(len(self.userlist))
        
        
#         self.trainSet['s_watch_dt_ucnt.p30d'] = self.trainSet['s_watch_dt_ucnt.p30d'].astype('int16')
        #filter lowfreq user/movie

    def build_matrix(self):
        def build_submat(idx,subchunk,results):
            user_movie2 = pd.DataFrame(columns = self.cate.tolist())
            tmp = pd.pivot_table(subchunk,index = 'yyuid',
                                         columns = 'ayyuid',
                                         values='s_watch_dt_ucnt.p30d',
                                         fill_value = 0)
            user_movie2 = pd.concat([user_movie2,tmp],axis = 0)
            user_movie2.fillna(0,inplace = True)
            c = ssp.coo_matrix(user_movie2.values,dtype=np.int16)
            results[idx] = c
        self.cate = self.trainSet.ayyuid.drop_duplicates()
        self.vmatrix = []
        
        max_process = 5
        lens = len(self.userlist)
        blocks = int(len(self.userlist)/max_process)
        for b in tqdm(range(blocks+1)):
            jobs = []
            multiprocessing.freeze_support()
            manager = multiprocessing.Manager()
            results = manager.dict()
            if b<blocks:
                processes = max_process
            else:
                processes = len(self.userlist)%max_process
            for p in range(processes):
                subuser = self.userlist[p+b*max_process]
                idx = p+b*max_process
                print('building...')
                firstid = int(subuser[0:1].values)
                firstindex = int(self.trainSet[self.trainSet.yyuid==firstid].index[0:1].values)
                lastid = int(subuser[-1:].values)
                lastindex = int(self.trainSet[self.trainSet.yyuid==lastid].index[-1:].values)
                subchunk = self.trainSet.iloc[firstindex:lastindex,:]
                j = multiprocessing.Process(target=build_submat,args=(idx,subchunk,results))
                jobs.append(j)    
            for j in jobs:
                j.start()
            for j in jobs:
                j.join()
            for j in jobs:
                j.terminate()
            for r in sorted(results.items(), key=lambda d: d[0]):
                self.vmatrix.append(r[1])
                
    def calc_user_sim(self):
        print('Building movie-user table ...')
        cate = self.trainSet.ayyuid.drop_duplicates()
        self.user_sim_vmatrix = []
        
        def cal_sub_mat(idx,tmpmat,mtx,results,len1,len2):
            mtx2_T = ssp.csc_matrix(tmpmat,dtype=np.int16).T
            print (mtx2_T.shape)
            c = mtx.dot(mtx2_T)
            len1 = ssp.csc_matrix(len1.values.reshape(-1,1))
            len2 = ssp.csc_matrix(len2.values)
            n = len1.dot(len2)
            print(n.shape)
            c = c/n
            c = ssp.coo_matrix(c)
            results[idx] = c
            print(c.shape)   
        for i,t in tqdm(enumerate(self.vmatrix)):###########
            print(t.shape)
            mtx = ssp.csc_matrix(t,dtype=np.int16) 
            len1 = self.user_movielenlist[i]
            print(len1.shape)
            hmatrix = [
            multiprocessing.freeze_support()
            manager = multiprocessing.Manager()
            results = manager.dict()
            jobs = []
            for idx,subuser in enumerate(self.vmatrix):
                len2 = self.user_movielenlist[idx]
                j = multiprocessing.Process(target=cal_sub_mat,args=(idx,subuser,mtx,results,len1,len2))
                jobs.append(j)    
            for j in jobs:
                j.start()
            for j in jobs:
                j.join()
            for j in jobs:
                j.terminate()
            for i in sorted(results.items(), key=lambda d: d[0]):
                hmatrix.append(i[1])

            hmatrix = ssp.hstack(hmatrix)
        
            self.user_sim_vmatrix.append(hmatrix)
        self.user_sim_vmatrix = ssp.vstack(self.user_sim_vmatrix)
        print(self.user_sim_vmatrix.shape)   
