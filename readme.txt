=====================================================
フォルダの構成

-------------------------

＊学習データ作成ファイル
・mpcData_R.py
	回帰NN&dualNetの回帰NN部分用データセット
	データの大きさ(回帰)... 入力2: ラベル60
	[x1, x2, voltage[k], voltage[k+1], ..., voltage[k+n-1]]
	状態　-2 <= x <= 2
	入力　-1 <= u <= 1

	・dualNet　mpc_modeling_tool_v6　10000個 dataset_dualC.csv

	・分類NN mpc_modeling_tool_v6 20000個 dataset_C.csv
・mpcData_C.py
	分類NN&dualNetの分類NN部分用データセット
	データの大きさ(分類)... 入力2: ラベル180
	[x1, x2, uminus[k], uzero[k], uplus[k], ..., uminus[k+n-1], uzero[k+n-1], uplus[k+n-1]]
	状態　-2 <= x <= 2
	入力　-1 <= u <= 1

	・dualNet mpc_modeling_tool_v6　10000個 dataset_dualC.csv　

	・回帰NN mpc_modeling_tool_v6 20000個 dataset_R.csv　

※dataset_C.csvのうち10000個抜き出したものがdataset_dualC.csv
※dataset_R.csvのうち10000個抜き出して絶対値をとったものがdataset_dualR.csv

-------------------------

＊学習&パラメータモデル保存ファイル
・mpcLearn_dualR.py dualNet回帰部分学習
	dataset_dualR.csv => model_dualR.pth

・mpcLearn_dualC.py dualNet分類部分学習
	dataset_dualC.csv => model_dualC.pth

・mpcLearn_R.py 回帰学習 dataset_dualC.csv
	dataset_R.csv =>  model_R.pth

・mpcLearn_C.py 分類学習 dataset_dualC.csv
	dataset_C.csv =>  model_C.pth

-------------------------

＊推論シミュレーションファイル(基本的にはこちらを実行して結果を得る)
・mpcControl_Base.py 近似元シミュレーション//
・mpcInf_dual.py dualNetシミュレーション
・mpcInf_R.py 回帰シミュレーション
・mpcInf_C.py 分類シミュレーション
・mpcInf_Q.py 回帰＋量子化器シミュレーション

-------------------------

＊その他
・l1sample.py 最適化問題のソルバー//
・dataset/ 教師データ
・params/ 学習済みモデルパラメータ

=====================================================

問題設定

・状態行列、パラメータ
A = np.array([[0, -1],
                [2, 0]])
B = np.array([[2], [0]])

C = np.array([[1], [0]])
(nx, nu) = B.shape
Q = np.eye(nx)
P = np.eye(nx)

・離散時間サンプリング、予測ホライズン
Ts = 0.05
Th = 3
N=int(Th/Ts)
simutime = 10/3  or  1/Th

・パラメータ、制約
a=10 #L1ノルム
b=0.5 #L2ノルム
c=b #L2終端ノルム

umax= 1
umin= -1

・分類問題、量子化器閾値
u>0.1: u=+1, u<-0.1: u=-1, -0.1<=u<=0.1: u=0
=====================================================

