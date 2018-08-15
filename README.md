# 2D_incomp_invicid
Computational Fluid Dynamics BEM solver of 2D steady incompressible invicid flow around arbitary shape object including Kutta-condition.

dvm_beta.py : 離散渦法による定常解ソルバー
　複素平面にて物体表面に多数の渦糸を配置し，その重ね合わせで解を表現する．
　後縁から反時計回りに1周分配置すれば任意形状が表現できる．
　クッタ条件は，後縁が形成する角度の2等分線の方向にて流速が0になるという条件を組み込んでいる．
　誤差は翼面に500点の制御点を配置して10の-2乗程度のオーダー．
　計算速度は10万ケースを2時間ほど(intel Xeon Silver環境下)

joukowski_wing.py : joukowski翼・KarmanTreftz翼を作成し，反時計回りに配置する．
naca_4digit_test.py : NACA4桁系・NACA5桁系の翼を生成し，反時計回りに配置する．コード方向に等間隔プロットするオプションあり．
legacy_vtk_writer.py : numpy配列をrectilinear型のvtkファイルに出力する．
make_validation_figure.py : 計算結果の信憑性をjoukowski翼の解析解と比較する．

以下は計算結果を用いて機械学習させるためのプログラム群
fourier_expansion.py : 入力配列を数値的にフーリエ正弦展開する
make_shape_data.py : 
make_training_data.py :
trainig.py :
inference.py :



