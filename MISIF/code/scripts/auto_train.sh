#imput method
echo "python main.py -input_method=attention -result_path=../result/input_method.log"
python ./main.py -input_method='attention' -result_path='../result/input_method.log'
echo "python main.py -input_method=attention -result_path=../result/input_method.log"
python ./main.py -input_method='attention' -result_path='../result/input_method.log'

echo "python main.py -input_method=max -result_path=../result/input_method.log"
python ./main.py -input_method='max' -result_path='../result/input_method.log'
echo "python main.py -input_method=max -result_path=../result/input_method.log"
python ./main.py -input_method='max' -result_path='../result/input_method.log'

#KF
echo "python main.py -KF=-1 -result_path=../result/KF.log"
python ./main.py -KF=-1 -result_path='../result/KF.log'
echo "python main.py -KF=-1 -result_path=../result/KF.log"
python ./main.py -KF=-1 -result_path='../result/KF.log'

echo "python main.py -KF=0 -result_path=../result/KF.log"
python ./main.py -KF=0 -result_path='../result/KF.log'
echo "python main.py -KF=0 -result_path=../result/KF.log"
python ./main.py -KF=0 -result_path='../result/KF.log'

echo "python main.py -KF=10 -result_path=../result/KF.log"
python ./main.py -KF=10 -result_path='../result/KF.log'
echo "python main.py -KF=10 -result_path=../result/KF.log"
python ./main.py -KF=10 -result_path='../result/KF.log'

echo "python main.py -KF=999 -result_path=../result/KF.log"
python ./main.py -KF=999 -result_path='../result/KF.log'
echo "python main.py -KF=999 -result_path=../result/KF.log"
python ./main.py -KF=999 -result_path='../result/KF.log'

#KI
echo "python main.py -KI=2 -result_path=../result/KI.log"
python ./main.py -KI=2 -result_path='../result/KI.log'
echo "python main.py -KI=2 -result_path=../result/KI.log"
python ./main.py -KI=2 -result_path='../result/KI.log'

echo "python main.py -KI=4 -result_path=../result/KI.log"
python ./main.py -KI=4 -result_path='../result/KI.log'
echo "python main.py -KI=4 -result_path=../result/KI.log"
python ./main.py -KI=4 -result_path='../result/KI.log'

echo "python main.py -KI=6 -result_path=../result/KI.log"
python ./main.py -KI=6 -result_path='../result/KI.log'
echo "python main.py -KI=6 -result_path=../result/KI.log"
python ./main.py -KI=6 -result_path='../result/KI.log'

echo "python main.py -KI=8 -result_path=../result/KI.log"
python ./main.py -KI=8 -result_path='../result/KI.log'
echo "python main.py -KI=8 -result_path=../result/KI.log"
python ./main.py -KI=8 -result_path='../result/KI.log'

echo "python main.py -KI=10 -result_path=../result/KI.log"
python ./main.py -KI=10 -result_path='../result/KI.log'
echo "python main.py -KI=10 -result_path=../result/KI.log"
python ./main.py -KI=10 -result_path='../result/KI.log'

#optimizer
echo "python main.py -optimizer=SGD -result_path=../result/optimizer.log"
python ./main.py -optimizer='SGD'-result_path='../result/optimizer.log'
echo "python main.py -optimizer=SGD -result_path=../result/optimizer.log"
python ./main.py -optimizer='SGD'-result_path='../result/optimizer.log'

echo "python main.py -optimizer=Adam -result_path=../result/optimizer.log"
python ./main.py -optimizer='Adam'-result_path='../result/optimizer.log'
echo "python main.py -optimizer=Adam -result_path=../result/optimizer.log"
python ./main.py -optimizer='Adam'-result_path='../result/optimizer.log'

echo "python main.py -optimizer=RMSProp -result_path=../result/optimizer.log"
python ./main.py -optimizer='RMSProp'-result_path='../result/optimizer.log'
echo "python main.py -optimizer=RMSProp -result_path=../result/optimizer.log"
python ./main.py -optimizer='RMSProp'-result_path='../result/optimizer.log'



