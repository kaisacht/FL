python3.10 main.py --dataset cifar --model mobilenetv2 --iid 1 --defence DABA --malicious 0.25 --attack dba 
python3.10 main.py --dataset cifar --model mobilenetv2 --iid 1 --defence avg --malicious 0.25 --attack dba 
python3.10 main.py --dataset cifar --model mobilenetv2 --iid 1 --defence RLR --malicious 0.25 --attack dba 
python3.10 main.py --dataset cifar --model mobilenetv2 --iid 1 --defence flame --malicious 0.25 --attack dba 
python3.10 main.py --dataset cifar --model mobilenetv2 --iid 1 --defence krum --malicious 0.25 --attack dba 
python3.10 main.py --dataset cifar --model mobilenetv2 --iid 1 --defence fltrust --malicious 0.25 --attack dba 