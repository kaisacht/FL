import re
import os
import matplotlib.pyplot as plt
# Lấy đường dẫn đến thư mục
path = "./save"

# Lấy danh sách các tệp và thư mục trong thư mục
files = os.listdir(path)

# Lấy hết tên file trong thư mục

plt.figure(figsize=(20,20))


for file in files:
    if os.path.isfile(os.path.join(path, file)):
        ext = os.path.splitext(file)[1]
        first = os.path.splitext(file)[0]
        if ext == ".txt" and len(first) > 10:
            path_file = os.path.join(path, file)
            with open(path_file, "r") as f:
                content = f.read()
            # Trích xuất các giá trị trong main_task_accuracy và backdoor_accuracy
                main_task_accuracy = []
                backdoor_accuracy = []
                long_attack = 0
                wide_attack = 0
                threshold_reject = ''
                defense =''
                attack_method = ''
                dataset = ''
                lines = content.split("\n")
                
                for line in lines:
                    if line.startswith("    Aggregation Function:"):
                        defense = line.strip().split(": ")[1] +''
                    if line.startswith("    IID:"):
                        iid = line.strip().split(": ")[1] +''
                    if line.startswith("    Dataset:"):
                        dataset = line.strip().split(": ")[1] + ''
                    if line.startswith("    Threshold reject:"):
                        threshold_reject = line.strip().split(": ")[1]+''
                    if line.startswith("    Attack method:"):
                        attack_method = line.strip().split(": ")[1] +''
                    if line.startswith("    Long attack:"):
                        long_attack = line.strip().split(": ")[1] +''
                    if line.startswith("    Wide attack:"):
                        wide_attack = line.strip().split(": ")[1] +''
                    if line.startswith("main_task_accuracy="):
                        main_task_accuracy = [float(value) for value in line.strip().split("[")[1].split("]")[0].split(", ")]
                    if line.startswith('loss_list='):
                        test_loss = [float(value) for value in line.strip().split("[")[1].split("]")[0].split(", ")]
                    if line.startswith("backdoor_accuracy="):
                        backdoor_accuracy = [float(value) for value in line.strip().split("[")[1].split("]")[0].split(", ")]
            size_line = 1.
            if iid == '0':
                if dataset == 'fashion_mnist' and attack_method == 'badnet' and long_attack == '3' and wide_attack == '3':
                    if defense == 'mr_duc' and threshold_reject != '':
                    # if defense == 'RLR' or defense == 'mr_duc':
                        plt.subplot(321)
                        plt.plot(main_task_accuracy, label = "threshold_reject = " + threshold_reject, linewidth = size_line)
                        plt.xlabel(dataset +' '+ attack_method +' '+ long_attack +'x' + wide_attack)
                        plt.ylabel('main accuracy')
                        plt.legend()
                        plt.subplot(322)
                        plt.plot(backdoor_accuracy, label =  "threshold_reject = "+threshold_reject, linewidth = size_line)
                        plt.xlabel(dataset +' '+ attack_method +' '+ long_attack +'x' + wide_attack)
                        plt.ylabel('backdoor accuracy')
                        plt.legend()
                elif  dataset == 'fashion_mnist' and attack_method == 'badnet' and long_attack == '5' and wide_attack == '5':
                    if defense == 'mr_duc' and threshold_reject !='':
                    # if defense == 'RLR' or defense == 'mr_duc':
                        plt.subplot(323)
                        plt.plot(main_task_accuracy, label =  "threshold_reject = "+ threshold_reject, linewidth = size_line)
                        plt.xlabel(dataset +' '+ attack_method +' '+ long_attack +'x' + wide_attack)
                        plt.ylabel('main accuracy')
                        plt.legend()
                        plt.subplot(324)
                        plt.plot(backdoor_accuracy, label =  "threshold_reject = "+threshold_reject, linewidth = size_line)
                        plt.xlabel(dataset +' '+ attack_method +' '+ long_attack +'x' + wide_attack)
                        plt.ylabel('backdoor accuracy')
                        plt.legend()
                elif  dataset == 'fashion_mnist' and attack_method == 'dba' and long_attack == '5' and wide_attack == '5':
                    # if defense == 'RLR' or defense == 'mr_duc':
                    if defense == 'mr_duc' and threshold_reject !='':
                        plt.subplot(325)
                        plt.plot(main_task_accuracy, label =  "threshold_reject = "+ threshold_reject, linewidth = size_line)
                        plt.xlabel(dataset +' '+ attack_method +' '+ long_attack +'x' + wide_attack)
                        plt.ylabel('main accuracy')
                        plt.legend()
                        plt.subplot(326)
                        plt.plot(backdoor_accuracy, label =  "threshold_reject = "+threshold_reject, linewidth = size_line)
                        plt.xlabel(dataset +' '+ attack_method +' '+ long_attack +'x' + wide_attack)
                        plt.ylabel('backdoor accuracy')
                        plt.legend()
plt.savefig('../FL/save/threshold_reject_fmnist.pdf', format = 'pdf',bbox_inches='tight')