import re
import os
import matplotlib.pyplot as plt
# Lấy đường dẫn đến thư mục
path = "./save_result/cifar10_krum_flame_daba"

# Lấy danh sách các tệp và thư mục trong thư mục
files = os.listdir(path)

# Lấy hết tên file trong thư mục

plt.figure(figsize=(20,45))

dataset_draw = "cifar"

def check( parama):
    if parama == '0.1' or parama == '0.2':
        return True
    return False

for file in files:
    if os.path.isfile(os.path.join(path, file)):
        ext = os.path.splitext(file)[1]
        first = os.path.splitext(file)[0]
        if ext == ".txt" and len(first) > 2:
            path_file = os.path.join(path, file)
            with open(path_file, "r") as f:
                content = f.read()
            # Trích xuất các giá trị trong main_task_accuracy và backdoor_accuracy
                main_task_accuracy = []
                backdoor_accuracy = []
                long_attack = 0
                wide_attack = 0
                style_send = ''
                threshold_reject = ''
                threshold_low = ''
                defense =''
                attack_method = ''
                dataset = ''
                fract_noniid = 0
                Fraction_attack = ''
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
                    if line.startswith("    Threshold down:"):
                        threshold_low = line.strip().split(": ")[1]+''
                    if line.startswith("    Fraction NonIID:"):
                        fract_noniid = line.strip().split(": ")[1]+''
                    if line.startswith("    Attack method:"):
                        attack_method = line.strip().split(": ")[1] +''
                    if line.startswith("    Fraction of malicious agents:"):
                        Fraction_attack = line.strip().split(": ")[1]+''
                    if line.startswith("    Long attack:"):
                        long_attack = line.strip().split(": ")[1] +''
                    if line.startswith("    Wide attack:"):
                        wide_attack = line.strip().split(": ")[1] +''
                    if line.startswith("    Style send:"):
                        style_send = line.strip().split(": ")[1] +''
                    if line.startswith("main_task_accuracy="):
                        main_task_accuracy = [float(value) for value in line.strip().split("[")[1].split("]")[0].split(",")]
                    if line.startswith('loss_list='):
                        test_loss = [float(value) for value in line.strip().split("[")[1].split("]")[0].split(", ")]
                    if line.startswith("backdoor_accuracy="):
                        backdoor_accuracy = [float(value) for value in line.strip().split("[")[1].split("]")[0].split(",")]
            size_line = 0.5
            set_attack_method = 'badnet'
            if iid == '0':
                # if defense == 'mr_duc' and threshold_reject != '' and check(threshold_low) and Fraction_attack =='16.0%':
                if dataset == dataset_draw and attack_method == set_attack_method :
                    print(long_attack, wide_attack)
                    # if defense == 'RLR' or defense == 'zkp' or defense == 'flame':
                    if long_attack == '3' and wide_attack == '3':
                        plt.subplot(621)
                        plt.plot(main_task_accuracy, label = defense, linewidth = size_line)
                        plt.xlabel("Size trigger attack:" + long_attack + 'x' + wide_attack)
                        plt.ylabel('main accuracy')
                        plt.legend()
                        plt.subplot(622)
                        plt.plot(backdoor_accuracy, label = defense, linewidth = size_line)
                        plt.xlabel("Size trigger attack:" + long_attack + 'x' + wide_attack)
                        plt.ylabel('backdoor accuracy')
                        plt.legend()
                    elif long_attack == '4' and wide_attack == '4':
                        plt.subplot(623)
                        plt.plot(main_task_accuracy, label = defense, linewidth = size_line)
                        plt.xlabel("Size trigger attack:" + long_attack + 'x' + wide_attack)
                        plt.ylabel('main accuracy')
                        plt.legend()
                        plt.subplot(624)
                        plt.plot(backdoor_accuracy, label = defense, linewidth = size_line)
                        plt.xlabel("Size trigger attack:" + long_attack + 'x' + wide_attack)
                        plt.ylabel('backdoor accuracy')
                        plt.legend()
                    elif long_attack == '5' and wide_attack == '5':
                        plt.subplot(625)
                        plt.plot(main_task_accuracy, label = defense, linewidth = size_line)
                        plt.xlabel("Size trigger attack:" + long_attack + 'x' + wide_attack)
                        plt.ylabel('main accuracy')
                        plt.legend()
                        plt.subplot(626)
                        plt.plot(backdoor_accuracy, label = defense, linewidth = size_line)
                        plt.xlabel("Size trigger attack:" + long_attack + 'x' + wide_attack)
                        plt.ylabel('backdoor accuracy')
                        plt.legend()
                # elif dataset == dataset_draw and attack_method == 'dba' and long_attack == long and wide_attack == wide:
                #     # if defense == 'RLR' or defense == 'zkp' or defense == 'flame':
                #     plt.subplot(423)
                #     plt.plot(main_task_accuracy, label = style_send, linewidth = size_line)
                #     plt.xlabel(dataset +' '+ attack_method +' '+ long_attack +'x' + wide_attack)
                #     plt.ylabel('main accuracy')
                #     plt.legend()
                #     plt.subplot(424)
                #     plt.plot(backdoor_accuracy, label = style_send, linewidth = size_line)
                #     plt.xlabel(dataset +' '+ attack_method +' '+ long_attack +'x' + wide_attack)
                #     plt.ylabel('backdoor accuracy')
                #     plt.legend()
                # elif dataset == dataset_draw and attack_method == 'badnet' and long_attack == '5' and wide_attack == '5' and fract_noniid == frac_data:
                #     # if defense == 'RLR' or defense == 'zkp' or defense == 'flame':
                #     plt.subplot(625)
                #     plt.plot(main_task_accuracy, label = style_send, linewidth = size_line)
                #     plt.xlabel(dataset +' '+ attack_method +' '+ long_attack +'x' + wide_attack)
                #     plt.ylabel('main accuracy')
                #     plt.legend()
                #     plt.subplot(626)
                #     plt.plot(backdoor_accuracy, label = style_send, linewidth = size_line)
                #     plt.xlabel(dataset +' '+ attack_method +' '+ long_attack +'x' + wide_attack)
                #     plt.ylabel('backdoor accuracy')
                #     plt.legend()
                # if dataset == dataset_draw and attack_method == 'dba' and long_attack == '5' and wide_attack == '5' and fract_noniid == frac_data:
                #     # if defense == 'RLR' or defense == 'zkp' or defense == 'flame':
                #     plt.subplot(627)
                #     plt.plot(main_task_accuracy, label = style_send, linewidth = size_line)
                #     plt.xlabel(dataset +' '+ attack_method +' '+ long_attack +'x' + wide_attack)
                #     plt.ylabel('main accuracy')
                #     plt.legend()
                #     plt.subplot(628)
                #     plt.plot(backdoor_accuracy, label = style_send, linewidth = size_line)
                #     plt.xlabel(dataset +' '+ attack_method +' '+ long_attack +'x' + wide_attack)
                #     plt.ylabel('backdoor accuracy')
                #     plt.legend()
                # elif dataset == dataset_draw and attack_method == 'badnet' and long_attack == '5' and wide_attack == '5' and fract_noniid == frac_data:
                #     # if defense == 'RLR' or defense == 'zkp' or defense == 'flame':
                #     plt.subplot(629)
                #     plt.plot(main_task_accuracy, label = style_send, linewidth = size_line)
                #     plt.xlabel(dataset +' '+ attack_method +' '+ long_attack +'x' + wide_attack)
                #     plt.ylabel('main accuracy')
                #     plt.legend()
                #     plt.subplot(6210)
                #     plt.plot(backdoor_accuracy, label = style_send, linewidth = size_line)
                #     plt.xlabel(dataset +' '+ attack_method +' '+ long_attack +'x' + wide_attack)
                #     plt.ylabel('backdoor accuracy')
                #     plt.legend()
                # if dataset == dataset_draw and attack_method == 'dba' and long_attack == '5' and wide_attack == '5' and fract_noniid == frac_data:
                #     # if defense == 'RLR' or defense == 'zkp' or defense == 'flame':
                #     plt.subplot(6211)
                #     plt.plot(main_task_accuracy, label = style_send, linewidth = size_line)
                #     plt.xlabel(dataset +' '+ attack_method +' '+ long_attack +'x' + wide_attack)
                #     plt.ylabel('main accuracy')
                #     plt.legend()
                #     plt.subplot(6212)
                #     plt.plot(backdoor_accuracy, label = style_send, linewidth = size_line)
                #     plt.xlabel(dataset +' '+ attack_method +' '+ long_attack +'x' + wide_attack)
                #     plt.ylabel('backdoor accuracy')
                #     plt.legend()
        
plt.savefig('../FL/test432''.pdf', format = 'pdf',bbox_inches='tight')