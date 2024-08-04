import argparse
import torch

# Khởi tạo ArgumentParser
parser = argparse.ArgumentParser(description='Your description here')

# Thêm đối số --gpu
parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")

# Phân tích các đối số dòng lệnh
args = parser.parse_args()

# Sử dụng args.gpu để thiết lập thiết bị (device)
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

print(f"Using device: {device}")
