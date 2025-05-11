import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from model import EnhancedAnimalClassifier
from datasets import create_datasets
from torch.cuda.amp import GradScaler, autocast

# #################### 配置部分 ####################
CHECKPOINT_DIR = os.path.abspath('./model/checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 训练参数
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 100
WEIGHT_DECAY = 1e-5
MIN_LR = 1e-6
WARMUP_EPOCHS = 5

# 保存策略
CHECKPOINT_STRATEGY = "best+epoch"
SAVE_EVERY_EPOCH = 5
EARLY_STOP_PATIENCE = 10

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

# #################### 工具函数 ####################
def save_checkpoint(model, optimizer, epoch, train_loader, val_acc, is_best=False):
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val_acc': val_acc,
        'classes': train_loader.dataset.class_names,
        'arch': model.__class__.__name__,
        'timestamp': datetime.now().isoformat()
    }
    
    filename = f"ckpt_epoch{epoch+1}_acc{val_acc:.2f}.pth"
    torch.save(state, os.path.join(CHECKPOINT_DIR, filename))
    
    if is_best:
        best_path = os.path.join(CHECKPOINT_DIR, "model_best.pth")
        torch.save(state, best_path)
        clean_old_checkpoints()

def clean_old_checkpoints(keep=3):
    ckpts = glob.glob(os.path.join(CHECKPOINT_DIR, "ckpt_epoch*.pth"))
    if len(ckpts) > keep:
        ckpts_sorted = sorted(
            ckpts,
            key=lambda x: float(x.split("_acc")[1].replace(".pth", "")),
            reverse=True
        )
        for old_ckpt in ckpts_sorted[keep:]:
            os.remove(old_ckpt)

def load_checkpoint(path, model, optimizer=None):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint

# #################### 训练逻辑 ####################
def validate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:  # 只解包2个值
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def get_optimizer(model):
    optimizer = optim.AdamW(model.parameters(), 
                          lr=LEARNING_RATE, 
                          weight_decay=WEIGHT_DECAY)
    return optimizer

def get_scheduler(optimizer, train_loader):
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
        final_div_factor=1000
    )
    return scheduler

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

def train_model():
    # 初始化数据
    train_set, val_set = create_datasets()
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, 
                            shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    
    # 初始化模型和优化器
    model = EnhancedAnimalClassifier(num_classes=len(train_set.class_names)).to(device)
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer, train_loader)
    criterion = FocalLoss()
    scaler = GradScaler()
    writer = SummaryWriter()
    
    # 训练状态跟踪
    best_val_acc = 0.0
    no_improve_epochs = 0
    
    print(f"Starting training on {device} with strategy '{CHECKPOINT_STRATEGY}'")
    print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")
    print(f"Model architecture:\n{model}")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):  # 只解包2个值
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 混合精度训练
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)  # 直接使用Focal Loss，无需手动加权
            
            # 梯度缩放和反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # 统计指标
            _, predicted = torch.max(outputs.data, 1)
            running_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            running_loss += loss.item()
            
            # 更新学习率
            scheduler.step()
            
            # 打印批次日志
            if batch_idx % 50 == 0:
                batch_acc = 100 * running_correct / total_samples
                print(f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_idx} | "
                      f"Loss: {running_loss/(batch_idx+1):.4f} | Acc: {batch_acc:.2f}% | "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 验证和保存
        val_acc = validate_model(model, val_loader)
        
        # TensorBoard记录
        writer.add_scalar('Loss/train', running_loss/len(train_loader), epoch)
        writer.add_scalar('Accuracy/train', 100*running_correct/total_samples, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 智能保存策略
        if "best" in CHECKPOINT_STRATEGY and val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, train_loader, val_acc, is_best=True)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if "epoch" in CHECKPOINT_STRATEGY and (epoch+1) % SAVE_EVERY_EPOCH == 0:
            save_checkpoint(model, optimizer, epoch, train_loader, val_acc)

        # 打印epoch日志
        print(f"Epoch {epoch+1} Summary | "
              f"Train Loss: {running_loss/len(train_loader):.4f} | "
              f"Train Acc: {100*running_correct/total_samples:.2f}% | "
              f"Val Acc: {val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 早停检查
        if no_improve_epochs >= EARLY_STOP_PATIENCE:
            print(f"No improvement for {EARLY_STOP_PATIENCE} epochs. Early stopping...")
            break

    writer.close()
    print(f"Training completed. Best Val Acc: {best_val_acc:.2f}%")

# #################### 主程序 ####################
if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise