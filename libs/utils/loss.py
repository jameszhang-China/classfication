import torch
import torch.nn as nn
from libs.engine.tasker import ClassificationModel
from libs.utils.utils import ModelEMA
import torch.nn.functional as F
'''
在 PyTorch 中，parameters() 和 state_dict() 返回的内容有本质区别： 
model.parameters() 返回什么？
返回类型：生成器（generator）
内容：模型所有可学习参数（nn.Parameter 类型），即：
权重（weight）
偏置（bias）
不包含：
不可学习的缓冲区（如 BatchNorm 的 running_mean）
优化器状态
其他非参数属性

state_dict() 方法返回一个 有序字典（OrderedDict），其中包含模型的所有可学习参数（如权重和偏置）以及持久缓冲区（如 BatchNorm 的 running_mean）
如果目标是严格对齐层结构，应使用 state_dict()（因为它包含完整的层名称映射）。
如果只需要检查可学习参数，可以用 parameters()，但会丢失层名称信息。

在 PyTorch 中，named_modules() 是一个用于递归遍历模型所有子模块（包括自身）的方法，返回每个模块的名称及其引用
for name, module in model.named_modules():
    print(f"Name: {name}, Type: {type(module)}")
包含所有子模块：从根模块（模型自身）到最底层的叶子模块（如 nn.Conv2d）
包含自身：第一个迭代项是模型本身（name='' 或自定义名称）
递归遍历：自动展开嵌套结构（如 nn.Sequential、自定义模块）

named_parameters() 是一个用于递归遍历模型所有可学习参数（即 nn.Parameter 类型）的方法，返回每个参数的名称及其引用
for name, param in model.named_parameters():
    print(f"Name: {name}, Shape: {param.shape}")
包含所有可学习参数：从根模块（模型自身）到最底层的叶子参数（如 nn.Conv2d.weight）
遍历的是所有参数的扁平列表，不管这个参数属于哪个层、哪个模块，全部平铺展示；
参数名虽然也带层级（比如model.backbone.conv1.weight），但只是名字里的字符串，无法直接筛选「模块」，只能筛选「参数名是否含 weight/bias」；
例：YOLOv8 中，named_parameters()只会返回 model.backbone.conv1.weight、model.backbone.conv1.bias、model.neck.conv2.weight 这类参数，没有模块的概念。

使用 named_children()（仅直接子层）

叶子节点：无上游依赖（如模型输入、模型参数），is_leaf=True；
非叶子节点：由上游节点运算生成（如卷积输出、x*2），is_leaf=False
边：运算（如 +、×、conv2d），记录 “如何从上游节点得到下游节点” 的规则（用于反向传播算梯度）。
y 的开关关了 → 不计算d_loss/d_y（y.grad=None）；
但反向传播仍会用 “局部梯度”（y=2x 的局部梯度是 2，z=3y 的局部梯度是 3），直接计算d_loss/d_x = d_loss/d_z × 3 × 2 = 1×3×2=6；
梯度跳过了 y 的梯度计算，但完整传递给了上游的 x —— 这就是 MGD 中冻结 1×1 卷积后，梯度仍能传给学生特征的核心逻辑！
requires_grad = True：反向传播时，会计算该节点的梯度，且满足条件时（叶子节点 / 手动retain_grad()）会把梯度存储到.grad属性；requires_grad = False：反向传播时，不会为该节点保留梯度流程
对叶子节点（如卷积权重）：既不计算、也不存储，直接截断梯度；
对非叶子节点（如卷积输出）：为了上游节点，会临时计算梯度，但不存储，用完就丢。只有关闭输入端叶子节点（如 x） 的梯度，才会彻底截断；关闭任何非叶子节点的梯度，都只是 “不存储中间梯度”，梯度仍会传给上游。
conv.weight的梯度截断（requires_grad=False），截断的是 **“conv.weight 自身的梯度计算 / 存储”，但梯度流的主线（loss→卷积输出→输入 / 上一层特征）** 完全不受影响 —— 因为计算上一层的梯度，不需要 conv.weight 的梯度，只需要 conv.weight 的数值
卷积层（模块）的requires_grad是批量控制其所有参数 的快捷开关 —— 修改模块的requires_grad，会同步修改其所有参数的requires_grad；反之，也可单独修改某一个参数（如weight）的requires_grad，不影响其他参数（如bias）。
x → conv1 → conv2 → conv3 → y 中，如果所有 conv 层的 requires_grad=False，但输入 x 的 requires_grad=True，x 仍然能拿到完整的梯度

随机初始化的权重可能让学生特征图的 “猫脸” 特征，映射到教师特征图时既包含猫脸又混进背景；
正交初始化的权重会让 “猫脸” 特征精准映射到教师的对应通道，背景特征也独立映射，互不重叠。

PyTorch 创建新nn.Conv2d时，weight 默认用kaiming_uniform_（凯明均匀初始化），bias（若开启）用均匀初始化

1. 常数初始化的核心原理（贴合你的 s-fm→conv→t-fm 场景）
对于 1×1 卷积，输出特征图的计算逻辑是：t_fm = conv.weight * s_fm + conv.bias；
冻结时conv.weight固定，设为 1/t_c 能保证：
降维场景（s_c > t_c，如 s_c=512→t_c=256）：学生通道的权重被均匀分配到教师通道，无单通道特征占比过高 / 过低；
升维场景（s_c < t_c，如 s_c=256→t_c=512）：调整为 1/s_c，保证学生通道权重复制后总权重和为 1，特征缩放稳定；
无随机因素：冻结后权重不再更新，常数初始化能避免 “随机权重导致部分教师通道学不到学生特征” 的问题。
2. 常规初始化的致命问题（冻结时）
常规初始化（Kaiming/Xavier）的权重是随机均匀 / 正态分布，比如：
部分权重可能接近 0 → 对应教师通道 “丢失” 学生特征；
部分权重可能过大 → 对应教师通道 “过度放大” 学生特征；
冻结后无法通过训练修正这些问题，最终蒸馏精度会比常数初始化低 1~2%（NPU 量化后差距更大）。
'''

'''
温度系数：使得率分布会变得更「平滑」，学生更容易拟合（这是温度 T 的作用）；KL 散度的损失值会被缩小 (T^2) 倍，损失值会变得特别小，几乎对总损失没有贡献
乘以 (T^2) 的目的：把损失值的量级恢复到原来的大小，让蒸馏损失的权重和学生的分类损失「匹配」，既不会被分类损失淹没，也不会喧宾夺主。
教师禁用 log_softmax：会数值溢出、梯度消失，教师只需要输出稳定的概率分布即可；
学生必用 log_softmax：梯度更大、训练更高效，完美匹配 KL 散度的数学公式；
必乘(T^2)：恢复损失量级，保证蒸馏损失的权重合理

学生模型是需要训练、需要梯度回传、需要不断更新参数的，用log_softmax对学生来说有一下好处：
1：解决梯度消失问题，优化训练效率；log_softmax 相比于 softmax，在反向传播时的梯度表达式更简洁、梯度值更大，能有效避免梯度消失
教师模型是「冻结的、无梯度的」，用 log_softmax 会产生数值灾难
'''

class MGDDecoder(nn.Module):
    def __init__(self, teacher_channels):
        super(MGDDecoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, teacher_feature):
        return self.decoder(teacher_feature)

class DistillationLoss(nn.Module):
    def __init__(self, args, student, device='cuda'):
        '''
        teacher: teacher weight or checkpoint path
        args.t_model: teacher model yaml path
        '''
        super(DistillationLoss, self).__init__()
        self.args = args
        self.student = student
        self.device = device

        if not self.args.distillation:
            return
        teacher = self.args.teacher
        if not teacher:
            raise ValueError("Teacher model is not defined")
        teacher = torch.load(teacher, map_location=device, weights_only=False)
        if isinstance(teacher, dict):
            model = teacher.get('ema', None) if 'ema' in teacher else teacher.get('model', None)
            if type(model) is ModelEMA:
                model = model.ema
            if isinstance(teacher, dict) and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in teacher.items()):
                # weights
                model = ClassificationModel(self.args.t_model, self.args, device=self.device)
                model.load_state_dict(teacher)
                teacher = model
            elif hasattr(model, 'forward') and hasattr(model, 'state_dict'):
                # full model
                teacher = model.to(self.device)
            else:
                raise ValueError('teacher model must be a weight or a model checkpoint')
        elif isinstance(self.model, ClassificationModel):
            pass
        else:
            raise ValueError('teacher model must be a weight or ClassificationModel instance')
        self.align = nn.ModuleList() # 这样使用局部conv append就会被train到
        self.teacher = teacher.to(self.device)
        if self.args.distillation_loss_type == 'MGD':
            self.mgd_decoder = nn.ModuleList()
        
    def _align_student_teacher(self, s_fm, t_fm):
        # check student and teacher model structure
        for i, ((m_name, m), (n_name, n)) in enumerate(zip(self.student.named_children(), self.teacher.named_children())):
            if type(m) is not type(n):
                raise ValueError(
                    f"Layer[{i}] type mismatch:\n"
                    f"Student: {m_name}\n"
                    f"Teacher: {n_name}"
                )
        # add align layer
        if len(self.align) == 0:
            for i in self.args.study_list:
                s_c = s_fm[i].shape[1]
                t_c = t_fm[i].shape[1]
                allowed_ratios = {0.5, 1, 2}
                w_ratio = s_fm[i].shape[2]//t_fm[i].shape[2]
                h_ratio = s_fm[i].shape[3]//t_fm[i].shape[3]
                if w_ratio != h_ratio:
                    raise ValueError(
                        f"Layer[{i}] shape mismatch:\n"
                        f"Student: {s_fm[i].shape}\n"
                        f"Teacher: {t_fm[i].shape}"
                    )
                ratio = w_ratio
                if ratio not in allowed_ratios: # 非冻结
                    if ratio > 1:
                        conv = nn.Conv2d(s_c, t_c, kernel_size=3, padding=1, stride=ratio, bias=False)
                    elif ratio < 1: 
                        conv = nn.ConvTranspose2d(s_c, t_c, kernel_size=3, padding=1, stride=1/ratio, bias=False)
                    nn.init.orthogonal_(conv.weight) # 正交初始化加速收敛，正交初始化是解决非整数倍通道比下特征映射紊乱的核心方法
                    self.align.append(conv)
                else: # 冻结
                    scale = nn.Identity()
                    conv = nn.Identity()
                    if ratio != 1:
                        # 优先平均池化，能保留特征图的全局语义分布，更贴合蒸馏 “学习教师特征分布” 的核心目标；最大池化仅在「小目标 / 纹理主导的检测场景」下少量使用
                        scale = nn.Upsample(scale_factor=ratio, mode='bilinear', align_corners=True) if ratio < 1 \
                            else nn.AvgPool2d(kernel_size=ratio, stride=ratio, padding=0) if ratio > 1 \
                            else nn.Identity()
                    if s_c != t_c:
                        conv = nn.Conv2d(s_c, t_c, kernel_size=1, bias=False, requires_grad=False)
                        nn.init.constant_(conv.weight, 1.0 / t_c) # 无偏常数初始化，初始化逻辑同理：每个输入通道均匀分配到多个输出通道，比如 1 个输入通道对应 2 个输出通道，权重设为1/128，保证输出通道是输入通道的均值扩展
                    self.align.append(nn.Sequential(scale, conv)) # 全新创建的实例，不需要copy
        s_fm_aligned = [self.align[i](s_fm[j]).clone() for i, j in enumerate(self.args.study_list)] # 局部conv forward
        t_fm_aligned = [t_fm[i].clone() for i in self.args.study_list]
        return s_fm_aligned, t_fm_aligned

    def _teacher_inference(self, x):
        self.teacher.eval()
        with torch.no_grad(): 
            out = self.teacher(x)
        fm = self.teacher.get_fm()
        return fm
        
    def forward(self, student_fm, x):
        if not self.args.distillation:
            return 0.0
        if not self.args.study_list:
            return 0.0
        t_fm = self._teacher_inference(x)
        s_fm_aligned, t_fm_aligned = self._align_student_teacher(student_fm, t_fm)
        return self._loss(s_fm_aligned, t_fm_aligned)
    
    def _mgd_loss(self, s_fm, t_fm):
        total_loss = 0.0
        if self.args.distillation_loss_type == 'MGD':
            if len(self.mgd_decoder) == 0:
                for s, t in zip(s_fm, t_fm):
                    self.mgd_decoder.append(MGDDecoder(in_channels=t.shape[1]))

        for i, (s, t) in enumerate(zip(s_fm, t_fm)):
            if self.args.distillation_loss_type == 'MGD':
                mask = (torch.rand_like(s) > self.args.mgd_mask_ratio).float()
                s = s * mask
                s = self.mgd_decoder[i](s)
            loss = F.mse_loss(s, t)
            total_loss += loss

        return total_loss / len(s_fm)

    '''
    CWD 的设计思路：只让学生学习「教师认为哪些通道重要、哪些通道不重要」的通道注意力分布
    '''
    def _cwd_loss(self, s_fm, t_fm):
        total_loss = 0.0
        temp = self.args.temperature
        
        for i, (s, t) in enumerate(zip(s_fm, t_fm)):                
            s_gap = F.adaptive_avg_pool2d(s, 1).squeeze()  # (B, C)提取通道注意力,是 CWD 的灵魂
            t_gap = F.adaptive_avg_pool2d(t, 1).squeeze()   # (B, C)
                     
            #学生log_softmax + 教师softmax 固定搭配
            s_att = F.log_softmax(s_gap / temp, dim=1)
            t_att = F.softmax(t_gap / temp, dim=1)
                        
            cwd_loss = F.kl_div(s_att, t_att, reduction='batchmean') * (temp ** 2) #KL散度 + 乘T² 恢复量级
            total_loss += cwd_loss
        return total_loss / len(self.args.study_list)

    def _ofm_loss(self, s_fm, t_fm):
        s_soft = F.log_softmax(s_fm / self.args.temperature, dim=1) #1. KL散度需要对数概率 2. 数值稳定性
        
        with torch.no_grad():
            t_soft = F.softmax(t_fm / self.args.temperature, dim=1)#1. 产生概率分布作为软标签 2. 直观的概率解释
        
        kl_loss = F.kl_div(s_soft, t_soft, reduction='batchmean')
        
        return kl_loss * (self.args.temperature ** 2)
        

    def _loss(self, s_fm, t_fm):
        if self.args.distillation_loss_type in ['MGD', 'KD']:
            internal_loss = self._mgd_loss(s_fm, t_fm)
        else:  # 'cwd'
            internal_loss = self._cwd_loss(s_fm, t_fm)

        output_loss = self._ofm_loss(s_fm, t_fm)
        
        total_loss = (self.args.feature_loss_weight * internal_loss +
                     self.args.output_loss_weight * output_loss)
        return total_loss
        

class ClassificationLoss:
    def __init__(self, args):
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
    def __call__(self, preds, targets):
        return self.criterion(preds, targets)
    
class SPARSELoss:
    def __init__(self, args, device='cuda'):
        self.args = args
        self.device = device
    def __call__(self, model):
        if not self.args.sparse_train:
            return 0.0
        l1_loss = torch.tensor(0., device=self.device)
        for i, (name, block) in enumerate(model.named_children()):
            if i in self.args.sparse_list:
                for name, param in block.named_parameters():
                    if 'weight' in name and 'conv' in name:
                        '''
                        torch.norm 是 PyTorch 中用于计算范数的函数
                        p=2: L2 范数（欧几里得范数）。最常用，计算向量的几何长度。
                        p=1: L1 范数。向量元素绝对值之和（曼哈顿距离）。
                        p=float('inf'): 无穷范数。向量元素绝对值的最大值。
                        p=0: 向量中非零元素的个数（有时用于稀疏性，但 PyTorch 中通常用 sum(x != 0)）。
                        p='fro': Frobenius 范数。矩阵元素的平方和的平方根（类似于矩阵的 L2 范数）。
                        '''
                        l1_loss += torch.norm(param, p=1) # get L1 norm of weights for conv
        return l1_loss * self.args.sparse_weight