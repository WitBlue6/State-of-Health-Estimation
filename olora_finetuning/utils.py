import torch.nn as nn
import torch.nn.functional as F
import torch
from pytorch_wavelets import DWT1DForward
from transformers import PreTrainedModel, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import os

class A_cSE(nn.Module):
	
	def __init__(self, in_ch):
		super(A_cSE, self).__init__()
		
		self.conv0 = nn.Sequential(
			nn.Conv1d(in_ch, in_ch, kernel_size=3, padding=1),
			nn.BatchNorm1d(in_ch),
			nn.ReLU(inplace=True),
		)
		self.conv1 = nn.Sequential(
			nn.Conv1d(in_ch, int(in_ch/2), kernel_size=1, padding=0),
			nn.BatchNorm1d(int(in_ch/2)),
			nn.ReLU(inplace=True),
		)
		self.conv2 = nn.Sequential(
			nn.Conv1d(int(in_ch/2), in_ch, kernel_size=1, padding=0),
			nn.BatchNorm1d(in_ch)
		)
		
	def forward(self, in_x):
		
		x = self.conv0(in_x)
		x = nn.AvgPool1d(x.size()[2:])(x)
		#print('channel',x.size())
		x = self.conv1(x)
		x = self.conv2(x)
		x = torch.sigmoid(x)
		
		return in_x * x + in_x

class SConv_1D(nn.Module):
	'''(conv => BN => ReLU) * 2'''
	def __init__(self, in_ch, out_ch, kernel, pad):
		super(SConv_1D, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv1d(in_ch, out_ch, kernel, padding=pad),
			nn.GroupNorm(2, out_ch),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		
		x = self.conv(x)
		return x

class MWA_CNN(nn.Module):
	def __init__(self, input_dim=64, numf=4):
		super(MWA_CNN, self).__init__()
		
		self.input_dim = input_dim
		self.numf = numf
		self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
		
		self.pre_conv = nn.Sequential(
			nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2),
			nn.BatchNorm1d(16),
			nn.ReLU(inplace=True),
			nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
			nn.BatchNorm1d(32),
			nn.ReLU(inplace=True),
			# nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
			# nn.BatchNorm1d(64),
			# nn.ReLU(inplace=True)
		).to(self.device)
		
		self.DWT0= DWT1DForward(J=1, wave='db16').to(self.device)
		
		self.SConv1 = SConv_1D(input_dim, numf, 3, 0).to(self.device)
		self.DWT1= DWT1DForward(J=1, wave='db16').to(self.device)
		self.dropout1 = nn.Dropout(p=0.1).to(self.device)
		self.cSE1 = A_cSE(numf*2).to(self.device)
		
		self.SConv2 = SConv_1D(numf*2, numf*2, 3, 0).to(self.device)
		self.DWT2= DWT1DForward(J=1, wave='db16').to(self.device)
		self.dropout2 = nn.Dropout(p=0.1).to(self.device)
		self.cSE2 = A_cSE(numf*4).to(self.device)
		
		self.SConv3 = SConv_1D(numf*4, numf*4, 3, 0).to(self.device)
		self.DWT3= DWT1DForward(J=1, wave='db16').to(self.device)
		self.dropout3 = nn.Dropout(p=0.1).to(self.device)
		self.cSE3 = A_cSE(numf*8).to(self.device)
		
		self.SConv4 = SConv_1D(numf*8, numf*8, 3, 0).to(self.device)
		self.DWT4= DWT1DForward(J=1, wave='db16').to(self.device)
		self.dropout4 = nn.Dropout(p=0.1).to(self.device)
		self.cSE4 = A_cSE(numf*16).to(self.device)
		
		self.SConv5 = SConv_1D(numf*16, numf*16, 3, 0).to(self.device)			
		self.DWT5= DWT1DForward(J=1, wave='db16').to(self.device)
		self.dropout5 = nn.Dropout(p=0.1).to(self.device)
		self.cSE5 = A_cSE(numf*32).to(self.device)
		self.SConv6 = SConv_1D(numf*32, numf*32, 3, 0).to(self.device)				
		
		self.avg_pool = nn.AdaptiveAvgPool1d((1)).to(self.device)
		self.fc = nn.Linear(numf*32, 6).to(self.device)
	
	def update(self, input_dim):
		self.input_dim = input_dim
		self.SConv1 = SConv_1D(input_dim, self.numf, 3, 0)
		
	def forward(self, input):
		#print(f"原始 input 形状: {input.shape}")  # Debug
		# 先通过卷积降维
		# add by lzh
		input = self.pre_conv(input.permute(0, 2, 1))  # (batch, 32, input_dim)
		# 这里macbook MPS无法跑，只能换成CPU
		input = F.adaptive_avg_pool1d(input.to(torch.device("cpu")), self.input_dim)
		input = input.to(self.device)
		#print(input.shape) # Debug
		DMT_yl, DMT_yh = self.DWT0(input)
		output = torch.cat([DMT_yl, DMT_yh[0]], dim=1)
		output = self.SConv1(output)
		#print(f"降维后: {output.shape}")  # Debug
		DMT_yl, DMT_yh = self.DWT1(output)
		output = torch.cat([DMT_yl, DMT_yh[0]], dim=1)
		output = self.dropout1(output)
		output = self.cSE1(output)
		
		output = self.SConv2(output)
		DMT_yl, DMT_yh = self.DWT2(output)
		output = torch.cat([DMT_yl, DMT_yh[0]], dim=1)
		output = self.dropout2(output)
		output = self.cSE2(output)
		
		output = self.SConv3(output)
		DMT_yl,DMT_yh = self.DWT3(output)
		output = torch.cat([DMT_yl,DMT_yh[0]], dim=1) 
		output = self.dropout3(output)
		output = self.cSE3(output)
		
		output = self.SConv4(output)
		DMT_yl,DMT_yh = self.DWT4(output)
		output = torch.cat([DMT_yl,DMT_yh[0]], dim=1) 
		output = self.dropout4(output)
		output = self.cSE4(output)
		
		output = self.SConv5(output)
		DMT_yl,DMT_yh = self.DWT5(output)
		output = torch.cat([DMT_yl,DMT_yh[0]], dim=1)	 
		output = self.dropout5(output)
		output = self.cSE5(output)
		
		output = self.SConv6(output)			 
			
		output = self.avg_pool(output)
		output = output.view(output.size(0), -1)
		output = self.fc(output)
		
		return output

class TransformerWithCNN(PreTrainedModel):
	def __init__(self, transformer, cnn_model, config):
		super().__init__(config)
		self.transformer = transformer
		self.cnn = cnn_model

	def forward(self, input_ids, attention_mask=None, labels=None, inputs_embeds=None, generate_reason=False, **kwargs):
		transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask, **kwargs)
		# 获取 logits（一般是 CausalLMOutputWithPast 对象的一部分）
		logits = transformer_outputs.logits
		hidden_states = logits[:, -1, :]  # 获取序列中的最后一个时间步的 logits
		# 将隐藏状态传递给 CNN 层
		cnn_output = self.cnn(hidden_states.unsqueeze(2))  # 需要调整形状为 (batch_size, hidden_dim, 1)
		loss = None
		if labels is not None:
			#print(labels)
			#exit()
			# 如果标签存在，计算损失
			loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
			#print(1)
			shift_logits = logits[..., :-1, :].contiguous()
			shift_labels = labels[..., 1:].contiguous()
			#print(2)
			loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
			#loss = loss_fct(cnn_output, labels)
			#print('hellow')
			#exit()
		return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions
        )
	def generate(self, *args, **kwargs):
        # 直接调用底层 transformer 模型的 generate 方法
		return self.transformer.generate(*args, **kwargs)
	
	def save_pretrained(self, save_directory: str, **kwargs):
		# 保存cnn状态字典
		print('保存CNN状态字典')
		torch.save(self.cnn.state_dict(), os.path.join(save_directory, "cnn_model.pth"))
		# 这里调用父类的 save_pretrained 方法来保存模型
		# super().save_pretrained(save_directory, **kwargs)
		
	
	@classmethod
	def from_pretrained(cls, pretrained_model_name_or_path: str, cnn_model, config, **kwargs):
		# 加载预训练的 Transformer 模型
		transformer = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)
		return cls(transformer, cnn_model, config)
	
	def state_dict(self, *args, **kwargs):
		original_state_dict = super().state_dict(*args, **kwargs)
		new_state_dict = {}
        
		for key, value in original_state_dict.items():
            # 处理三种可能的键名前缀情况
			if key.startswith("base_model.model.transformer.gpt_neox."):
				new_key = key.replace("base_model.model.transformer.gpt_neox.", "gpt_neox.")
			elif key.startswith("transformer.gpt_neox."):
				new_key = key.replace("transformer.gpt_neox.", "gpt_neox.")
			elif key.startswith("base_model.model.gpt_neox."):
				new_key = key.replace("base_model.model.gpt_neox.", "gpt_neox.")
			else:
				new_key = key
			new_state_dict[new_key] = value
		return new_state_dict

def fix_lora_keys(model):
	state_dict = model.state_dict()
	new_state_dict = {}
	for k, v in state_dict.items():
        # 修复所有可能的键名前缀情况
		 # 修复所有可能的键名前缀
		if k.startswith("base_model.model.transformer.gpt_neox."):
			new_k = k.replace("base_model.model.transformer.gpt_neox.", "gpt_neox.")
		elif k.startswith("transformer.gpt_neox."):
			new_k = k.replace("transformer.gpt_neox.", "gpt_neox.")
		elif k.startswith("base_model.model.gpt_neox."):
			new_k = k.replace("base_model.model.gpt_neox.", "gpt_neox.")
		elif k.startswith("base_model.model.cnn."):  # 这个条件重复了，可以删除
			new_k = k.replace("base_model.model.cnn.", "cnn.")
		new_state_dict[new_k] = v
	model.load_state_dict(new_state_dict, strict=False)
	return model